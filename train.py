import argparse
import datetime
import json
import os
import time
import gc

import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm
from warpctc_pytorch import CTCLoss

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from data.utils import reduce_tensor, get_cer_wer
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns

VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') or ['0']

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=70, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--batch-norm-momentum', default=0.1, type=float, help='BatchNorm momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--silent', dest='silent', action='store_true', help='Turn off progress tracking per iteration')
parser.add_argument('--checkpoint', dest='checkpoint', action='store_true', help='Enables checkpoint saving of model')
parser.add_argument('--checkpoint-per-batch', default=0, type=int, help='Save checkpoint per batch. 0 means never save')
parser.add_argument('--visdom', dest='visdom', action='store_true', help='Turn on visdom graphing')
parser.add_argument('--tensorboard', dest='tensorboard', action='store_true', help='Turn on tensorboard graphing')
parser.add_argument('--log-dir', default='visualize/deepspeech_final', help='Location of tensorboard log')
parser.add_argument('--log-params', dest='log_params', action='store_true', help='Log parameter values and gradients')
parser.add_argument('--id', default='Deepspeech training', help='Identifier for visdom/tensorboard run')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--continue-from', default='', help='Continue from checkpoint model')
parser.add_argument('--finetune', dest='finetune', action='store_true',
                    help='Finetune the model from checkpoint "continue_from"')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--reverse-sort', dest='reverse_sort', action='store_true',
                    help='Turn off reverse ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:1550', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str, help='distributed backend')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--rank', default=0, type=int,
                    help='The rank of this process')
parser.add_argument('--gpu-rank', default=None,
                    help='If using distributed parallel for multi-gpu, sets the GPU for the process')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)


def to_np(x):
    return x.data.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def build_optimizer(args_, parameters_):
    # import aggmo
    # return aggmo.AggMo(model.parameters(), args_.lr, betas=[0, 0.6, 0.9])
    return torch.optim.SGD(parameters_, lr=args_.lr,
                           momentum=args_.momentum, nesterov=True)


viz = None


class PlotWindow:
    def __init__(self, title, suffix):
        self.loss_results = torch.Tensor(10000)
        self.cer_results = torch.Tensor(10000)
        self.wer_results = torch.Tensor(10000)
        self.epochs = torch.arange(1, 10000)
        self.viz_window = None

        global viz, tensorboard_writer
        hour_now = str(datetime.datetime.now()).split('.', 1)[0][:-3]

        self.opts = dict(title=title + ': ' + hour_now, ylabel='', xlabel=suffix, legend=['Loss', 'WER', 'CER'])
        self.opts['layoutopts'] = {'plotly': {'yaxis': {'type': 'log'}}}

        if args.visdom and is_leader:
            if viz is None:
                from visdom import Visdom
                viz = Visdom()

        if args.tensorboard and is_leader:
            os.makedirs(args.log_dir, exist_ok=True)
            from tensorboardX import SummaryWriter

            tensorboard_writer = SummaryWriter(args.log_dir)

    def plot_history(self, start_epoch, total_avg_loss):
        global viz, tensorboard_writer

        if is_leader and args.visdom:
            # Add previous scores to visdom graph
            x_axis = self.epochs[0:start_epoch]
            y_axis = torch.stack(
                (self.loss_results[0:start_epoch],
                 self.wer_results[0:start_epoch],
                 self.cer_results[0:start_epoch]),
                dim=1)
            self.viz_window = viz.line(
                X=x_axis,
                Y=y_axis,
                opts=self.opts,
            )
        if is_leader and args.tensorboard:
            # Previous scores to tensorboard logs
            for i in range(start_epoch):
                values = {
                    'Avg Train Loss': self.loss_results[i],
                    'Avg WER': self.wer_results[i],
                    'Avg CER': self.cer_results[i]
                }
                tensorboard_writer.add_scalars(args.id, values, i + 1)

    def plot_progress(self, epoch, avg_loss, cer_avg, wer_avg):
        global viz, tensorboard_writer

        if args.visdom and is_leader:
            x_axis = self.epochs[0:epoch + 1]
            y_axis = torch.stack(
                (self.loss_results[0:epoch + 1], self.wer_results[0:epoch + 1], self.cer_results[0:epoch + 1]), dim=1)
            if self.viz_window is None:
                self.viz_window = viz.line(
                    X=x_axis,
                    Y=y_axis,
                    opts=self.opts,
                )
            else:
                viz.line(
                    X=x_axis.unsqueeze(0).expand(y_axis.size(1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                    Y=y_axis,
                    win=self.viz_window,
                    update='replace',
                )
        if args.tensorboard and is_leader:
            values = {
                'Avg Train Loss': avg_loss,
                'Avg WER': wer_avg,
                'Avg CER': cer_avg
            }
            tensorboard_writer.add_scalars(args.id, values, epoch + 1)
            if args.log_params:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                    tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)


def get_lr():
    optim_state = optimizer.state_dict()
    return optim_state['param_groups'][0]['lr']


def set_lr(lr):
    optim_state = optimizer.state_dict()
    optim_state['param_groups'][0]['lr'] = lr
    optimizer.load_state_dict(optim_state)


class Trainer:
    def __init__(self):
        self.end = time.time()

    def check_model_quality(self, epoch, checkpoint, avg_loss):
        gc.collect()
        torch.cuda.empty_cache()

        total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
        model.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                inputs, targets, filenames, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                # unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                inputs = inputs.to(device)

                out, output_sizes = model(inputs, input_sizes)

                decoded_output, _ = decoder.decode(out, output_sizes)
                target_strings = decoder.convert_to_strings(split_targets)
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x][0]
                    wer, cer, wer_ref, cer_ref = get_cer_wer(decoder, transcript, reference)

                    total_wer += wer
                    total_cer += cer
                    num_tokens += wer_ref
                    num_chars += cer_ref

                del inputs, targets, input_percentages, target_sizes
                del out, output_sizes, input_sizes
                del split_targets

                if args.cuda:
                    torch.cuda.synchronize()

            avg_wer = 100 * total_wer / num_tokens
            avg_cer = 100 * total_cer / num_chars
            print('Validation Summary Epoch: [{0}]\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'.format(epoch + 1, wer=avg_wer, cer=avg_cer))

            plots.loss_results[epoch] = avg_loss
            plots.wer_results[epoch] = avg_wer
            plots.cer_results[epoch] = avg_cer

            plots_checkpoints.loss_results[checkpoint] = avg_loss
            plots_checkpoints.wer_results[checkpoint] = avg_wer
            plots_checkpoints.cer_results[checkpoint] = avg_cer

            plots.plot_progress(epoch, avg_loss, avg_cer, avg_wer)
            plots_checkpoints.plot_progress(checkpoint, avg_loss, avg_cer, avg_wer)
        return avg_wer, avg_loss

    def train_batch(self, epoch, i, data):
        inputs, targets, filenames, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        # measure data loading time
        data_time.update(time.time() - self.end)

        inputs = inputs.to(device)
        input_sizes = input_sizes.to(device)

        out, output_sizes = model(inputs, input_sizes)
        assert out.is_cuda
        assert output_sizes.is_cuda
        out = out.transpose(0, 1)  # TxNxH

        loss = criterion(out, targets, output_sizes.cpu(), target_sizes)
        loss = loss / inputs.size(0)  # average the loss by minibatch
        loss = loss.to(device)

        inf = float("inf")
        if args.distributed:
            loss_value = reduce_tensor(loss, args.world_size).item()
        else:
            loss_value = loss.item()
        if loss_value == inf or loss_value == -inf:
            print("WARNING: received an inf loss, setting loss value to 0")
            loss_value = 0

        losses.update(loss_value, inputs.size(0))

        # compute gradient
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        # SGD step
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - self.end)
        if not args.silent:
            print('GPU-{0} Epoch {1} [{2}/{3}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                  'Loss {loss.val:.2f} ({loss.avg:.2f})\t'.format(
                args.gpu_rank or VISIBLE_DEVICES[0],
                epoch + 1, i + 1, len(train_sampler),
                batch_time=batch_time, data_time=data_time, loss=losses))

        del inputs, targets, input_percentages, input_sizes
        del out, output_sizes, target_sizes, loss
        return loss_value


def train(from_epoch, from_iter):
    print('Starting training with id="{}" at GPU="{}" with lr={}'.format(args.id, args.gpu_rank or VISIBLE_DEVICES[0], get_lr()))
    trainer = Trainer()
    best_wer = None
    for epoch in range(from_epoch, args.epochs):
        checkpoint = 0
        total_loss = 0
        steps_passed = 1
        model.train()
        trainer.end = time.time()
        start_epoch_time = time.time()

        for i, data in enumerate(train_loader, start=from_iter):
            steps_passed = i + 1 - from_iter
            if i >= len(train_sampler):
                break
            total_loss += trainer.train_batch(epoch, i, data)

            if (i + 1) % 5 == 0:
                # deal with GPU memory fragmentation
                gc.collect()
                torch.cuda.empty_cache()

            if args.checkpoint_per_batch > 0 and is_leader:
                if 0 < i < len(train_sampler) - 1 and (i + 1) % args.checkpoint_per_batch == 0:
                    file_path = '%s/checkpoint_epoch_%02d_iter_%05d.model' % (save_folder, epoch + 1, i + 1)
                    print("Saving checkpoint model to %s" % file_path)
                    torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=i,
                                                    loss_results=plots.loss_results,
                                                    wer_results=plots.wer_results, cer_results=plots.cer_results,
                                                    avg_loss=total_loss / steps_passed), file_path)

                    trainer.check_model_quality(epoch, checkpoint, avg_loss=total_loss / steps_passed)
                    checkpoint += 1
                    model.train()

            trainer.end = time.time()

        epoch_avg_loss = total_loss / steps_passed

        epoch_time = time.time() - start_epoch_time

        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=epoch_avg_loss))

        from_iter = 0  # Reset start iteration for next epoch
        wer_avg = trainer.check_model_quality(epoch, checkpoint, epoch_avg_loss)

        if args.checkpoint and is_leader:
            file_path = '%s/deepspeech_%d.model' % (save_folder, epoch + 1)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=plots.loss_results,
                                            wer_results=plots.wer_results, cer_results=plots.cer_results), file_path)
            # anneal lr
            new_lr = get_lr() / args.learning_anneal
            set_lr(new_lr)
            print('Learning rate annealed to: {lr:.6f}'.format(lr=new_lr))

        if (best_wer is None or best_wer > wer_avg) and is_leader:
            print("Found better validated model, saving to %s" % args.model_path)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=plots.loss_results,
                                            wer_results=plots.wer_results, cer_results=plots.cer_results),
                       args.model_path)
            best_wer = wer_avg

        if not args.no_shuffle:
            print("Shuffling batches...")
            train_sampler.shuffle(epoch)


if __name__ == '__main__':
    args = parser.parse_args()
    args.distributed = args.world_size > 1
    args.model_path = os.path.join(args.save_folder, 'best.model')

    is_leader = True
    device = torch.device("cuda" if args.cuda else "cpu")
    if args.distributed:
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        is_leader = args.rank == 0  # Only the first proc should save models

    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)

    plots = PlotWindow(args.id, 'epochs')
    plots_checkpoints = PlotWindow(args.id, 'checkpoints')

    total_avg_loss, start_epoch, start_iter = 0, 0, 0
    if args.continue_from:  # Starting from previous model
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        model = DeepSpeech.load_model_package(package)
        labels = DeepSpeech.get_labels(model)
        audio_conf = DeepSpeech.get_audio_conf(model)
        parameters = model.parameters()
        optimizer = build_optimizer(args, parameters)
        if not args.finetune:  # Don't want to restart training
            model = model.to(device)
            optimizer.load_state_dict(package['optim_dict'])
            set_lr(args.lr)
            start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
            start_iter = package.get('iteration', None)
            if start_iter is None:
                start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
                start_iter = 0
            else:
                start_iter += 1
                total_avg_loss = int(package.get('avg_loss', 0))
            plots.loss_results = package['loss_results']
            plots.cer_results = package['cer_results']
            plots.wer_results = package['wer_results']
            if package['loss_results'] is not None and start_epoch > 0:
                plots.plot_history(start_epoch, total_avg_loss)
    else:
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                           nb_layers=args.hidden_layers,
                           labels=labels,
                           rnn_type=supported_rnns[rnn_type],
                           audio_conf=audio_conf,
                           bidirectional=args.bidirectional,
                           bnm=args.batch_norm_momentum)
        parameters = model.parameters()
        optimizer = build_optimizer(args, parameters)

    criterion = CTCLoss()
    decoder = GreedyDecoder(labels)
    train_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                       manifest_filepath=args.train_manifest,
                                       labels=labels, normalize=False, augment=args.augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                      manifest_filepath=args.val_manifest,
                                      labels=labels, normalize=False, augment=False)
    if args.reverse_sort:
        # XXX: A hack to test max memory load.
        train_dataset.ids.reverse()

    if not args.distributed:
        train_sampler = BucketingSampler(train_dataset, batch_size=args.batch_size)
    else:
        train_sampler = DistributedBucketingSampler(train_dataset,
                                                    batch_size=args.batch_size,
                                                    num_replicas=args.world_size,
                                                    rank=args.rank)
    train_loader = AudioDataLoader(train_dataset,
                                   num_workers=args.num_workers,
                                   batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    if (not args.no_shuffle and start_epoch != 0) or args.no_sorta_grad:
        print("Shuffling batches for the following epochs")
        train_sampler.shuffle(start_epoch)

    model = model.to(device)
    if args.distributed:
        device_id = [int(args.gpu_rank)] if args.rank else None
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=device_id)

    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    train(start_epoch, start_iter)
