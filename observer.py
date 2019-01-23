import os
import torch
import logging

from model import DeepSpeech


class Observer(object):
    '''
    Train Observer base class.
    '''

    def __init__(self, logger):
        self.logger = logger

    def on_epoch_start(self, model, epoch): pass

    def on_epoch_end(self, model, optimizer, epoch, loss_results, wer_results, cer_results): pass

    def on_batch_start(self, model, epoch, batch_no): pass

    def on_batch_end(self, model, optimizer, epoch, batch_no, loss_results, wer_results, cer_results, avg_loss): pass


def to_np(x):
    return x.data.cpu().numpy()


class TensorboardWriter(Observer):
    """
    Update Tensorboard at the end of each epoch
    """

    def __init__(self, id, log_dir, log_params):
        super().__init__(logging.getLogger('TensorboardWriter'))
        os.makedirs(log_dir, exist_ok=True)
        from tensorboardX import SummaryWriter

        self.id = id
        self.log_params = log_params
        self.tensorboard_writer = SummaryWriter(log_dir)

    def on_epoch_end(self, model, optimizer, epoch, loss_results, wer_results, cer_results):
        self.logger.debug("Updating tensorboard for epoch {} {}".format(epoch + 1, loss_results))
        values = {
            'Avg Train Loss': loss_results[epoch],
            'Avg WER': wer_results[epoch],
            'Avg CER': cer_results[epoch],
        }
        self.tensorboard_writer.add_scalars(self.id, values, epoch + 1)
        if self.log_params:
            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                self.tensorboard_writer.add_histogram(tag, to_np(value), epoch + 1)
                if value.grad is not None:
                    self.tensorboard_writer.add_histogram(tag + '/grad', to_np(value.grad), epoch + 1)


class CheckpointWriter(Observer):
    """
    Save model checkpoint at the end of epoch
    """

    def __init__(self, save_folder):
        super().__init__(logging.getLogger('CheckpointWriter'))
        self.logger.debug("CheckpointWriter")
        self.save_folder = save_folder
        os.makedirs(save_folder, exist_ok=True)

    def on_epoch_end(self, model, optimizer, epoch, loss_results, wer_results, cer_results):
        self.logger.debug("Saving checkpoint {}".format(epoch + 1))
        file_path = '%s/deepspeech_%d.pth' % (self.save_folder, epoch + 1)
        torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                        wer_results=wer_results, cer_results=cer_results),
                   file_path)


class CheckpointBatchWriter(Observer):
    """
    Save model checkpoint every number of mini-batches
    """

    def __init__(self, save_folder, checkpoint_per_batch):
        super().__init__(logging.getLogger('CheckpointBatchWriter'))
        self.logger.debug("CheckpointBatchWriter")
        self.save_folder = save_folder
        self.checkpoint_per_batch = checkpoint_per_batch
        os.makedirs(save_folder, exist_ok=True)

    def on_batch_end(self, model, optimizer, epoch, batch_no, loss_results, wer_results, cer_results, avg_loss):
        if batch_no > 0 and (batch_no + 1) % self.checkpoint_per_batch == 0:
            file_path = '%s/deepspeech_checkpoint_epoch_%d_iter_%d.pth' % (self.save_folder, epoch + 1, batch_no + 1)
            self.logger.debug("Saving checkpoint model to %s" % file_path)
            torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, iteration=batch_no,
                                            loss_results=loss_results,
                                            wer_results=wer_results, cer_results=cer_results, avg_loss=avg_loss),
                       file_path)


class VisdomWriter(Observer):
    def __init__(self, id, epochs):
        super().__init__(logging.getLogger('VisdomWriter'))
        from visdom import Visdom

        self.viz = Visdom()
        self.opts = dict(title=id, ylabel='', xlabel='Epoch', legend=['Loss', 'WER', 'CER'])
        self.viz_window = None
        self.epochs = torch.arange(1, epochs + 1)

    def on_epoch_end(self, model, optimizer, epoch, loss_results, wer_results, cer_results):
        self.logger.debug('Updating Visdom')

        x_axis = self.epochs[0:epoch + 1]
        y_axis = torch.stack(
            (loss_results[0:epoch + 1], wer_results[0:epoch + 1], cer_results[0:epoch + 1]), dim=1)
        if self.viz_window is None:
            self.viz_window = self.viz.line(
                X=x_axis,
                Y=y_axis,
                opts=self.opts,
            )
        else:
            self.viz.line(
                X=x_axis.unsqueeze(0).expand(y_axis.size(1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                Y=y_axis,
                win=self.viz_window,
                update='replace',
            )
