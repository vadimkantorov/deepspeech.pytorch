import argparse
import csv
import os

import numpy as np
import torch
import gc
from tqdm import tqdm

from data.data_loader import SpectrogramDataset, AudioDataLoader
from data.utils import get_cer_wer
from decoder import GreedyDecoder
from model import DeepSpeech
from opts import add_decoder_args, add_inference_args

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--errors', action="store_true", help="print error report")
parser.add_argument('--best', action="store_true", help="print best results")
parser.add_argument('--report-file', metavar='DIR', default='data/test_report.csv', help="Filename to save results")
no_decoder_args = parser.add_argument_group("No Decoder Options", "Configuration options for when no decoder is "
                                                                  "specified")
no_decoder_args.add_argument('--output-path', default=None, type=str, help="Where to save raw acoustic output")
parser = add_decoder_args(parser)
args = parser.parse_args()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    model = DeepSpeech.load_model(args.model_path)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = model.to(device)
    model.eval()

    labels = DeepSpeech.get_labels(model)
    audio_conf = DeepSpeech.get_audio_conf(model)

    report_file = None
    if args.report_file:
        os.makedirs(os.path.dirname(args.report_file), exist_ok=True)
        report_file = csv.writer(open(args.report_file, 'wt'))
        report_file.writerow(['wav', 'text', 'transcript', 'offsets', 'CER', 'WER'])

    if args.decoder == "beam":
        from decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    else:
        decoder = None
    target_decoder = GreedyDecoder(labels, blank_index=labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=audio_conf,
                                      manifest_filepath=args.test_manifest,
                                      labels=labels,
                                      normalize='max_frame')
    # import random;random.shuffle(test_dataset.ids)

    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
        inputs, targets, filenames, input_percentages, target_sizes = data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        inputs = inputs.to(device)

        # print(inputs.shape, inputs.is_cuda, input_sizes.shape, input_sizes.is_cuda)
        out, output_sizes = model(inputs, input_sizes)

        del inputs, targets, input_percentages, target_sizes

        if decoder is None:
            # add output to data array, and continue
            output_data.append((out.numpy(), output_sizes.numpy()))
            continue

        decoded_output, _ = decoder.decode(out.data, output_sizes.data)
        target_strings = target_decoder.convert_to_strings(split_targets)
        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            wer, cer, wer_ref, cer_ref = get_cer_wer(decoder, transcript, reference)

            if args.verbose:
                print("Ref:", reference)
                print("Hyp:", transcript)
                print("Wav:", filenames[x])
                print("WER:", "{:.2f}".format(100 * wer / wer_ref), "CER:", "{:.2f}".format(100 * cer / cer_ref), "\n")
            elif args.errors:
                if cer / cer_ref > 0.5 and transcript.strip():
                    # print("FN:", )
                    print("Ref:", reference)
                    print("Hyp:", transcript)
                    print("Wav:", filenames[x])
                    print("WER:", "{:.2f}".format(100 * wer / wer_ref), "CER:", "{:.2f}".format(100 * cer / cer_ref),
                          "\n")
            elif args.best:
                if cer / cer_ref < 0.15:
                    # print("FN:", )
                    print("Ref:", reference)
                    print("Hyp:", transcript)
                    print("Wav:", filenames[x])
                    print("WER:", "{:.2f}".format(100 * wer / wer_ref), "CER:", "{:.2f}".format(100 * cer / cer_ref),
                          "\n")

            if report_file:
                # report_file.write_row(['wav', 'text', 'transcript', 'offsets', 'CER', 'WER'])

                report_file.writerow([
                    filenames[x],
                    reference,
                    transcript,
                    '',
                    cer / cer_ref,
                    wer / wer_ref
                ])

            total_wer += wer
            total_cer += cer
            num_tokens += wer_ref
            num_chars += cer_ref

        del out, output_sizes
        if (i + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    if decoder is not None:
        wer_avg = float(total_wer) / num_tokens
        cer_avg = float(total_cer) / num_chars

        print('Test Summary \t'
              'Average WER {wer:.3f}\t'
              'Average CER {cer:.3f}\t'.format(wer=wer_avg * 100, cer=cer_avg * 100))
    else:
        np.save(args.output_path, output_data)
