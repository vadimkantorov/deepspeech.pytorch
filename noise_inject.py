import argparse

import torch
import torchaudio

from data.data_loader_aug import load_audio, NoiseInjection

parser = argparse.ArgumentParser()
parser.add_argument('--input-path', default='input.wav', help='The input audio to inject noise into')
parser.add_argument('--noise-path', default='noise.wav', help='The noise file to mix in')
parser.add_argument('--output-path', default='output.wav', help='The noise file to mix in')
parser.add_argument('--sample-rate', default=16000, help='Sample rate to save output as')
parser.add_argument('--noise-level', type=float, default=1.0,
                    help='The Signal to Noise ratio (higher means more noise)')
args = parser.parse_args()

noise_injector = NoiseInjection()
data, sample_rate_ = load_audio(args.input_path)
assert sample_rate_ == args.sample_rate
mixed_data = noise_injector.inject_noise_sample(data, args.noise_path, args.noise_level)
mixed_data = torch.tensor(mixed_data, dtype=torch.float).unsqueeze(1)  # Add channels dim
torchaudio.save(args.output_path, mixed_data, args.sample_rate)
print('Saved mixed file to %s' % args.output_path)
