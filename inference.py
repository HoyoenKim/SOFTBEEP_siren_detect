from cnn import CNNNetwork
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import librosa
import soundfile as sf
import os
#import numpy as np
#import soundfile as sf
import time
start_time = time.time()

#print(librosa.__version__)

wav = './record.wav'
(file_dir, file_id) = os.path.split(wav)
#print("file_dir:", file_dir)
#print("file_id:", file_id)

# original
#print("1: ", time.time() - start_time)
y, sr = librosa.load(wav, sr=16000)

#print(len(y))
y2 = y[len(y)-35106:len(y)]
#print(y2)

#deprecated at librosa==0.9.2
#librosa.output.write_wav('./result/1/cut_file.wav', y2, sr)
sf.write('./cut_file.wav', y2, sr)


import os
SAMPLE_RATE = 22050
classes = ('Alarm', 'No')
cnn = CNNNetwork()
cnn.load_state_dict(torch.load("./feedforwardnet.pth"))
def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

class UrbanSoundDataset(Dataset):

    def __init__(self,

                 audio_dir,
                 transformation,
                 target_sample_rate,
                 num_samples,
                 device):

        self.audio_dir = audio_dir
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return 1

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)

        signal, sr = torchaudio.load(audio_sample_path)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = self.transformation(signal)
        return signal

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            resampler = resampler.to(self.device)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        path = "./result/1/cut_file.wav"
        return path

    #def _get_audio_sample_label(self, index):
    #    return self.annotations.iloc[index, 2]
AUDIO_DIR = "./result"
ANNOTATIONS_FILE = "./inferencedata.csv"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
usd = UrbanSoundDataset(

                        AUDIO_DIR,
                        mel_spectrogram,
                        SAMPLE_RATE,
                        NUM_SAMPLES,
                        device)
train_dataloader = create_data_loader(usd, 1)
for input in train_dataloader:
    input = input.to("cpu")
    outputs = cnn(input)

    _, predicted = torch.max(outputs, 1)

    print('Predicted: ', ' '.join(classes[predicted[0]]))
print("2: ", time.time() - start_time)
