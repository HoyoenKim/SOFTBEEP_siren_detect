
#-*- coding: utf-8 -*-
'''
written by: thfdk0101(cabinkhy@gist.ac.kr), 유형균
date: 2022.07.21
'''

# torch import
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# sound package import
import librosa
import soundfile as sf
import torchaudio

# python lib import
import os
import time

# custom model import
from model import CNNNetwork
from model import UrbanSoundDataset

import matplotlib.pyplot as plt
import numpy as np
import socket

# socket 접속 정보 설정
SERVER_IP = '172.17.3.45'
SERVER_PORT = 6000
SIZE = 1024
SERVER_ADDR = (SERVER_IP, SERVER_PORT)

# global settings TODO get input by json format 
wave_file_path = './record.wav'
cut_wave_file_path = './bw_2s_data.wav' # not use
cnn_model_path = './siren_detection.pth'
audio_dir_path = "./result" # not use
annotations_file_path = "./inferencedata.csv" # not use
SAMPLE_RATE = 44100
SAMPLE_RATE = 44100
NUM_SAMPLES = 22050

# model hyper parameter
batch_size = 1
classes = ['Alarm', 'No']

# model
cnn = CNNNetwork()
mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

# code configuration
debug_mode = True
device = 'cpu'

def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader

def model_load():
    global cnn
    global mel_spectrogram
    global device

    # cnn
    cnn = CNNNetwork()
    cnn.load_state_dict(torch.load(cnn_model_path, map_location=device))
    
    # mel spectrogram
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    # device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

if __name__ == '__main__':
    print('______________initializng...______________')
    # start record
    #try:
    #    os.system('arecord -Dac108 -f S32_LE -r 44100 -c 1 record.wav &')
    #except:
    #    print('there is no arecord')

    # program start time
    start_time = time.time()

    # socket state
    detect_state = 1

    # model load
    model_load()

    # read record.wav periodic
    while(True):
        # extract backward 1 ~ 2s of wav file
        wave_data, sample_rate = torchaudio.load(wave_file_path)
        bw_2s_wave_data = wave_data.split(30000, dim=1)
        max_hz = max(bw_2s_wave_data[-1][0])

        # wav to malspectogram
        usd = UrbanSoundDataset(audio_dir_path,
                                mel_spectrogram,
                                SAMPLE_RATE,
                                NUM_SAMPLES,
                                device,
                                bw_2s_wave_data[-1],
                                sample_rate,
                                cut_wave_file_path
                                )

        # malspectogram to CNN
        train_dataloader = create_data_loader(usd, batch_size)
        now_state = detect_state
        for input in train_dataloader:
            input = input.to(device)
            outputs = cnn(input)

            # Detection Siren data
            _, predicted = torch.max(outputs, 1)
            print('max_hz: ', max_hz)

            if max_hz < 0.6:
                now_state = 1
            else:
                now_state = predicted[0].item()
            print('Predicted: ', ''.join(classes[now_state]))

            if now_state != detect_state:
                print('diff need to send to server')
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                    client_socket.connect(SERVER_ADDR)
                    client_socket.send(('Siren Detection State is Changed: ' + str(now_state)).encode()) 
            else:
                print('same')

            detect_state = now_state
        print("executable time: ", time.time() - start_time)