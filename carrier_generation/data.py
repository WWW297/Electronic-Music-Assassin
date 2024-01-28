# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
import json
import os
import glob
import librosa
import numpy as np
import torch
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self, file_path, seq_len,epoch_len):
        self.file_path = file_path
        self.seq_len = seq_len
        self.epoch_len = epoch_len
        if not self.file_path:
            print(f'No files found in {self.file_path}')
        self.wav_data=librosa.load(self.file_path, sr=16000, mono=True)[0]
        self.wav_length=len(self.wav_data)

        print(self.file_path,':',self.wav_length)

    def __getitem__(self, _):
        wav = self.get_wav_seq()
        return wav

    def get_wav_seq(self):
        start_time = random.randint(0, self.wav_length - self.seq_len-1)
        data = self.wav_data[start_time: start_time + self.seq_len]
        return data

    def __len__(self):
        return self.epoch_len

class DataLoader:
    def __init__(self, file_path, batch_size,num_workers, seq_len=81792):
        self.dataset = Dataset(file_path, seq_len=seq_len, epoch_len=10000000000000)
        self.train_loader = data.DataLoader(self.dataset,
                                            batch_size=batch_size,
                                            num_workers=num_workers,
                                            pin_memory=True)
        self.train_iter = iter(self.train_loader)

