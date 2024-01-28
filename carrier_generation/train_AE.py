# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_value_
import os
from warnings import simplefilter
import argparse
from itertools import chain
import numpy as np
from pathlib import Path
import tqdm
import os
import warnings
import imageio
import wave
import torchaudio
from model import *
from data import *

warnings.filterwarnings("ignore")
simplefilter(action='ignore', category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
parser = argparse.ArgumentParser()
torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


class Trainer_AE:
    def __init__(self):
        torch.manual_seed(2)
        torch.cuda.manual_seed(2)
        self.batch_size=32
        self.num_workers=32
        self.epochs=1000
        self.epoch_len=500
        self.lr=1e-5
     
        self.dataloader = DataLoader(
            file_path='', #electric music path
            batch_size=self.batch_size, num_workers=self.num_workers)

        self.encoder=Encoder().cuda()
        self.encoder.train()
        self.decoder = Decoder().cuda()
        self.decoder.train()
        self.discriminator =ZDiscriminator().cuda()
        self.discriminator.train()

        model_state = torch.load('') #pretrained model
        self.encoder.load_state_dict(model_state['encoder_state'])
        self.decoder.load_state_dict(model_state['decoder_state'])
        self.discriminator.load_state_dict(model_state['discriminator_state'])

        self.discriminator_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.lr)
        self.model_optimizer = optim.Adam(chain(self.encoder.parameters(),self.decoder.parameters()), lr=self.lr)
        
    def train_epoch(self,epoch):
        d_label=torch.cat((torch.zeros(self.batch_size),torch.ones(self.batch_size)),0).long().cuda()
        with tqdm.tqdm(total=self.epoch_len, desc='AE Train:') as train_enum:
            with torch.autograd.set_detect_anomaly(True):
                for batch_num in range(self.epoch_len):
                    wav = next(self.dataloader.train_iter).cuda()
                    stft = wav_preprocess(wav,512,128)[:,:,0,:]

                    z = self.encoder(square_smooth(stft,square_kernel_size=[11],kernel_size=[25]))
                    y = self.decoder(stft-square_smooth(stft,square_kernel_size=[11],kernel_size=[25]), z)
                    d_logits = self.discriminator(torch.cat((stft,y),0))
                    d_acc = torch.sum(torch.argmax(d_logits,dim=-1)==d_label)/self.batch_size/2
                    d_loss = F.cross_entropy(d_logits,d_label)
                    self.discriminator_optimizer.zero_grad()
                    d_loss.backward(retain_graph=True)
                    self.discriminator_optimizer.step()

                    d_logits = self.discriminator(y)
                    dg_loss = F.cross_entropy(d_logits,torch.zeros(self.batch_size).long().cuda())
                    recon_loss = F.l1_loss(y,stft)
                    model_loss=recon_loss+dg_loss*0.01
                    self.model_optimizer.zero_grad()
                    model_loss.backward(retain_graph=True)
                    self.model_optimizer.step()

                    if batch_num%20==0:
                        imageio.imwrite('../AE_encoding.jpg',(z[0].cpu().detach().numpy()+1)*128)
                        imageio.imwrite('../AE_training.jpg',(torch.cat((stft[0],y[0]),0).cpu().detach().numpy()+1)*128)
                        
                    train_enum.set_description(f'Train AE:(EPOCH:{epoch},recon_loss:{recon_loss:.5f},dg_loss:{dg_loss:.5f},d_loss:{d_loss:.5f},d_acc:{d_acc*100:.2f}%)')
                    train_enum.update()

        

    def train_AE(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.save_model('') #model path

    def save_model(self, filename):
        save_path = Path('../models')/filename
        torch.save({'discriminator_state':self.discriminator.state_dict(),'encoder_state': self.encoder.state_dict(),'decoder_state': self.decoder.state_dict()},save_path)

def wav_write(wav_signal: np.ndarray, wav_path: str, sample_rate: int, scale_bit_length: bool = True, bit_length: int = 16):
    """
    write the wav file
    :param wav_signal: wav signal.
    :param wav_path: wav path.
    :param scale_bit_length: whether scale signal to bit length.
    :param sample_rate: sample rate.
    :param bit_length: bit length.
    """
    wav_handler = wave.open(wav_path, "wb")
    wav_handler.setparams((1, 2, sample_rate, 0, 'NONE', 'not compressed'))
    max_amp = 2 ** (bit_length - 1)
    if scale_bit_length:
        wav_data = (wav_signal * max_amp).astype(np.int16)
    else:
        wav_data = wav_signal.astype(np.int16)
    wav_data = np.clip(wav_data, -max_amp, max_amp - 1)
    wav_handler.writeframes(wav_data.tobytes())
    wav_handler.close()

def log10(x):
    numerator = torch.log(x)
    denominator = torch.log(torch.tensor(10,dtype=torch.float))
    return numerator / denominator

def normalize(S):
    S=(S+25.0)/75
    return S

def power2db(power, ref_value=1.0, amin=1e-10):
    log_spec = 10.0 * log10(torch.maximum(torch.ones_like(power)*amin, power))
    log_spec -= 10.0 * log10(torch.maximum(torch.ones_like(power)*amin, torch.ones_like(power)*ref_value))
    return log_spec

def denormalize(S):
    return (S * 75) -25.0

def db2power(S_db, ref=1.0):
    return ref * torch.pow(10.0, 0.1 * S_db)

def conc_tog_specphase(S, P):
    S = denormalize(S)
    S = db2power(S)
    P = P * np.pi
    SP = S * torch.complex(torch.cos(P),torch.sin(P))
    wav = torch.istft(SP,n_fft=512,hop_length=128,win_length=512,window=torch.hann_window(512).cuda())
    return wav

def wav_preprocess(wav,n_fft,hop_length):
    stft = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft).cuda())
    stft_mag=torch.abs(torch.sqrt(torch.pow(stft[:,:,:,0], 2) + torch.pow(stft[:,:,:,1], 2)))
    stft_phase=torch.atan2(stft[:,:,:,1].data, stft[:,:,:,0].data)/np.pi
    stft_mag = normalize(power2db(stft_mag))
    stft = torch.cat((stft_mag.unsqueeze(2),stft_phase.unsqueeze(2)),2)
    return stft

def square_smooth(input,square_kernel_size=[],kernel_size=[]):
    input=input.unsqueeze(1)
    bias=torch.zeros(1).cuda()
    for size in kernel_size:
        kernel=(torch.ones(1,1,size,1)/size).cuda()
        padding_size=int((size-1)/2)
        input=F.pad(input,(0,0,padding_size,padding_size),mode='reflect')
        input=F.conv2d(input, kernel, bias, stride=1)
    for size in square_kernel_size:
        kernel=(torch.ones(1,1,size,size)/size**2).cuda()
        padding_size=int((size-1)/2)
        input=F.pad(input,(padding_size,padding_size,padding_size,padding_size),mode='reflect')
        input=F.conv2d(input, kernel, bias, stride=1)
    return input.squeeze()

def main():
    Trainer_AE().train_AE()
   

if __name__ == '__main__':
    main()

