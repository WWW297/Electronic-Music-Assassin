import torch
import torch.nn.functional as F
import torch.optim as optim
from warnings import simplefilter
import argparse
from tqdm import tqdm
from pathlib import Path
import numpy as np
import scipy
import librosa
import random
import imageio
import wave
from typing import Union
import os
import glob
import math 
import json
from cmaes import CMA
import matplotlib.pyplot as plt
import sys
sys.path.append('./cloud_decode')
from cloud_decode.aliyun_function import *
from cloud_decode.tencentyun_function import *
from cloud_decode.baidu_api import *
from cloud_decode.xunfei_api import *
from cloud_decode.google_api import *
from cloud_decode.azure_api import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

simplefilter(action='ignore', category=Warning)
parser = argparse.ArgumentParser(description='ASR attack')
plt.rcParams.update({'font.size': 24})
torch.backends.cudnn.enabled = False

parser.add_argument('--seed', type=int, default=2023, metavar='S',help='random seed (default: 2023)')
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--speech-file-path', default='',type=str, help='speech file path')
parser.add_argument('--music-file-path', default='',type=str, help='music file path')
parser.add_argument('--attack-target', default='google',type=str,choices=['tencentyun','aliyun','iflytec','google','azure'])
parser.add_argument('--sample-num', default=0,type=int, help='samples num')
parser.add_argument('--sound-db', default=75,type=int, help='sound db')

class Attacker:
    def __init__(self, args):
        self.args = args
        self.epoch=args.epoch
        self.sound_db = args.sound_db
        self.n_fft=512
        self.hop_length=256
        self.frame_num=320
        self.seq_len=self.hop_length*(self.frame_num-1)+self.n_fft

        if args.attack_target == 'tencentyun':
            self.decode_function=tencent_recong
        if args.attack_target == 'aliyun':
            self.decode_function=aliyun_recong
        if args.attack_target == 'iflytec':
            self.decode_function=xunfei_decode
        if args.attack_target == 'google':
            self.decode_function=google_decode
        if args.attack_target == 'azure':
            self.decode_function=azure_decode

        self.speech,_=self.read_wav_file(args.speech_file_path)
        speech_name=args.speech_file_path.split('/')[-1].split('.')[0]
        self.command=speech_name.replace('_',' ')
        digital_samples_path=Path('success_samples')/speech_name/args.attack_target
        if not os.path.exists(digital_samples_path):
            os.makedirs(digital_samples_path)
        self.digital_samples_path=digital_samples_path

        count=0
        self.decode_count=0
        while count < args.sample_num:
            music_file_list=glob.glob(args.music_file_path+'/*.wav')
            music_file=random.choice(music_file_list)
            # music_file='./music/Nirvana.wav'
            music_name=music_file.split('/')[-1].split('.')[0]
            music,start_time=self.read_wav_file(music_file)
            self.attack_music_name=speech_name+'_'+music_name+'_'+str(start_time)
            print(self.attack_music_name,'  attack begin!!')
            success=self.distribution_attack(self.speech,music)
            if success:
                count+=1
            else:
                print(self.attack_music_name,'  Failed!!')
                

    def read_wav_file(self,file_path,start_time=0):
        wav_data,_ = librosa.load(file_path, sr=16000, mono=True)
        if len(wav_data)>self.seq_len:
            if not start_time:
                start_time=random.randint(0, len(wav_data)-self.seq_len)
            wav_data=wav_data[start_time:start_time+self.seq_len]
        else:
            wav_data=wav_data[np.convolve(np.abs(wav_data), np.ones((512))/512, mode='same')>0.001]
        mean_amp = np.mean(np.abs(wav_data*(2**15)))
        scale_ratio = self.compute_scale_ratio(mean_amp, self.sound_db, max_amp_value=32768.)
        wav_data = scale_ratio * wav_data
        wav_data=np.clip(np.array(wav_data).ravel(),-1,1)
        return wav_data,start_time

    def compute_scale_ratio(self,amp, expected_db, max_amp_value: float = 32768.):
        return np.power(10, (expected_db + 20 * np.log10(max_amp_value) - 96.) / 20) / amp
    
    
    def distribution_attack(self,speech,music): 
        music_mag,music_stft_phase=wav_preprocess(music,self.n_fft,self.hop_length)
        music_mag=normalize_e(music_mag)
        clean_music=conc_tog_specphase(music_mag.unsqueeze(0),music_stft_phase.unsqueeze(0),n_fft=self.n_fft,hop_length=self.hop_length).squeeze()
        speech_mag,_=wav_preprocess(speech,self.n_fft,self.hop_length)
        speech_mag=normalize_e(speech_mag)
        speech_frame=speech_mag.size(1)
        speech_rhythm=F.softmax(torch.mean(square_smooth(denormalize_e(speech_mag),square_kernel_size=[7]),dim=0))

        plt.plot(range(speech_frame), speech_rhythm.cpu().detach().numpy(),color='g')
        plt.plot(range(speech_frame), torch.ones(speech_frame)*torch.mean(speech_rhythm).cpu(),color='r')
        plt.savefig('speech_rhythm.png')
        plt.close()

        speech_seq_len_list=[]
        start_frame_idx=0
        end_frame_idx=0
        for frame_idx in range(2,speech_frame-2):
            if frame_idx==(speech_frame-3):
                end_frame_idx=speech_frame
            elif speech_rhythm[frame_idx]<=speech_rhythm[frame_idx-1] and speech_rhythm[frame_idx-1]<=speech_rhythm[frame_idx-2]:
                if speech_rhythm[frame_idx]<=speech_rhythm[frame_idx+1] and speech_rhythm[frame_idx+1]<=speech_rhythm[frame_idx+2]:
                    if speech_rhythm[frame_idx]<torch.mean(speech_rhythm):
                        end_frame_idx=frame_idx
            if start_frame_idx != end_frame_idx:
                if end_frame_idx-start_frame_idx>=10 or end_frame_idx==speech_frame:
                    speech_seq_len_list.append(end_frame_idx-start_frame_idx)
                    start_frame_idx=end_frame_idx
        if speech_seq_len_list[-1] < 15:
            speech_seq_len_list[-2]+=speech_seq_len_list[-1]
            speech_seq_len_list.pop()
        
        music_mag_edge=edge_detection(square_smooth(denormalize_e(music_mag),square_kernel_size=[5]))
        music_rhythm=torch.mean(F.softmax(music_mag_edge),dim=0)
        plt.plot(range(self.frame_num), music_rhythm.cpu(),color='r')
        plt.plot(range(self.frame_num), torch.ones(self.frame_num)*torch.mean(music_rhythm[music_rhythm>torch.mean(music_rhythm)]).cpu(),color='g')
        plt.savefig('music_rhythm.png')
        plt.close()
    
        music_seq_idx_list=[]
        music_seq_len_list=[]
        start_frame_idx=0
        end_frame_idx=0
        for frame_idx in range(2,self.frame_num-2):
            if frame_idx==self.frame_num-3:
                end_frame_idx=self.frame_num
            elif music_rhythm[frame_idx]>=music_rhythm[frame_idx-1] and music_rhythm[frame_idx-1]>=music_rhythm[frame_idx-2]:
                if music_rhythm[frame_idx]>=music_rhythm[frame_idx+1] and music_rhythm[frame_idx+1]>=music_rhythm[frame_idx+2]:
                    if music_rhythm[frame_idx]>torch.mean(music_rhythm[music_rhythm>torch.mean(music_rhythm)]):
                        end_frame_idx=frame_idx
            if start_frame_idx != end_frame_idx:
                if end_frame_idx-start_frame_idx>=10 or end_frame_idx==self.frame_num:
                    music_seq_idx_list.append([start_frame_idx,end_frame_idx])
                    music_seq_len_list.append(end_frame_idx-start_frame_idx)
                    start_frame_idx=end_frame_idx
        if music_seq_len_list[-1] < 10:
            music_seq_len_list[-2]+=music_seq_len_list[-1]
            music_seq_len_list.pop()

        if len(music_seq_len_list)<=len(speech_seq_len_list):
            print('music seq num is too small!')
            return False
        
        
        smooth_speech_mag=square_smooth(speech_mag,kernel_size=[11,9])
        smooth_music_mag=square_smooth(music_mag,kernel_size=[11,9])     
        
        smooth_speech_mag_split=torch.split(smooth_speech_mag,speech_seq_len_list,-1)
        speech_mag_split=torch.split(speech_mag,speech_seq_len_list,-1)

        fitness_dict={}
        for seq_idx in range(len(music_seq_len_list)-len(speech_seq_len_list)):
            if torch.max(torch.abs(torch.tensor(music_seq_len_list[seq_idx:seq_idx+len(speech_seq_len_list)])-torch.tensor(speech_seq_len_list)).float())<5:
                frame_idx=music_seq_idx_list[seq_idx][0]
                speech_frame=torch.sum(torch.tensor(music_seq_len_list[seq_idx:seq_idx+len(speech_seq_len_list)]))
                interpolate_smooth_speech_mag=[]
                for idx,seq_len in enumerate(music_seq_len_list[seq_idx:seq_idx+len(speech_seq_len_list)]):
                    interpolate_smooth_speech_mag.append(F.interpolate(smooth_speech_mag_split[idx].unsqueeze(0),size=seq_len).squeeze())
                interpolate_smooth_speech_mag=torch.cat(interpolate_smooth_speech_mag,dim=-1)
                speech_feature=(interpolate_smooth_speech_mag-square_smooth(interpolate_smooth_speech_mag,kernel_size=[15]))
                music_feature=smooth_music_mag[:,frame_idx:frame_idx+speech_frame]-square_smooth(smooth_music_mag[:,frame_idx:frame_idx+speech_frame],kernel_size=[15])
                mag_fitness=F.l1_loss(speech_feature,music_feature)
                if mag_fitness<0:
                    continue
                fitness_dict[seq_idx]=mag_fitness
        candidate_seq_idx=sorted(fitness_dict.items(),key=lambda d: d[1],reverse=True)
        for seq_idx,similarity_value in candidate_seq_idx[:3]:
            frame_idx=music_seq_idx_list[seq_idx][0]
            speech_frame=torch.sum(torch.tensor(music_seq_len_list[seq_idx:seq_idx+len(speech_seq_len_list)]))
            if frame_idx<10 or frame_idx>self.frame_num-speech_frame-10:
                continue
            
            print('frame_idx:',frame_idx,'    speech_frame:',speech_frame.item(),'     similarity_value:',similarity_value.item())
            interpolate_smooth_speech_mag=[]
            interpolate_speech_mag=[]
            for idx,seq_len in enumerate(music_seq_len_list[seq_idx:seq_idx+len(speech_seq_len_list)]):
                interpolate_smooth_speech_mag.append(F.interpolate(smooth_speech_mag_split[idx].unsqueeze(0),size=seq_len).squeeze())
                interpolate_speech_mag.append(F.interpolate(speech_mag_split[idx].unsqueeze(0),size=seq_len).squeeze())
            interpolate_smooth_speech_mag=torch.cat(interpolate_smooth_speech_mag,dim=-1)
            interpolate_speech_mag=torch.cat(interpolate_speech_mag,dim=-1)
            speech_feature=(interpolate_smooth_speech_mag-square_smooth(interpolate_smooth_speech_mag,kernel_size=[15]))
            music_feature=smooth_music_mag[:,frame_idx:frame_idx+speech_frame]-square_smooth(smooth_music_mag[:,frame_idx:frame_idx+speech_frame],kernel_size=[15])
            feature_mask=square_smooth(torch.tensor(square_smooth(torch.abs(speech_feature),kernel_size=[7])<0.01).float(),kernel_size=[5])
            target_feature=speech_feature+feature_mask*music_feature

            advised_gain=torch.zeros_like(music_mag)
            advised_gain.requires_grad=True
            optimizer=optim.Adam(params=[advised_gain],lr=5e-2)
            balance_param=50
            result=''
            with tqdm(total=self.epoch, desc='Adversarial Train') as train_enum:
                for epoch in range(self.epoch):
                    with torch.autograd.set_detect_anomaly(True):
                        smooth_advised_gain=square_smooth(advised_gain,square_kernel_size=[3])
                        advised_mag=torch.clamp_min(music_mag+smooth_advised_gain,0)
                        smooth_advised_mag=square_smooth(advised_mag,kernel_size=[11,9])
                        wav_feature=(smooth_advised_mag-square_smooth(smooth_advised_mag,kernel_size=[15]))[:,frame_idx:frame_idx+speech_frame]
                        feature_loss=F.mse_loss(wav_feature,target_feature)
                        loss = feature_loss*balance_param+F.mse_loss(smooth_advised_gain,square_smooth(smooth_advised_gain,square_kernel_size=[5],kernel_size=[5]))
                        optimizer.zero_grad()
                        loss.backward(retain_graph=True)
                        optimizer.step()
                        train_enum.set_description(f'Train attacker:(loss:{loss:.10f},feature_loss:{feature_loss:.8f})')
                        train_enum.update()

            mask=torch.zeros_like(music_mag)
            mask[5:,frame_idx-5:frame_idx+speech_frame+5]=1
            mask=square_smooth(mask,square_kernel_size=[9])
            smooth_advised_gain*=mask
            target_mag=torch.clamp_min(music_mag+smooth_advised_gain,0)
            wav=GLA(denormalize_e(target_mag).unsqueeze(0),music_stft_phase.unsqueeze(0),self.n_fft,self.hop_length).squeeze()
            audio_file=self.attack_music_name+'_'+str(frame_idx)+'_'+str(speech_frame.item())+'_'+self.args.attack_target+'_'+str(self.decode_count)+'.wav'
            wav_write(np.clip(wav.cpu().detach().numpy().ravel(),-1,1),audio_file,16000)
            result,success = self.decode_function(audio_file)
            self.decode_count+=1
            print(result)
            assert success,'Decode Failed!!!'
            if self.command not in result.lower().replace('\'s',' is').replace(',','').replace('.','').replace('911','nine one one').replace('9','nine').replace('1','one'):
                # time.sleep(15)
                os.system('rm -f '+audio_file)
                continue

            # imageio.imwrite('attack_stft.png',torch.cat((smooth_speech_mag,target_mag,music_mag),1).cpu().detach().numpy())
            # fig = plt.figure(figsize=(40, 50), dpi=20)
            # ax=fig.add_subplot(321, projection='3d')
            # x,y=np.meshgrid(np.arange(self.n_fft//2+1),np.arange(speech_frame))
            # ax.plot_surface(x, y, interpolate_smooth_speech_mag.cpu().detach().numpy().T, cmap='plasma')

            # ax=fig.add_subplot(322, projection='3d')
            # x,y=np.meshgrid(np.arange(self.n_fft//2+1),np.arange(speech_frame))
            # ax.plot_surface(x, y, smooth_music_mag[:,frame_idx:frame_idx+speech_frame].cpu().detach().numpy().T, cmap='plasma')
            
            # ax=fig.add_subplot(323, projection='3d')
            # x,y=np.meshgrid(np.arange(self.n_fft//2+1),np.arange(speech_frame))
            # ax.plot_surface(x, y, speech_feature.cpu().detach().numpy().T, cmap='plasma')

            # ax=fig.add_subplot(324, projection='3d')
            # x,y=np.meshgrid(np.arange(self.n_fft//2+1),np.arange(speech_frame))
            # ax.plot_surface(x, y, smooth_advised_gain.cpu().detach().numpy().T, cmap='plasma')

            # ax=fig.add_subplot(325, projection='3d')
            # x,y=np.meshgrid(np.arange(self.n_fft//2+1),np.arange(speech_frame))
            # ax.plot_surface(x, y, target_feature.cpu().detach().numpy().T, cmap='plasma')

            # ax=fig.add_subplot(326, projection='3d')
            # x,y=np.meshgrid(np.arange(self.n_fft//2+1),np.arange(speech_frame))
            # ax.plot_surface(x, y, normalize_e(target_mag)[:,frame_idx:frame_idx+speech_frame].cpu().detach().numpy().T, cmap='plasma')
            # plt.savefig('distribution.png')
            # plt.show()
            # plt.close()

           
            success_audio_file=str(self.digital_samples_path)+'/'+audio_file
            wav_write(np.clip(wav.cpu().detach().numpy().ravel(),-1,1),success_audio_file,16000)
            with open(success_audio_file.replace('.wav','.json'), 'w') as json_file:
                json.dump({'result':result}, json_file, ensure_ascii=False)
            self.decode_count=0
                    
            variable_num=((speech_frame.item()+20)//10)*int(self.n_fft/20)
            optimizer = CMA(mean=np.ones(variable_num),sigma=0.05,bounds=np.array([[0,1]]*variable_num),population_size=15)
            best_value=1e11
            start_index=(frame_idx-5)*self.hop_length
            noise_len=(speech_frame+10)*self.hop_length+self.n_fft
            for generation in range(1000):
                solutions = []
                for _ in range(optimizer.population_size):
                    variable = optimizer.ask()
                    value=0
                    CMA_advised_mag=smooth_advised_gain.clone()
                    CMA_advised_mag[:,frame_idx-10:frame_idx+speech_frame+10]*=square_smooth(F.interpolate(torch.tensor(variable).view(1,1,int(self.n_fft/20),-1).to(device),size=(int(self.n_fft/2+1), speech_frame+20), mode='area').squeeze().float(),square_kernel_size=[9])
                    target_mag=denormalize_e(torch.clamp_min(music_mag+CMA_advised_mag,0))
                    cmaes_wav=GLA(target_mag.unsqueeze(0),music_stft_phase.unsqueeze(0),self.n_fft,self.hop_length).squeeze()
                    wav_write(np.clip(cmaes_wav.cpu().detach().numpy().ravel(),-1,1),audio_file,16000)
                    noise=cmaes_wav-clean_music
                    noise=noise[start_index:start_index+noise_len]
                    SNR=10*torch.log10((torch.sum(clean_music[start_index:start_index+noise_len]**2))/(torch.sum(noise**2)))
                    cmaes_result,success= self.decode_function(audio_file)
                    assert success,'Decode Failed!!!'
                    value+=torch.sum(torch.abs(target_mag-denormalize_e(music_mag))).item()
                    if self.command not in cmaes_result.lower().replace('\'s',' is').replace(',','').replace('.',''):
                        value=5e10
                    if value < best_value and value<1e10:
                        best_value=value
                        best_generation=generation
                        best_cmaes_wav=cmaes_wav
                        imageio.imwrite('attack_stft.png',torch.cat((smooth_speech_mag,torch.clamp_min(normalize_e(target_mag),0)),1).cpu().detach().numpy())
                        wav_write(np.clip(best_cmaes_wav.cpu().detach().numpy().ravel(),-1,1),success_audio_file,16000)
                        best_SNR=SNR
                        with open(success_audio_file.replace('.wav','.json'), 'w') as json_file:
                            json.dump({'result':cmaes_result}, json_file, ensure_ascii=False)
                    solutions.append((variable,value))
                    print(f'#{generation}: {value}  target:{self.args.attack_target}  SNR:{SNR:.5f}')
                if generation-best_generation>5:
                    break
                optimizer.tell(solutions)
            if best_value<1e10:
                save_file=success_audio_file.split('.')[0]
                save_file=save_file+f'_snr_{best_SNR:.3f}_{optimizer.population_size*(generation+1)}.wav'
                os.rename(success_audio_file,save_file)
                os.rename(success_audio_file.replace('.wav','.json'),save_file.replace('.wav','.json'))
                os.system('rm -f '+audio_file)
                print(f'cma_es success!!! best_value:{best_value}')
            else:
                print('cma_es Failed!!!')
            return True
                        
        return False


def square_smooth(input,square_kernel_size=[],kernel_size=[]):
    bias=torch.zeros(1).to(device)
    input=input.clone().unsqueeze(0).unsqueeze(0)
    for size in square_kernel_size:
        kernel=(torch.ones(1,1,size,size)/size**2).to(device)
        padding_size=int((size-1)/2)
        input=F.pad(input,(padding_size,padding_size,padding_size,padding_size),mode='replicate')
        input=F.conv2d(input, kernel, bias, stride=1)
    for size in kernel_size:
        kernel=(torch.ones(1,1,size,1)/size).to(device)
        padding_size=int((size-1)/2)
        input=F.pad(input,(0,0,padding_size,padding_size),mode='replicate')
        input=F.conv2d(input, kernel, bias, stride=1)
    return input.squeeze()


def edge_detection(input):
    bias=torch.zeros(1).to(device)
    input=input.clone().unsqueeze(0).unsqueeze(0)
    kernel=torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]],dtype=torch.float).to(device)
    padding_size=1
    input=F.pad(input,(padding_size,padding_size,padding_size,padding_size),mode='replicate')
    input=F.conv2d(input, kernel, bias, stride=1)
    return input.squeeze()

def standardization(stft_distribution):
    return (stft_distribution-stft_distribution.mean())/stft_distribution.std()
    

def wav_write(wav_signal: np.ndarray, wav_path: str, sample_rate: Union[int, float], scale_bit_length: bool = True, bit_length: int = 16):
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

def normalize_e(stft_mag):
    return torch.log(stft_mag+1)

def denormalize_e(stft_mag):
    return torch.pow(math.e,stft_mag)-1

def normalize_10(stft_mag):
    return torch.log10(stft_mag+1)

def denormalize_10(stft_mag):
    return torch.pow(10,stft_mag)-1

def conc_tog_specphase(S, P,n_fft,hop_length):
    S = denormalize_e(S)
    P = P * np.pi
    SP = S * torch.complex(torch.cos(P),torch.sin(P))
    wav = torch.istft(SP,n_fft=n_fft,hop_length=hop_length,win_length=n_fft,window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,length=hop_length*(S.size(2)-1)+n_fft)
    return wav

def wav_preprocess(wav,n_fft,hop_length):
    stft = torch.stft(torch.tensor(wav).to(device), n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,return_complex=False)
    stft_mag=torch.abs(torch.sqrt(torch.sum(torch.pow(stft,2),dim=-1)+1e-10))
    stft_phase=(torch.atan2(stft[:,:,1].data, stft[:,:,0].data)/np.pi)
    return stft_mag,stft_phase

def GLA(S,P, n_fft, hop_length, n_iter = 1000):
    P = P * np.pi
    for i in range(n_iter):
        SP = S * torch.complex(torch.cos(P),torch.sin(P))
        wav = torch.istft(SP,n_fft=n_fft,hop_length=hop_length,win_length=n_fft,window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,length=hop_length*(S.size(2)-1)+n_fft)
        next_SP = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=torch.hann_window(n_fft+2)[1:-1].to(device),center=False,onesided=True,return_complex=False)
        P = torch.atan2(next_SP[:,:,:,1].data, next_SP[:,:,:,0].data)
    return wav

def main():
    args = parser.parse_args()
    Attacker(args)


if __name__ == '__main__':
    main()
