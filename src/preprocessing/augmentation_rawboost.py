# Copyright ArcFace_ADD (https://github.com/juliusmuet/ArcFace_ADD). All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0)

import random
import copy
from types import SimpleNamespace
import torch
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class RawBoostAugmentation:
    """
    RawBoost data augmentation class for waveform enhancement.

    This class simulates convolutive noise, additive noise, and reverberation
    augmentations controlled by an algorithm index (1-7) and parameters.

    Processing logic and parameter settings are from https://github.com/TakHemlata/RawBoost-antispoofing.

    Args:
        config (dict):
            - 'algorithm' (int): RawBoost algorithm index (default: None). If None, random algorithm is picked.
            - 'prob' (float): Probability of applying RawBoost augmentation (default 0.0).
            - 'sample_rate' (int): Sampling rate of input waveforms (default: 16000).
    """

    def __init__(self, config):
        self.algo = config.get('algorithm', None)
        self.prob = config.get('prob', 0.0)
        self.sample_rate = config.get('sample_rate', 16000)
        self.args = SimpleNamespace(**{
            'algo': 0,
            'nBands': 5,
            'minF': 20,
            'maxF': 8000,
            'minBW': 100,
            'maxBW': 1000,
            'minCoeff': 10,
            'maxCoeff': 100,
            'minG': 0,
            'maxG': 0,
            'minBiasLinNonLin': 5,
            'maxBiasLinNonLin': 20,
            'N_f': 5,
            'P': 10,
            'g_sd': 2,
            'SNRmin': 10,
            'SNRmax': 40,
        })

        logger.info(f"Initialised RawBoostAugmentation with parameters:\n{self}")


    def __str__(self):
        return (
            f"RawBoostAugmentation(probability={self.prob}, "
            f"algorithm={self.algo}, "
            f"sample_rate={self.sample_rate})"
        )


    def __call__(self, wav):
        """
        Apply RawBoost augmentation to the input waveform.

        Args:
            wav (torch.Tensor): 1D tensor containing mono audio waveform.

        Returns:
            torch.Tensor: 1D tensor of the augmented waveform tensor.
        """
        wav_np = wav.numpy()
        
        if self.algo is None:
            algo = random.randint(1, 8)
        else:
            algo = self.algo
        
        wav_np = self.process_Rawboost_feature(wav_np, algo, self.args)

        return torch.from_numpy(wav_np.astype(np.float32))
    

    def process_Rawboost_feature(self, wav, algo, args):
        # Data process by Convolutive noise (1st algo)
        if algo == 1:
            wav = self.LnL_convolutive_noise(wav,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,
                                             args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,self.sample_rate)
                                
        # Data process by Impulsive noise (2nd algo)
        elif algo == 2:
            wav = self.ISD_additive_noise(wav, args.P, args.g_sd)
                                
        # Data process by coloured additive noise (3rd algo)
        elif algo == 3:
            wav = self.SSI_additive_noise(wav,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,
                                          args.maxCoeff,args.minG,args.maxG,self.sample_rate)
        
        # Data process by all 3 algo. together in series (1+2+3)
        elif algo == 4:
            wav = self.LnL_convolutive_noise(wav,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                                             args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,self.sample_rate)                         
            wav = self.ISD_additive_noise(wav, args.P, args.g_sd)  
            wav = self.SSI_additive_noise(wav,args.SNRmin,args.SNRmax,args.nBands,args.minF,
                                          args.maxF,args.minBW,args.maxBW,args.minCoeff,args.maxCoeff,args.minG,args.maxG,self.sample_rate)                 

        # Data process by 1st two algo. together in series (1+2)
        elif algo == 5:
            wav = self.LnL_convolutive_noise(wav,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                                             args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,self.sample_rate)                         
            wav= self.ISD_additive_noise(wav, args.P, args.g_sd)                
                                

        # Data process by 1st and 3rd algo. together in series (1+3)
        elif algo == 6:  
            wav = self.LnL_convolutive_noise(wav,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                                             args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,self.sample_rate)                         
            wav = self.SSI_additive_noise(wav,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,
                                          args.maxCoeff,args.minG,args.maxG,self.sample_rate) 

        # Data process by 2nd and 3rd algo. together in series (2+3)
        elif algo == 7: 
            wav = self.ISD_additive_noise(wav, args.P, args.g_sd)
            wav = self.SSI_additive_noise(wav,args.SNRmin,args.SNRmax,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,args.minCoeff,
                                          args.maxCoeff,args.minG,args.maxG,self.sample_rate) 
    
        # Data process by 1st two algo. together in Parallel (1||2)
        elif algo == 8:
            wav1 = self.LnL_convolutive_noise(wav,args.N_f,args.nBands,args.minF,args.maxF,args.minBW,args.maxBW,
                                              args.minCoeff,args.maxCoeff,args.minG,args.maxG,args.minBiasLinNonLin,args.maxBiasLinNonLin,self.sample_rate)                         
            wav2 = self.ISD_additive_noise(wav, args.P, args.g_sd)

            wav_para = wav1 + wav2
            wav = self.normWav(wav_para,0)  #normalised resultant waveform
    
        # original data without Rawboost processing           
        else:
            wav = wav
        
        return wav


    def randRange(self, x1, x2, integer):
        y = np.random.uniform(low=x1, high=x2, size=(1,))
        if integer:
            y = int(y)
        return y


    def normWav(self, x, always):
        if always:
            x = x/np.amax(abs(x))
        elif np.amax(abs(x)) > 1:
                x = x/np.amax(abs(x))
        return x


    def genNotchCoeffs(self, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
        b = 1
        for i in range(0, nBands):
            fc = self.randRange(minF,maxF,0);
            bw = self.randRange(minBW,maxBW,0);
            c = self.randRange(minCoeff,maxCoeff,1);
            
            if c/2 == int(c/2):
                c = c + 1
            f1 = fc - bw/2
            f2 = fc + bw/2
            if f1 <= 0:
                f1 = 1/1000
            if f2 >= fs/2:
                f2 =  fs/2-1/1000
            b = np.convolve(signal.firwin(c, [float(f1), float(f2)], window='hamming', fs=fs),b)

        G = self.randRange(minG,maxG,0); 
        _, h = signal.freqz(b, 1, fs=fs)    
        b = pow(10, G/20)*b/np.amax(abs(h))   
        return b


    def filterFIR(self, x, b):
        N = b.shape[0] + 1
        xpad = np.pad(x, (0, N), 'constant')
        y = signal.lfilter(b, 1, xpad)
        y = y[int(N/2):int(y.shape[0]-N/2)]
        return y


    def LnL_convolutive_noise(self, x, N_f, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, minBiasLinNonLin, maxBiasLinNonLin, fs):
        # Linear and non-linear convolutive noise
        y = [0] * x.shape[0]
        for i in range(0, N_f):
            if i == 1:
                minG = minG-minBiasLinNonLin;
                maxG = maxG-maxBiasLinNonLin;
            b = self.genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
            y = y + self.filterFIR(np.power(x, (i+1)),  b)     
        y = y - np.mean(y)
        y = self.normWav(y,0)
        return y


    def ISD_additive_noise(self, x, P, g_sd):
        # Impulsive signal dependent noise
        beta = self.randRange(0, P, 0)
        
        y = copy.deepcopy(x)
        x_len = x.shape[0]
        n = int(x_len*(beta/100))
        p = np.random.permutation(x_len)[:n]
        f_r= np.multiply(((2*np.random.rand(p.shape[0]))-1),((2*np.random.rand(p.shape[0]))-1))
        r = g_sd * x[p] * f_r
        y[p] = x[p] + r
        y = self.normWav(y,0)
        return y


    def SSI_additive_noise(self, x, SNRmin, SNRmax, nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs):
        # Stationary signal independent noise
        noise = np.random.normal(0, 1, x.shape[0])
        b = self.genNotchCoeffs(nBands,minF,maxF,minBW,maxBW,minCoeff,maxCoeff,minG,maxG,fs)
        noise = self.filterFIR(noise, b)
        noise = self.normWav(noise,1)
        SNR = self.randRange(SNRmin, SNRmax, 0)
        noise = noise / np.linalg.norm(noise,2) * np.linalg.norm(x,2) / 10.0**(0.05 * SNR)
        x = x + noise
        return x
