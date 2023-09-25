from config import Config
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torchaudio
import torchvision
import torch

# https://github.com/musikalkemist/audioDataAugmentationTutorial/blob/main/3/dataaugmentation.py
def add_white_noise(signal, noise_percentage_factor):
    signal2 = signal.numpy()
    noise = np.random.normal(0, signal2.std(), signal2.size)
    noise2 = torch.from_numpy(noise).float()
    augmented_signal = signal + noise2 * noise_percentage_factor
    
    return augmented_signal


class AudioFragment:
  def __init__(self):
    self.path = ""
    self.start = ""
    self.end = ""
    self.duration = 0.0
    self.label = 0.0

  def calculate_duration(self):
    wave = self.get_audio_wave()

    self.duration = wave.shape[1] / Config.Audio.sample_rate

  def get_audio_wave(self): 
    wave, sample_rate = torchaudio.load(self.path, normalize=True)
    if Config.Train.noise == True:
      wave = add_white_noise(wave, Config.Train.noise_factor)
    if sample_rate != Config.Audio.sample_rate:
      resample = torchaudio.transforms.Resample(sample_rate, Config.Audio.sample_rate)
      wave = resample(wave) #aqui q coloca o ruido
    return wave

  def get_fragment_spectrogram(self):
    # assumindo que MelSpectrogram não é thread safe, instanciando um por fragmento
    if Config.Audio.MelSpectrogram == True and Config.Audio.MFCC == False:
      audio_processor = torchaudio.transforms.MelSpectrogram(
        sample_rate = Config.Audio.sample_rate,
        n_fft = Config.Audio.n_fft,
        win_length = Config.Audio.win_length,
        hop_length = Config.Audio.hop_length,
        n_mels = Config.Audio.num_mels,
        f_min = Config.Audio.mel_fmin,
        f_max = Config.Audio.mel_fmax
      )
    elif Config.Audio.MFCC == True and Config.Audio.MelSpectrogram == False:
      audio_processor = torchaudio.transforms.MFCC(
        sample_rate = Config.Audio.sample_rate, 
        n_mfcc = Config.Audio.num_mfcc, 
        log_mels = Config.Audio.log_mels, 
        melkwargs = {'n_fft':Config.Audio.n_fft, 'win_length':Config.Audio.win_length,
                    'hop_length':Config.Audio.hop_length, 'n_mels':Config.Audio.num_mels}
      )
    elif Config.Audio.MelSpectrogram == True and Config.Audio.MFCC == True:
      print("Erro nas definições de Audio")
      import sys; sys.exit(0)
    wave = self.get_audio_wave()
    i = self.start * Config.Audio.sample_rate
    j = self.end * Config.Audio.sample_rate
    wave = wave[:, i:j]
    spectrogram = audio_processor(wave)
    return spectrogram

  def show_spectrogram(self):
    spec = self.get_fragment_spectrogram()[0].numpy()
    librosa.display.specshow(librosa.power_to_db(spec, ref=np.max))
    #power_spectrum, freqencies_found, time, image_axis = plt.specgram(spec, Fs=Config.Audio.sample_rate)
    plt.xlabel('Time')
    plt.ylabel('Mel-Frequency')
    plt.show()
    print(spec.shape)
    #import sys; sys.exit(0)

  def __str__(self):
    return (
      f"arquivo: {self.path}\n" +
      f"rótulo: {self.label}\n" +
      f"duração: {self.duration}\n" +
      f"início: {self.start}\n" +
      f"fim: {self.end}\n"
    )


