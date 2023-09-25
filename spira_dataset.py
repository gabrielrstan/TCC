from config import Config
from audio_fragment import AudioFragment
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
import random
import torch
import torchaudio

class SpiraDataset(Dataset):

  def __init__(self, csv):
    # setando as sementes aleatóriasSpiraDataset
    random.seed(Config.seed)
    torch.manual_seed(Config.seed)
    torch.cuda.manual_seed(Config.seed)
    np.random.seed(Config.seed)

    # recuperando csv
    f = os.path.join(Config.Dataset.dataset_folder, csv)
    csv = pd.read_csv(f, sep=',')
    self.fragments = []
    self.load_fragment_info(csv)

    # debug
    Config.Train.max_seq_len = int(
      ((Config.Dataset.win_length * Config.Audio.sample_rate)/Config.Audio.hop_length)+1)
    #print("usando janelamento; max_seq_len = ", Config.Train.max_seq_len)

  def __getitem__(self, i):
    melspec = self.fragments[i].get_fragment_spectrogram()
    label = self.fragments[i].label
    return melspec, label

  def __len__(self):
    return len(self.fragments)

  def load_fragment_info(self, csv):
    # fase 1: obtendo metados dos arquivos
    originals = []
    #print("computando durações...")
    for i in range(len(csv)):
      fragment = AudioFragment()
      fragment.path = os.path.join(Config.Dataset.dataset_folder, csv["arquivo"][i])
      fragment.label = csv["oxigenacao"][i]
      fragment.calculate_duration()
      # ToDo: para teste, parar processo após 10 iterações
      originals.append(fragment)
      if Config.Train.debug and i >= 3:
        Config.Train.batch_size = 3
        break
    #print("durações computadas para", i, "audios")

    # fase 2: fatiando os arquivos em pedaços menores
    for original in originals:
      start = 0
      end = int(original.duration) - Config.Dataset.win_length + 1
      step = Config.Dataset.step
      for i in range(start, end, step):
        fragment = AudioFragment()
        fragment.path = original.path
        fragment.label = original.label
        fragment.duration = original.duration
        fragment.start = i
        fragment.end = i + Config.Dataset.win_length
        self.fragments.append(fragment)

    # para visualizar espectrogramas de mel e debugar
    #for i in [0, 1, 2, 10, 11, 12, 17, 18, 19]:
     # self.fragments[i].show_spectrogram()

  def train_dataloader():
    return DataLoader(
      SpiraDataset(Config.Dataset.train_csv),
      batch_size=Config.Train.batch_size,
      shuffle=True,
      num_workers=Config.Train.num_works,
      pin_memory=True, # possivelmente pode deixar o default (falso)
      drop_last=True)

  def eval_dataloader():
    return DataLoader(
      SpiraDataset(Config.Dataset.eval_csv),
      batch_size=Config.Train.batch_size,
      shuffle=True,
      num_workers=Config.Train.num_works,
      pin_memory=True, # possivelmente pode deixar o default (falso)
      drop_last=True)

  def test_dataloader():
    return DataLoader(
      SpiraDataset(Config.Dataset.test_csv),
      batch_size=Config.Test.batch_size,
      shuffle=True,
      num_workers=Config.Test.num_works,
      pin_memory=True, # possivelmente pode deixar o default (falso)
      drop_last=True)