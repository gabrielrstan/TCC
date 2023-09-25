# configurações gerais
class Config:
  # semente aleatória para reproduzir experimentos
  seed = 42

  # configurações do dataset
  class Dataset:
    # descolamento em segundos ao dividir o áudio grande em áudios menores
    step = 1

    # tamanho em que um áudio grande é quebrado em áudios menores
    win_length = 4

    # caminhos para os arquivos
    dataset_folder = "/home/gabrielrstan/CodigoNovo/codigo/Tinder/"
    train_csv = "metadata_treino3_novo.csv"
    eval_csv = "metadata_eval3.csv"
    test_csv = "metadata_teste3.csv"

  # configurações para carregar os áudios e extrair espectrogramas
  class Audio:
    # taxa de amostragem
    sample_rate = 16000
    # número de coeficientes para transformada de fourier (tamanho do eixo y)
    n_fft = 1200
    # tamanho da janela usada na transformada de fourier
    win_length = 400
    # tamanho do salto da janela da transformada de fourier
    hop_length = 160
    # número de canais de mel ao converter o espectrograma para mel espectrograma
    num_mels = 40
    # frequência mínima para o mel-spec
    #  ~50 para homens e ~95 para mulheres
    mel_fmin = 0.0
    # frequência máxima para o mel-spec.
    # um bom valor é 8000.0
    mel_fmax = None
    #numero de canais de mfcc
    num_mfcc = 40
    #whether to use log-mel spectrograms instead of db-scaled
    log_mels = False
    #modo espetogramas de mel
    MelSpectrogram = False
    #modo MFCC
    MFCC = True

  # configurações de treino
  class Train:
    # localização dos arquivos de log
    log_path = "/home/gabrielrstan/CodigoNovo/codigo/checkpoints/4.0/4.6/4.6-MFCC"
    # quando ativo, só vai carregar 2 áudios na memória e vai fazer um janelamento menor
    debug = False
    # tamanho do batch durante o treino
    batch_size = 20
    # este campo é preenchido automaticamente pelo script e armazena o número de colunas
    # esperado no spec, melspec ou mfcc
    max_seq_len = 0
    # número de instâncias trabalhando em parelelo quando o experimento roda na CPU
    num_works = 12
    # número de épocas
    epochs = 150
    # taxa de aprendizado
    learning_rate = 1e-5
    # weight decay
    weight_decay = 0.01
    noise = False
    # noise factor
    noise_factor = 0.25

  # configurações do Modelo
  class Model:
    # número de neurônios na primeira camada densa
    fc1_dim = 100
    # número de neurônios na segunda camada densa
    fc2_dim = 1

  class Test:
    # localização dos arquivos de log para carregar o best checkpoint
    log_path = "/home/gabrielrstan/CodigoNovo/codigo/checkpoints/4.0/4.6/4.6-MFCC/best_checkpoint.pt"
    # tamanho do batch durante o teste
    batch_size = 15
    # número de instâncias trabalhando em parelelo quando o experimento roda na CPU
    num_works = 12
