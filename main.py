from config import Config
from spira_dataset import SpiraDataset
from spiranet import SpiraNet
import os
import torch
from tensorboardX import SummaryWriter

class Main:

  best_loss = 0

  def __init__(self):
    # criando pasta para log
    log_path = os.path.join(
      Config.Train.log_path,
      "seed_" + str(Config.seed)
    )
    os.makedirs(log_path, exist_ok=True)
    self.tensorboard = SummaryWriter(os.path.join(Config.Train.log_path, 'tensorboard'))
    self.train()
    self.save_config_treino()
    print("sucesso")

  def train(self):
    loss = 0
    dataloader = SpiraDataset.train_dataloader()
    dataloader2 = SpiraDataset.eval_dataloader()
    model = SpiraNet()
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.Train.learning_rate)
    criterion = torch.nn.MSELoss(reduction='sum')

    model.train() # para usar regularização
    for epoch in range(Config.Train.epochs):
      print("época", epoch)
      print("~~~ TREINO ~~~")
      cost = 0
      preds = []
      targets = []
      for feature, target in dataloader:
        feature = feature.cuda()
        target = target.cuda()
        output = model(feature)
        output = output.squeeze()
        target = target.float()
        loss = criterion(output, target)
        cost += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        preds += output.reshape(-1).float().cpu().detach().numpy().tolist()
        targets += target.reshape(-1).float().cpu().detach().numpy().tolist()

      mean_loss = cost / len(dataloader.dataset)
      self.tensorboard.add_scalar('Train_loss', mean_loss, epoch)
      print("custo", mean_loss)
      '''for res in ["Saida Treino: {}  Alvo Treino: {}".format(x,y) for x,y in zip(preds,targets)] :
            print(res)'''
      print("~~~ FIM TREINO ~~~\n")

      self.validation(criterion, model, dataloader2, epoch, optimizer)
      if Config.Train.debug and epoch >= 100:
        break
    
  def validation(self, criterion, model, dataloader, epoch, optimizer):
    preds = []
    targets = []
    mean_loss = 0
    loss = 0
    model.zero_grad()
    model.eval()
    model = model.cuda()
    print("\n~~~ VALIDAÇÂO ~~~")
    cost = 0
    with torch.no_grad():
        for feature, target in dataloader:
          feature = feature.cuda()
          target = target.cuda()
          output = model(feature)
          output = output.squeeze()
          target = target.float()
          loss = criterion(output, target)
          cost += loss.item()
          preds += output.reshape(-1).float().cpu().detach().numpy().tolist()
          targets += target.reshape(-1).float().cpu().detach().numpy().tolist()
        mean_loss = cost / len(dataloader.dataset)
        self.tensorboard.add_scalar('validation_loss', mean_loss, epoch)
    print("custo", mean_loss)
    if epoch == 0:
      Main.best_loss = mean_loss
      self.save_best_checkpoint(Config.Train.log_path, model, optimizer, epoch, mean_loss)
    elif mean_loss < Main.best_loss:
      Main.best_loss = mean_loss
      self.save_best_checkpoint(Config.Train.log_path, model, optimizer, epoch, mean_loss)

    '''for res in ["Saida Validação: {}  Alvo Validação: {}".format(x,y) for x,y in zip(preds,targets)] :
              print(res)'''
    print("~~~ FIM VALIDAÇÂO ~~~\n")
    model.train()


  def save_config_treino(self):
    f = open("config_treino.txt", "w")
    f.write("Seed = " + str(Config.seed))
    f.write("\nBatch Size = " + str(Config.Train.batch_size))
    f.write("\nNumero de Processos = " + str(Config.Train.num_works))
    f.write("\nEpocas = " + str(Config.Train.epochs))
    f.write("\nTaxa de Aprendizado = " + str(Config.Train.learning_rate))
    f.write("\nWeight Decay = " + str(Config.Train.weight_decay))
    f.close()

  def save_best_checkpoint(self, log_path, model, optimizer, epoch, mean_loss):
        save_path = os.path.join(log_path, 'best_checkpoint.pt')
        torch.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'loss' : mean_loss
        }, save_path)
        print("\nBEST MODEL:\n Loss:{}\n Epoca:{}".format(
            mean_loss, epoch))
     

Main()
