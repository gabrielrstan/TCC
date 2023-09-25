from config import Config
from spira_dataset import SpiraDataset
from spiranet import SpiraNet
import os
import torch
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from torchmetrics import R2Score


class Test:

    def __init__(self):
        self.test()
        self.save_config_test()


    def test(self):
        loss = 0
        dataloader = SpiraDataset.test_dataloader()
        model = SpiraNet()
        optimizer = torch.optim.Adam(model.parameters(), lr=Config.Train.learning_rate)
        criterion = torch.nn.MSELoss(reduction='sum')

        #carregar best checkpoint
        checkpoint = torch.load(Config.Test.log_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        #Começando o teste
        model.zero_grad()
        model.eval()
        model = model.cuda()
        #print(len(dataloader))
        with torch.no_grad():
            preds = []
            targets = []
            cost = 0
            for feature, target in dataloader:
                #print("oi")
                #print(len(dataloader))
                feature = feature.cuda()
                target = target.cuda()
                output = model(feature)
                output = output.squeeze()
                target = target.float()
                loss = criterion(output, target)
                cost += loss.item()
                preds += output.reshape(-1).float().cpu().detach().numpy().tolist()
                targets += target.reshape(-1).float().cpu().detach().numpy().tolist()
            #print(len(dataloader))
            mean_loss = cost / len(dataloader.dataset)
            #calculo R2 askpython.com/python/coefficient-of-determination
            corr_matrix = np.corrcoef(targets, preds)
            corr = corr_matrix[0,1]
            R_sq = corr**2
            #Calculo Pearson https://www.geeksforgeeks.org/python-pearson-correlation-test-between-two-variables/
            pearson, _ = pearsonr(preds, targets)
            
            print("Loss:", mean_loss)
            print("R2:", R_sq)
            print("Pearson:", pearson )
            '''for res in ["Saida: {}  Alvo: {}".format(x,y) for x,y in zip(preds,targets)] :
              print(res)'''

    def save_config_test(self):
        f = open("config_teste.txt", "w")
        f.write("Batch Size = " + str(Config.Test.batch_size))
        f.write("\nNumero de Processos = " + str(Config.Test.num_works))
        f.close()

Test()    