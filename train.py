import warnings
warnings.filterwarnings("ignore")
import os
import math
import torch
import torch.nn as nn
import traceback

import time
import numpy as np

import argparse

from utils.generic_utils import load_config, save_config_file
from utils.generic_utils import set_init_dict

from utils.generic_utils import NoamLR, binary_acc

from utils.generic_utils import save_best_checkpoint

from utils.tensorboard import TensorboardWriter
from utils.tensorboard import SummaryWriter

from utils.dataset import train_dataloader, eval_dataloader

from models.spiraconv import SpiraConvV2
from utils.audio_processor import AudioProcessor 
from sklearn.metrics import r2_score



def validation(criterion, ap, model, c, testloader, tensorboard, step,  cuda):
    padding_with_max_lenght = c.dataset['padding_with_max_lenght'] or c.dataset['split_wav_using_overlapping']
    model.zero_grad()
    model.eval()
    loss = 0 
    acc = 0
    preds = []
    targets = []
    with torch.no_grad():
        for feature, target in testloader:       
            #try:
            if cuda:
                feature = feature.cuda()
                target = target.cuda()

            output = model(feature).float()

            # Calculate loss
            if not padding_with_max_lenght:
                target = target[:, :output.shape[1],:target.shape[2]]
            loss += criterion(output, target).item()
            
            # tirado calculo de acurracy
            y_pred_tag = torch.round(output)
            acc += (y_pred_tag == target).float().sum().item()
            preds += y_pred_tag.reshape(-1).int().cpu().numpy().tolist()
            targets += target.reshape(-1).int().cpu().numpy().tolist()
         # write loss to tensorboard
            if step % c.train_config['summary_interval'] == 0:
         #       tensorboard.log_training(loss, step)
                tensorboard.add_scalar('validation_loss', loss, step)

           #     print("Write summary at step %d" % step, ' Validation: ', loss)
         
        mean_acc = acc / len(testloader.dataset)
        mean_loss = loss / len(testloader.dataset)
        # write loss to tensorboard
        if step % c.train_config['summary_interval'] == 0:
         #       tensorboard.log_training(loss, step)
                tensorboard.add_scalar('validation_loss', mean_loss, step)

           #     print("Write summary at step %d" % step, ' Validation: ', loss)
    print("Validation:\n Loss:", mean_loss)
    print("R2:", r2_score(targets, preds))
    print("Pearson:",np.corrcoef(preds, targets, rowvar=False))

    model.train()
    return mean_loss

def train(args, log_dir, checkpoint_path, trainloader, testloader, tensorboard, c, model_name, ap, cuda=True):
    padding_with_max_lenght = c.dataset['padding_with_max_lenght'] or c.dataset['split_wav_using_overlapping']
    acc = 0
    preds = []
    targets = []
    if (model_name == 'spiraconv_v2'):
        model = SpiraConvV2(c)
    #elif(model_name == 'voicesplit'):
    else:
        raise Exception(" The model '"+model_name+"' is not suported")

    if c.train_config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=c.train_config['learning_rate'], weight_decay=c.train_config['weight_decay'])
    else:
        raise Exception("The %s  not is a optimizer supported" % c.train['optimizer'])

    step = 0
    if checkpoint_path is not None:
        print("Continue training from checkpoint: %s" % checkpoint_path)
        try:
            if c.train_config['reinit_layers']:
                raise RuntimeError
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            if cuda:
                model = model.cuda()
        except:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except:
            print(" > Optimizer state is not loaded from checkpoint path, you see this mybe you change the optimizer")
        
        step = checkpoint['step']
    else:
        print("Starting new training run")
        step = 0


    if c.train_config['lr_decay']:
        scheduler = NoamLR(optimizer,
                           warmup_steps=c.train_config['warmup_steps'],
                           last_epoch=step - 1)
    else:
        scheduler = None
    # convert model from cuda
    if cuda:
        model = model.cuda()

    # define loss function //mudado para MSE
    criterion = nn.MSELoss()
    eval_criterion = nn.MSELoss()

    best_loss = float('inf')

    # early stop definitions
    early_epochs = 0

    model.train()
    for epoch in range(c.train_config['epochs']):
        for feature, target in trainloader:
                if cuda:
                    feature = feature.cuda()
                    target = target.cuda()

                output = model(feature)

                # Calculate loss 
                # adjust target dim
                if not padding_with_max_lenght:
                    target = target[:, :output.shape[1],:]
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step() 
                y_pred_tag = torch.round(output)
                acc += (y_pred_tag == target).float().sum().item()
                preds += y_pred_tag.reshape(-1).int().cpu().numpy().tolist()
                targets += target.reshape(-1).int().cpu().numpy().tolist()
                # update lr decay scheme
                if scheduler:
                    scheduler.step()
                step += 1

                loss = loss.item()
                if loss > 1e8 or math.isnan(loss):
                    print("Loss exploded to %.02f at step %d!" % (loss, step))
                    break

                # write loss to tensorboard
                if step % c.train_config['summary_interval'] == 0:
                    tensorboard.log_training(loss, step)
                   # tensorboard.add_scalar('training_loss2', nn.MSELoss(output, target), step)
                    print("Write summary at step %d" % step, ' Loss: ', loss)

                # save checkpoint file  and evaluate and save sample to tensorboard
                if step % c.train_config['checkpoint_interval'] == 0:
                    save_path = os.path.join(log_dir, 'checkpoint_%d.pt' % step)
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'config_str': str(c),
                    }, save_path)
                    print("Saved checkpoint to: %s" % save_path)
                    # run validation and save best checkpoint
                    val_loss = validation(eval_criterion, ap, model, c, testloader, tensorboard, step,  cuda=cuda)
                    best_loss, _ = save_best_checkpoint(log_dir, model, optimizer, c, step, val_loss, best_loss, early_epochs if c.train_config['early_stop_epochs'] != 0 else None)
        
        print('=================================================')
        print("Epoch %d End !"%epoch)
        print('=================================================')
        # run validation and save best checkpoint at end epoch
        val_loss = validation(eval_criterion, ap, model, c, testloader, tensorboard, step,  cuda=cuda)
        best_loss, early_epochs = save_best_checkpoint(log_dir, model, optimizer, c, step, val_loss, best_loss,  early_epochs if c.train_config['early_stop_epochs'] != 0 else None)
        print("\n Treino: \n MSE:", loss)
        #print("MSE2:", val_loss) val_loss é o mesmo da validação
        print("R2:", r2_score(targets, preds))
        print("Pearson:",np.corrcoef(preds, targets, rowvar=False))
       # print("alvo:",targets)
       # print("predição:",preds)
        
        if c.train_config['early_stop_epochs'] != 0:
            if early_epochs is not None:
                if early_epochs >= c.train_config['early_stop_epochs']:
                    break # stop train
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, required=True,
                        help="json file with configurations")
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help="path of checkpoint pt file, for continue training")
    parser.add_argument('-s', '--seed', type=int, default=None,
                        help="Seed for training")
    args = parser.parse_args()

    c = load_config(args.config_path)
    ap = AudioProcessor(**c.audio)
    if args.seed is None:
        log_path = os.path.join(c.train_config['logs_path'], c.model_name)
    else:
        log_path = os.path.join(os.path.join(c.train_config['logs_path'], str(args.seed)), c.model_name)
        c.train_config['seed'] = args.seed

    os.makedirs(log_path, exist_ok=True)

    tensorboard = TensorboardWriter(os.path.join(log_path,'tensorboard'))

    train_dataloader = train_dataloader(c, ap)
    max_seq_len = train_dataloader.dataset.get_max_seq_lenght()
    c.dataset['max_seq_len'] = max_seq_len
    
    # save config in train dir, its necessary for test before train and reproducity
    save_config_file(c, os.path.join(log_path,'config.json'))

    eval_dataloader = eval_dataloader(c, ap, max_seq_len=max_seq_len)
    
    train(args, log_path, args.checkpoint_path, train_dataloader, eval_dataloader, tensorboard, c, c.model_name, ap, cuda=True)
