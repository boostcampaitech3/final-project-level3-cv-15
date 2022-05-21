import torch
import argparse
import os
import shutil
from importlib import import_module

import timm

from tqdm import tqdm
# from dataset.base_dataset import CustomDataset
# from utils.train_method import train
from utils.set_seed import setSeed

from sklearn.metrics import f1_score, recall_score, precision_score

import numpy as np
import wandb
import warnings

from torchvision import models

warnings.filterwarnings('ignore')

def getArgument():
    # Custom 폴더 내 훈련 설정 목록을 선택
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str ,required=True)
    parser.add_argument('--arg_n',type=str ,required=True)
    parser.add_argument('--use_meta', action='store_true')
    
    return parser.parse_known_args()[0].dir, parser.parse_known_args()[0].arg_n, parser.parse_known_args()[0].use_meta

def train(args, model, train_loader, device,  criterion, optimizer, use_meta):
    
    model.train()
    
    corrects=0
    count = 0
    losses = []
    
    train_pbar = tqdm(train_loader)
    for step, data in enumerate(train_pbar):
        train_pbar.set_description('Train')
        if use_meta:
            images, labels, _, meta = data
            images, labels, meta = images.to(device), labels.to(device), meta.to(device)
            meta = meta.reshape(-1,1).float()
            outputs= model(images, meta)
        else:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs= model(images)

        
        loss = criterion(outputs, labels)
        losses.append(loss.item())
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        # if step % args.log_steps == 0:
        #     print(f"Training steps: {step} Loss : {str(loss.item())}")
            
        _, preds = torch.max(outputs,1)
        
        corrects += torch.sum(preds == labels.data)
        count += outputs.shape[0]

        train_pbar.set_postfix({'acc': (corrects/count).item(), 'loss' : sum(losses)/len(losses)})
    
    acc = corrects / count
    # wandb.log({'train/accuracy' : acc,
    #             'train/loss' : sum(losses)/len(losses)})
    
    return acc

def valid(args, model, valid_loader, device, criterion, optimizer, use_meta):
    model.eval()
    
    corrects=0
    count = 0
    losses, f1_items, recall_items, precision_items = [], [], [], []
    
    valid_pbar = tqdm(valid_loader)
    for step, data in enumerate(valid_pbar):
        valid_pbar.set_description('Valid')
        if use_meta:
            images, labels, _, meta = data
            images, labels, meta = images.to(device), labels.to(device), meta.to(device)
            meta = meta.reshape(-1,1).float()
            outputs= model(images, meta)
        
        else:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs= model(images)

        loss = criterion(outputs, labels)
        losses.append(loss.item())

        _, preds = torch.max(outputs,1)
        corrects += torch.sum(preds == labels.data)
        count += outputs.shape[0]
        
        ## f1 score
        f1_item = f1_score(labels.cpu(), preds.cpu(), average = 'macro') # 추가 
        f1_items.append(f1_item) 

        ## recall
        recall_item = recall_score(labels.cpu(), preds.cpu(), average = 'macro')
        recall_items.append(recall_item)

        ## precision
        precision_item = precision_score(labels.cpu(), preds.cpu(), average = 'macro')
        precision_items.append(precision_item)

        valid_pbar.set_postfix({'acc': (corrects/count).item(), 
                                'loss' : sum(losses)/len(losses),
                                'f1' : sum(f1_items)/len(f1_items),
                                'recall' : sum(recall_items)/len(recall_items),
                                'precision' : sum(precision_items)/len(precision_items)
                                })
    
    
    acc = corrects / count
    val_loss = sum(losses) / len(losses)
    f1 = sum(f1_items) / len(f1_items) #추가
    recall = sum(recall_items) / len(recall_items)
    precision = sum(precision_items) / len(precision_items)

    # wandb.log({'valid/accuracy' : acc,
    #             'valid/loss' : val_loss,
    #             'valid/F1_score' : f1,
    #             'valid/recall' : recall,
    #             'valid/precision' : precision})
    
    return {"accuracy": acc, 
            "loss" : loss, 
            "f1_score" : f1,
            "recall_score" : recall, 
            "precision" : precision, 
            }

def main(custom_dir, arg_n, use_meta):
    print("using meta : ", use_meta)
    arg = getattr(import_module(f"custom.{custom_dir}.settings.{arg_n}"), "getArg")()
    print(arg.modeltype)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setSeed(arg.seed)

    train_transform, val_transform = getattr(import_module(f"custom.{custom_dir}.settings.transform"), "getTransform")()

    trainLoader, valLoader = getattr(import_module(f"custom.{custom_dir}.settings.dataloader"), "getDataloader")(
        train_transform, val_transform, arg.batch, arg.train_worker, arg.valid_worker, use_meta)

    model = getattr(import_module(f"custom.{custom_dir}.settings.model"), "getModel")(arg.modeltype, device)
    # model = models.resnet101(pretrained=True)
    # model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=5)
    model.to(device)

    criterion = getattr(import_module(f"custom.{custom_dir}.settings.loss"), "getLoss")(arg.loss)
    
    optimizer = getattr(import_module(f"custom.{custom_dir}.settings.optimizer"), "getOptimizer")(model, arg.optimizer, arg.lr)
    scheduler = getattr(import_module(f"custom.{custom_dir}.settings.scheduler"), "getScheduler")(optimizer, arg.scheduler, arg.epoch)

    outputPath = os.path.join(arg.output_path, arg.custom_name)

    #output Path 내 설정 저장
    shutil.copytree(f"custom/{custom_dir}",outputPath)
    os.makedirs(outputPath+"/models", exist_ok=True)
    
    # wandb
    if arg.wandb:
            wandb.init(project=arg.wandb_project, entity=arg.wandb_entity, name = arg.custom_name)
            wandb.watch(model)
            wandb.run.summary['metric'] = arg.metric
            wandb.run.summary['optimizer'] = arg.optimizer
            wandb.run.summary['model'] = arg.modelname

    best_metric = 0.
    save_name = ''
    
    iter_n = 1
    
    for epoch in range(arg.epoch):
        print(f'Epoch {epoch+1}/{arg.epoch}')
        
        train_acc = train(arg, model, trainLoader, device, criterion, optimizer, use_meta)

        metrics = valid(arg, model, valLoader,device, criterion, optimizer, use_meta)
        goal_metric = metrics[arg.metric]
        
        if goal_metric > best_metric :
            print(f'valid {arg.metric} is imporved from {best_metric:.4f} -> {goal_metric:.4f}... model saved! {save_name}')
            best_metric = goal_metric
            # wandb.run.summary['Best_metric'] = best_metric
            try:
                os.remove(os.path.join(outputPath+"/models", save_name))
            except:
                pass
            save_name = f"{arg.custom_name}_best_{str(best_metric.item())[:4]}.pth"
            torch.save(model.state_dict(), os.path.join(outputPath+"/models", save_name))
        
        scheduler.step()

if __name__=="__main__":
    custom_dir, arg_n, use_meta = getArgument()
    main(custom_dir, arg_n, use_meta)