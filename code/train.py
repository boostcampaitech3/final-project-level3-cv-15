import torch
import argparse
import os
import shutil
from importlib import import_module

from tqdm import tqdm
# from dataset.base_dataset import CustomDataset
# from utils.train_method import train
from utils.set_seed import setSeed

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import warnings

warnings.filterwarnings('ignore')

def getArgument():
    # Custom 폴더 내 훈련 설정 목록을 선택
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str ,required=True)
    parser.add_argument('--arg_n',type=str ,required=True)
    
    return parser.parse_known_args()[0].dir, parser.parse_known_args()[0].arg_n

def train(args, model, train_loader, device,  criterion, optimizer):
    
    model.train()
    
    corrects=0
    count = 0
    losses = []
    
    train_pbar = tqdm(train_loader)
    for step,(images,labels) in enumerate(train_pbar):
        train_pbar.set_description('Train')
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        
        loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
        losses.append(loss.item())
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()

        if not args.regression:    
            _, preds = torch.max(outputs,1)
        else:
            preds = torch.round(outputs).squeeze().type(torch.int64)
        
        corrects += torch.sum(preds == labels.data)
        count += outputs.shape[0]

        train_pbar.set_postfix({'acc': (corrects/count).item(), 'loss' : sum(losses)/len(losses)})
    
    acc = corrects / count
    if args.wandb:
        wandb.log({'train/accuracy' : acc,
                    'train/loss' : sum(losses)/len(losses)})
    
    return acc

def valid(args, model, valid_loader, device,  criterion, optimizer):
    model.eval()
    
    all_preds, all_labels, all_regs = [], [], []
    corrects=0
    count = 0
    losses, f1_items, recall_items, precision_items = [], [], [], []
    
    valid_pbar = tqdm(valid_loader)
    with torch.no_grad():
        for images,labels in valid_pbar:
            valid_pbar.set_description('Valid')
            images = images.to(device)
            all_labels += list(map(lambda x : x.item(), labels))
            labels = labels.to(device)
            
            outputs= model(images)
            all_regs += outputs.squeeze().tolist()
            loss = criterion(outputs.to(torch.float32), labels.to(torch.float32))
            losses.append(loss.item())

            if not args.regression:
                _, preds = torch.max(outputs,1)
            else:
                preds = torch.round(outputs).squeeze().type(torch.int64)
            
            all_preds += (preds.cpu().detach().tolist())
            corrects += torch.sum(preds == labels.data)
            count += outputs.shape[0]
            
            ## f1 score
            f1_item = f1_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average = 'macro') # 추가 
            f1_items.append(f1_item) 

            ## recall
            recall_item = recall_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average = 'macro')
            recall_items.append(recall_item)

            ## precision
            precision_item = precision_score(labels.cpu().detach().numpy(), preds.cpu().detach().numpy(), average = 'macro')
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

    cf_matrix = confusion_matrix(all_preds, all_labels, labels = range(5))
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in range(5)], columns = [i for i in range(5)])
    fig, ax = plt.subplots(figsize = (10,7), ncols=2)
    g = sns.heatmap(df_cm, annot=True, cmap=sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True), ax=ax[0])
    g.set_xlabel("Actual")
    g.set_ylabel("Predicted")

    df = pd.DataFrame({'reg' : all_regs, 'lab' : all_labels })
    df = df.sort_values(['lab', 'reg'])
    df.reset_index(inplace=True, drop=True)

    g = ax[1].plot(df.reg, label='predicted')
    g = ax[1].plot(df.lab, label='actual')
    g = ax[1].legend()

    if args.wandb:
        wandb.log({'valid/accuracy' : acc,
                    'valid/loss' : val_loss,
                    'valid/F1_score' : f1,
                    'valid/recall' : recall,
                    'valid/precision' : precision,
                    "Confusion_Matrix" : wandb.Image(g),
                    })
    
    return {"accuracy": acc, 
            "loss" : loss, 
            "f1_score" : f1,
            "recall_score" : recall, 
            "precision" : precision, 
            }

def main(custom_dir, arg_n):

    arg = getattr(import_module(f"custom.{custom_dir}.settings.{arg_n}"), "getArg")()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    setSeed(arg.seed)

    train_transform, val_transform = getattr(import_module(f"custom.{custom_dir}.settings.transform"), "getTransform")()

    trainLoader, valLoader = getattr(import_module(f"custom.{custom_dir}.settings.dataloader"), "getDataloader")(
        train_transform, val_transform, arg.batch, arg.train_worker, arg.valid_worker)

    if not arg.regression:
        model = getattr(import_module(f"custom.{custom_dir}.settings.model"), "getModel")(arg.modeltype, device, arg.modelname)
    else:
        model = getattr(import_module(f"custom.{custom_dir}.settings.model"), "getRegressionModel")(arg.modeltype, device, arg.modelname)
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
        
        train_acc = train(arg, model, trainLoader, device, criterion, optimizer)

        metrics = valid(arg, model, valLoader,device, criterion, optimizer)
        goal_metric = metrics[arg.metric]
        
        if goal_metric > best_metric :
            print(f'valid {arg.metric} is imporved from {best_metric:.4f} -> {goal_metric:.4f}... model saved! {save_name}')
            best_metric = goal_metric
            if arg.wandb:
                wandb.run.summary['Best_metric'] = best_metric
            try:
                os.remove(os.path.join(outputPath+"/models", save_name))
            except:
                pass
            save_name = f"{arg.custom_name}_best_{str(best_metric.item())[:4]}"

            torch.save(model, os.path.join(outputPath+"/models", save_name+'.pt'))
        
        if arg.scheduler in ["reduceLROnPlateau", "cycliclr"]:
            scheduler.step(goal_metric)
        else:
            scheduler.step()

if __name__=="__main__":
    custom_dir, arg_n = getArgument()
    main(custom_dir, arg_n)