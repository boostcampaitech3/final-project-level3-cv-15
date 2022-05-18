import torch
import argparse
import os
import shutil
from importlib import import_module

from tqdm import tqdm
# from dataset.base_dataset import CustomDataset
# from utils.train_method import train
from utils.set_seed import setSeed

from sklearn.metrics import f1_score

import numpy as np

def getArgument():
    # Custom 폴더 내 훈련 설정 목록을 선택
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str ,required=True)
    parser.add_argument('--arg_n',type=str ,required=True)
    
    return parser.parse_known_args()[0].dir, parser.parse_known_args()[0].arg_n

def train(args, model, train_loader, device,  criterion, optimizer):
        
    # criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    corrects=0
    count = 0
    
    for step,(images,labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs= model(images)
        # outputs = outputs[0]
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss : {str(loss.item())}")
            
        _, preds = torch.max(outputs,1)
        
        corrects += torch.sum(preds == labels.data)
        count += outputs.shape[0]
    
    acc = corrects / count
    
    return acc

def valid(args, model, valid_loader, device,  criterion, optimizer):
    model.eval()
    
    corrects=0
    count = 0
    best_f1 = 0 # 추가
    f1_items = []
    
    for images,labels in valid_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs= model(images)

        _, preds = torch.max(outputs,1)
        corrects += torch.sum(preds == labels.data)
        count += outputs.shape[0]
        
        f1_item = f1_score(labels.cpu(), preds.cpu(), average = 'macro') # 추가 
        f1_items.append(f1_item) 
    
    
    acc = corrects / count
    f1 = np.sum(f1_items) / len(valid_loader) #추가
    
    print(f'VALID ACC : {acc} VALID F1 : {f1} \n')
    
    return acc, outputs, f1

def main(custom_dir, arg_n):

    arg = getattr(import_module(f"custom.{custom_dir}.settings.{arg_n}"), "getArg")()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    setSeed(arg.seed)

    train_transform, val_transform = getattr(import_module(f"custom.{custom_dir}.settings.transform"), "getTransform")()

    
    # train_dataset = CustomDataset(data_dir=os.path.join(arg.image_root,arg.train_json),image_root=arg.image_root, mode='train', transform=train_transform)
    # val_dataset = CustomDataset(data_dir=os.path.join(arg.image_root,arg.val_json),image_root=arg.image_root, mode='val', transform=val_transform)

    trainLoader, valLoader = getattr(import_module(f"custom.{custom_dir}.settings.dataloader"), "getDataloader")(
        train_transform, val_transform, arg.batch, arg.train_worker, arg.valid_worker)

    model = getattr(import_module(f"custom.{custom_dir}.settings.model"), "getModel")(arg.modeltype, device, arg.modelname)
    criterion = getattr(import_module(f"custom.{custom_dir}.settings.loss"), "getLoss")(arg.loss)

#     optimizer, scheduler = getattr(import_module(f"custom.{custom_dir}.settings.opt_scheduler"), "getOptAndScheduler")(model, arg.lr)
    
    optimizer = getattr(import_module(f"custom.{custom_dir}.settings.optimizer"), "getOptimizer")(model, arg.optimizer, arg.lr)
    scheduler = getattr(import_module(f"custom.{custom_dir}.settings.scheduler"), "getScheduler")(optimizer, arg.scheduler)

    outputPath = os.path.join(arg.output_path, arg.custom_name)

    #output Path 내 설정 저장
    shutil.copytree(f"custom/{custom_dir}",outputPath)
    os.makedirs(outputPath+"/models", exist_ok=True)
    
    # wandb
    # if arg.wandb:
    #         from utils.wandb_method import WandBMethod
    #         WandBMethod.login(arg, model, criterion)

    # train(arg.epoch, model, trainLoader, valLoader, criterion, optimizer,scheduler, outputPath, arg.save_capacity, device, arg.wandb)
    best_acc = -1
    best_f1 = -1
    
    iter_n = 1
    
    for epoch in tqdm(range(arg.epoch)):
        print(f'Epoch {epoch+1}/{arg.epoch}')
        
        train_acc = train(arg, model, trainLoader, device, criterion, optimizer)
        
        valid_acc , outputs, f1_score = valid(arg, model, valLoader,device, criterion, optimizer)
        
        if arg.f1 :
            if f1_score > best_f1 :
                best_f1 = f1_score

                save_name = f"{arg.custom_name}_{str(best_f1.item())[:4]}"

                torch.save(model, os.path.join(outputPath+"/models", save_name))
                print(f'model saved! {save_name}')
            
        else :  
            if valid_acc > best_acc :
                best_acc = valid_acc

                save_name = f"{arg.custom_name}_{str(best_acc.item())[:4]}"

                torch.save(model, os.path.join(outputPath+"/models", save_name))
                print(f'model saved! {save_name}')
        
        scheduler.step()

if __name__=="__main__":
    custom_dir, arg_n = getArgument()
    main(custom_dir, arg_n)