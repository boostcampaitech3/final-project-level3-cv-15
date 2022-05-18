import torch
import argparse
import os
import shutil
from importlib import import_module

from dataset.base_dataset import CustomDataset
from utils.train_method import train
from utils.set_seed import setSeed

def getArgument():
    # Custom 폴더 내 훈련 설정 목록을 선택
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir',type=str ,required=True)
    return parser.parse_known_args()[0].dir

def train(args, model, train_loader, optimizer):
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    corrects=0
    
    for step,(images,labels) in enumerate(train_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        
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
    
    acc = corrects / args.train_len
    
    return acc

def valid(args, model, valid_loader, optimizer):
    model.eval()
    
    corrects=0
    
    for images,labels in valid_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        
        outputs= model(images)

        _, preds = torch.max(outputs,1)
        corrects += torch.sum(preds == labels.data)
    
    acc = corrects / args.valid_len
    
    print(f'VALID ACC : {acc}\n')
    
    return acc, outputs

def main(custom_dir):

    arg = getattr(import_module(f"custom.{custom_dir}.settings.arg"), "getArg")()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    setSeed(arg.seed)

    train_transform, val_transform = getattr(import_module(f"custom.{custom_dir}.settings.transform"), "getTransform")()

    
    # train_dataset = CustomDataset(data_dir=os.path.join(arg.image_root,arg.train_json),image_root=arg.image_root, mode='train', transform=train_transform)
    # val_dataset = CustomDataset(data_dir=os.path.join(arg.image_root,arg.val_json),image_root=arg.image_root, mode='val', transform=val_transform)

    trainLoader, valLoader = getattr(import_module(f"custom.{custom_dir}.settings.dataloader"), "getDataloader")(
        train_transform, val_transform, arg.batch, arg.train_worker, arg.valid_worker)

    model = getattr(import_module(f"custom.{custom_dir}.settings.model"), "getModel")(arg.modeltype, device)
    criterion = getattr(import_module(f"custom.{custom_dir}.settings.loss"), "getLoss")()

    optimizer, scheduler = getattr(import_module(f"custom.{custom_dir}.settings.opt_scheduler"), "getOptAndScheduler")(model, arg.lr)
    
    optimizer = getattr(import_module(f"custom.{custom_dir}.settings.optimizer"), "getOptimizer")(model, args.optimizer, args.lr)
    scheduler = getattr(import_module(f"custom.{custom_dir}.settings.scheduler"), "getScheduler")(optimizer)

    outputPath = os.path.join(arg.output_path, arg.custom_name)

    #output Path 내 설정 저장
    shutil.copytree(f"custom/{custom_dir}",outputPath)
    os.makedirs(outputPath+"/models")
    
    # wandb
    # if arg.wandb:
    #         from utils.wandb_method import WandBMethod
    #         WandBMethod.login(arg, model, criterion)

    # train(arg.epoch, model, trainLoader, valLoader, criterion, optimizer,scheduler, outputPath, arg.save_capacity, device, arg.wandb)
#     best_acc = -1
    
#     iter_n = 1
    
#     for epoch in notebook.tqdm(range(args.n_epochs)):
#         print(f'Epoch {epoch+1}/{args.n_epochs}')
        
#         train_acc = train(args, model, train_loader, optimizer)
        
#         valid_acc , outputs = valid(args, model, valid_loader, optimizer)
        
#         if valid_acc > best_acc :
#             best_acc = valid_acc
            
#             save_name = f"{args.timm_model}_{str(best_acc.item())[:4]}"
            
#             torch.save(model, os.path.join(args.model_path, save_name))
#             print(f'model saved! {save_name}')
        
#         scheduler.step()

if __name__=="__main__":
    custom_dir = getArgument()
    main(custom_dir)