from torch.optim.lr_scheduler import StepLR

def getScheduler(optimizer):
    if args.scheduler == 'steplr':
        StepLR(optimizer, 20, gamma=0.5)
        
    return scheduler