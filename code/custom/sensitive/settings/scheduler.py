from torch.optim.lr_scheduler import StepLR

def getScheduler(optimizer, scheduler):
    if scheduler == 'steplr':
        scheduler = StepLR(optimizer, 20, gamma=0.5)
        
    return scheduler