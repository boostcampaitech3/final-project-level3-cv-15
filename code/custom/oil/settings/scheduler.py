from torch.optim.lr_scheduler import StepLR

def get_scheduler(optimizer, args):
    if args.scheduler == 'steplr':
        StepLR(optimizer, 20, gamma=0.5)
        
    return scheduler