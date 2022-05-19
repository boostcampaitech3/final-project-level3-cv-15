from torch.optim import lr_scheduler

def getScheduler(optimizer, scheduler, epoch):
    
    if scheduler == 'lambdalr':
        scheduler = lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    if scheduler == 'multiplicativelr':
        scheduler = lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)

    if scheduler == 'steplr':
        scheduler = lr_scheduler.StepLR(optimizer, 20, gamma=0.5)

    if scheduler == 'multisteplr':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30,80], gamma=0.5)

    if scheduler == 'exponentiallr':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.5)

    if scheduler == 'cosineannealinglr':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0)

    if scheduler == 'onecyclelr':
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=10, epochs=10,anneal_strategy='linear')

    if scheduler == 'cosineannealingwarmrestart':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=0.00001)
        
    # metric에 따른 lr변환
    if scheduler == 'reduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    if scheduler == 'cycliclr':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=0.00005, step_size_up=5, max_lr=0.0001, gamma=0.5, mode='exp_range')
    
    return scheduler