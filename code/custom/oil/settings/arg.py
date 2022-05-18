from easydict import EasyDict as eDict

def getArg():
    arg = eDict()
    
    # settings
    arg.epoch = 20
    arg.seed = 42
    
    # dataloader
    arg.batch = 16
    arg.train_worker = 4
    arg.valid_worker = 4
    
    # model
    arg.modeltype = 'timm'
    
    # optimizer 
    arg.lr = 1e-3
    arg.optimizer = 'adam'
    
    # scheduler
    arg.scheduler ='steplr'
    arg.step = 20
    
    # loss
    arg.loss = 'cross_entropy'

    #
    arg.output_path = "../output"
    arg.custom_name = "oil4"
    arg.log_steps=20
    arg.f1 = True

    # arg.wandb = False
    # arg.wandb_project = "segmentation"
    # arg.wandb_entity = "cv4"

    return arg