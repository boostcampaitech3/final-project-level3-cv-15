from easydict import EasyDict as eDict

def getArg():
    arg = eDict()
    
    # settings
    arg.epoch = 35
    arg.seed = 42
    
    # dataloader
    arg.batch = 4
    arg.train_worker = 4
    arg.valid_worker = 4
    
    # model
    arg.modeltype = 'timm'
    arg.modelname = 'efficientnet_b4'
    # timm : efficientnet_b4, vit_base_patch16_224

    
    # optimizer 
    arg.lr = 0.0057
    arg.optimizer = 'adamw'
    
    # scheduler
    arg.scheduler ='lambdalr'
    arg.step = 20
    
    # loss
    arg.loss = 'focal' 

    #
    arg.output_path = "../output"
    arg.custom_name = "oil"
    arg.log_steps=20

    # accuracy, loss, f1_score, recall_score, precision_score
    arg.metric = "f1_score" 

    arg.wandb = True
    arg.wandb_project = "XAI project"
    arg.wandb_entity = "boostcampaitech3"

    return arg