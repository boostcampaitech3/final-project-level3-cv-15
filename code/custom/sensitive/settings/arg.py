from easydict import EasyDict as eDict

def getArg():
    arg = eDict()
    
    # settings
    arg.epoch = 20
    arg.seed = 42
    
    # dataloader
    arg.batch = 8
    arg.train_worker = 4
    arg.valid_worker = 4
    
    # model
    arg.modeltype = 'giff'
    arg.modelname = 'efficientnet_b4'
    # timm : efficientnet_b4, vit_base_patch16_224

    
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
    arg.custom_name = "sensitive"
    arg.log_steps=20

    # accuracy, loss, f1_score, recall_score, precision_score
    arg.metric = "accuracy" 

    arg.wandb = False
    arg.wandb_project = "XAI project"
    arg.wandb_entity = "boostcampaitech3"

    return arg