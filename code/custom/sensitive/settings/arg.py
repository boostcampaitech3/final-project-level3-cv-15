from easydict import EasyDict as eDict

def getArg():
    arg = eDict()
    
    # settings
    arg.epoch = 50
    arg.seed = 42
    
    # dataloader
    arg.batch = 16
    arg.train_worker = 4
    arg.valid_worker = 4
    
    # model
    arg.modeltype = 'timm'
    arg.modelname = 'efficientnet_b4'
    # timm : efficientnet_b4, vit_base_patch16_224

    
    # optimizer 
    arg.lr = 0.001
    arg.optimizer = 'adamw'
    
    # scheduler
    arg.scheduler ='lambdalr'
    arg.step = 20
    
    # loss
    arg.loss = 'focal' 

    #
    arg.output_path = "../output"
    arg.custom_name = "Son_sensitive_sampler"
    arg.log_steps=20

    # accuracy, loss, f1_score, recall_score, precision_score
    arg.metric = "f1_score" 

    arg.wandb = True
    arg.wandb_project = "XAI project"
    arg.wandb_entity = "boostcampaitech3"

    #========
    # arg.p = 1.0
    # arg.limit = 4

    return arg