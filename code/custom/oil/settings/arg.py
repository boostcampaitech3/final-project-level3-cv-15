from easydict import EasyDict as eDict

def getArg():
    arg = eDict()
    
    # settings
    arg.epoch = 30
    arg.seed = 42
    
    # dataloader
    arg.batch = 16
    arg.train_worker = 4
    arg.valid_worker = 4
    
    # model
    arg.regression = False
    arg.ordinalclassification = True
    arg.modeltype = 'timm'
    arg.modelname = 'efficientnet_b4'
    # timm : efficientnet_b4, vit_base_patch16_224

    
    # optimizer 
    arg.lr = 1e-3
    arg.optimizer = 'adam'
    
    # scheduler
    arg.scheduler ='cos'
    arg.step = 20
    
    #transform
    arg.resize = 380

    # loss
    # pick 'mse' if regression mode
    arg.loss = 'cross_entropy'

    arg.output_path = "../output"
    arg.custom_name = "oil_L2_classification"
    arg.log_steps=20

    # accuracy, loss, f1_score, recall_score, precision_score
    arg.metric = "accuracy" 

    arg.wandb = True
    arg.wandb_project = "XAI project"
    arg.wandb_entity = "boostcampaitech3"

    return arg