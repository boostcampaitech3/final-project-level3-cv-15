from easydict import EasyDict as eDict

def getArg():
    arg = eDict()
    
    #settings
    arg.epoch = 20
    arg.seed = 42
#     arg.save_capacity = 5

#     arg.image_root = "../input/data"
#     arg.train_json = "train_all.json"
#     arg.val_json = "valid_0.json"
#     arg.test_json = "test.json"
    
    #dataloader
    arg.batch = 16
    arg.train_worker = 4
    arg.valid_worker = 4
    # arg.test_worker = 4
    
    #model
    args.modeltype = 'timm'
    
    #optimizer 
    arg.lr = 1e-3
    args.optimizer
    
    #scheduler
    args.step = 20
    
    #
    arg.output_path = "../output"
    arg.custom_name = "resnest269e_panet_all"

    # arg.wandb = False
    # arg.wandb_project = "segmentation"
    # arg.wandb_entity = "cv4"


    return arg