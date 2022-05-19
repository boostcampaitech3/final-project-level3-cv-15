# from torch.optim import Adam, AdamW
# def getOptimizer(model, optimizer, lr):
#     if optimizer == 'adam':
#         optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.0)
#     optimizer.zero_grad()
#     return optimizer

from torch import optim

def getOptimizer(model, optimizer, lr):
    origin = ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'SparseAdam', 'Adamax', 
              'ASGD', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD']
    lowers = ['adadelta', 'adagrad', 'adam', 'adamw', 'sparseadam', 'adamax', 
              'asgd', 'lbfgs', 'nadam', 'radam', 'rmsprop', 'rprop', 'sgd']
    if optimizer.lower() in lowers:
        idx = lowers.index(optimizer.lower())
        exec(f'global optimizer1; optimizer1 = optim.{origin[idx]}(model.parameters(), lr=lr)')
    else : raise Exception(f"{optimizer} is not in torch.optim")
    optimizer1.zero_grad()
    
    return optimizer1


