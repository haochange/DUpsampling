import torch

def create_model(opt, dataset=None):
    if opt.model == "DUNet":
        from models.dunet import DUNet_Solver
        model = DUNet_Solver(opt)

    return model
