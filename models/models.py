import torch

def create_model(opt, dataset=None):
    if opt.model == "DeeplabV3":
        pass
    elif opt.model == "PSP":
        from models.psp import PSP_Solver
        model = PSP_Solver(opt, dataset)
    print("model [%s] was created" % (model.name()))

    return model
