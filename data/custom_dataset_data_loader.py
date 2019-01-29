import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None

    if opt.dataset_mode == 'voc':
        from data.dataset import VOCDataSet, VOCDataValSet
        dataset = VOCDataSet(opt)
        if opt.val_list_path !='':
            dataset_val = VOCDataValSet(opt)
        else:
            dataset_val = None

    print("dataset [%s, %s] was created" % (dataset.name(), dataset_val.name()))
    return dataset, dataset_val

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset, self.dataset_val = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        if self.dataset_val != None:
            self.dataloader_val = torch.utils.data.DataLoader(
                self.dataset_val,
                batch_size=1,
                shuffle=False,
                num_workers=int(opt.nThreads))
        else:
            self.dataloader_val = None


    def load_data(self):
        return self.dataloader, self.dataloader_val

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
