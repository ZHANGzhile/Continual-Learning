import torch
from torch.utils.data import Dataset, DataLoader

class NLAMDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        # 在这里实现数据加载逻辑
        
    def __len__(self):
        # 返回数据集大小
        pass
        
    def __getitem__(self, idx):
        # 返回单个数据样本
        pass

def get_data_loader(config):
    dataset = NLAMDataset(config.DATA_PATH)
    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True
    ) 