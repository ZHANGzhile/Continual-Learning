import torch
import torch.nn as nn
from models.nlam import NLAM
from utils.config import Config
from utils.data_loader import get_data_loader

def train(model, train_loader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

def main():
    config = Config()
    model = NLAM(channels=config.CHANNELS)
    train_loader = get_data_loader(config)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    train(model, train_loader, criterion, optimizer, config.NUM_EPOCHS)

if __name__ == "__main__":
    main() 