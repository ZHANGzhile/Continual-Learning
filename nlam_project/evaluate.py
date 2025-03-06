import torch
from models.nlam import NLAM
from utils.config import Config
from utils.data_loader import get_data_loader

def evaluate(model, test_loader):
    model.eval()
    total_loss = 0
    criterion = torch.nn.MSELoss()
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            
    return total_loss / len(test_loader)

def main():
    config = Config()
    model = NLAM(channels=config.CHANNELS)
    test_loader = get_data_loader(config)
    
    avg_loss = evaluate(model, test_loader)
    print(f'Average test loss: {avg_loss:.4f}')

if __name__ == "__main__":
    main() 