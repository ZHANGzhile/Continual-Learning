import torch
from models.nlam import NLAM
from utils.config import Config

def predict(model, input_data):
    model.eval()
    with torch.no_grad():
        output = model(input_data)
    return output

def main():
    config = Config()
    model = NLAM(channels=config.CHANNELS)
    
    # 加载模型权重
    # model.load_state_dict(torch.load('path_to_saved_model.pth'))
    
    # 示例输入数据
    sample_input = torch.randn(1, config.CHANNELS, 32, 32)
    prediction = predict(model, sample_input)
    print(f'Prediction shape: {prediction.shape}')

if __name__ == "__main__":
    main() 