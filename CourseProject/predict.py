import os
import time
import torch
import argparse
import torch.nn as nn
from model import CNN1, CNN2, CNN3, CNN4, CNN5, FNN
from data import CaptchaData
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor

def comp_acc(output, target):
    with torch.no_grad():
        output, target = output.view(-1, 62), target.view(-1, 62) # [bs * 4, 62]
        output = torch.softmax(output, dim=-1)
        output = output.argmax(dim=-1) # [bs * 4]
        target = target.argmax(dim=-1) # [bs * 4]
        output, target = output.view(-1, 4), target.view(-1, 4) # [bs, 4]
        correct, total = 0, 0
        for pred, gold in zip(output, target):
            if torch.equal(pred, gold):
                correct += 1
            total += 1
    return correct / total

def predict(test_path, label_path, bs, model_path, device, layer):
    with torch.no_grad():
        transforms = Compose([ToTensor()])
        test_data = CaptchaData(test_path, label_path, transforms=transforms)
        test_data_loader = DataLoader(test_data, batch_size=bs,
                                      num_workers=0, shuffle=False,
                                      drop_last=False)
        if layer == 1:
            cnn = CNN1().to(device)
        elif layer == 2:
            cnn = CNN2().to(device)
        elif layer == 3:
            cnn = CNN3().to(device)
        elif layer == 4:
            cnn = CNN4().to(device)
        elif layer == 5:
            cnn = CNN5().to(device)
        else:
            cnn = FNN().to(device)
        
        cnn.load_state_dict(torch.load(model_path))
        cnn.eval()
        test_acc = []
        res = []
        for img, target in test_data_loader:
            img, target = img.to(device), target.to(device)
            output = cnn(img) # [bs, 64 * 4]
            test_acc.append(comp_acc(output, target))
    test_acc = sum(test_acc) / len(test_acc)
    print(f"Test Accuracy: {test_acc:.2%}")
    
def main():
    parser = argparse.ArgumentParser(description='CNN Captcha OCR')
    parser.add_argument('--test_path', type=str, default=None, help='the folder of test captcha images')
    parser.add_argument('--label_path', type=str, default=None, help='the path of captcha images label file')
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--model_path', type=str, help='model load path')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--layer', type=int, default=1)
    config = parser.parse_args()
    
    if config.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    predict(config.test_path, config.label_path, config.bs, config.model_path, device, config.layer)

if __name__ == "__main__":
    main()