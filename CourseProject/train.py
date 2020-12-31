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

def train(train_path, label_path, dev_path, bs, lr, num_epoch,
          load_model, model_path, device, layer):
    transforms = Compose([ToTensor()])
    train_data = CaptchaData(train_path, label_path, transforms=transforms)
    train_data_loader = DataLoader(train_data, batch_size=bs,
                                   num_workers=0, shuffle=True,
                                   drop_last=True)
    dev_data = CaptchaData(dev_path, label_path, transforms=transforms)
    dev_data_loader = DataLoader(dev_data, batch_size=bs,
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
    
    if load_model:
        cnn.load_state_dict(torch.load(model_path))
    optimizer = torch.optim.Adam(cnn.parameters(), lr=lr)
    criterion = nn.MultiLabelSoftMarginLoss()
    
    for epoch in range(num_epoch):
        start = time.time()
        
        # training
        cnn.train()
        tot_loss, max_acc = 0., 0.
        for img, target in train_data_loader:
            optimizer.zero_grad()
            img, target = img.to(device), target.to(device)
            output = cnn(img) # [bs, 64 * 4]
            loss = criterion(output, target)
            loss.backward()
            tot_loss += loss.item()
            optimizer.step()
        
        # evaluation
        cnn.eval()
        dev_acc = []
        for img, target in dev_data_loader:
            img, target = img.to(device), target.to(device)
            output = cnn(img) # [bs, 64 * 4]
            dev_acc.append(comp_acc(output, target))
        dev_acc = sum(dev_acc) / len(dev_acc)
        if dev_acc > max_acc:
            max_acc = dev_acc
            torch.save(cnn.state_dict(), model_path)
        
        print(f"Epoch {epoch+1}/{num_epoch}, Total Loss: {tot_loss:.4f}, Dev Acc: {dev_acc:.2%}, Time: {time.time()-start:.2f}s")

def main():
    parser = argparse.ArgumentParser(description='CNN Captcha OCR')
    parser.add_argument('--train_path', type=str, default=None, help='the folder of training captcha images')
    parser.add_argument('--label_path', type=str, default=None, help='the path of captcha images label file')
    parser.add_argument('--dev_path', type=str, default=None, help='the folder of development captcha images')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=10)
    parser.add_argument('--model_path', type=str, help='model save/load path')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--layer', type=int, default=1)
    config = parser.parse_args()
    
    if config.cuda:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    
    print("Start Training...")
    train(config.train_path, config.label_path, config.dev_path, config.bs, config.lr,
          config.num_epoch, config.load_model, config.model_path, device, config.layer)

if __name__ == "__main__":
    main()