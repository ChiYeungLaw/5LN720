import os
import torch
from PIL import Image
from torch.utils.data import Dataset


voc = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
num_class = len(voc)

def img_loader(img_path):
    img = Image.open(img_path)
    return img.convert('RGB')

def make_dataset(data_path, label_dict, voc, num_class, num_char):
    """
    Input:
    (1) data_path: the folder contains all images
    (2) voc: vocabulary
    (3) num_class: the number of character in vocabulary
    (4) num_char: the number of character in the captcha
    """
    imgs = os.listdir(data_path) # all images name
    samples = []
    for img in imgs:
        if '.jpg' not in img:
            continue
        img_path = os.path.join(data_path, img)
        target = label_dict[img]
        target_vec = []
        for char in target:
            vec = [0] * num_class
            vec[voc.find(char)] = 1
            target_vec += vec
        samples.append((img_path, target_vec))
    return samples

class CaptchaData(Dataset):
    def __init__(self, data_path, label_path, num_class=62, num_char=4,
                 transforms=None, voc=voc):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.num_class = num_class
        self.num_char = num_char
        self.transforms = transforms
        self.voc = voc
        
        self.d = {}
        with open(label_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip().split('\t')
                self.d[line[0]] = line[1]
        
        self.samples = make_dataset(self.data_path, self.d, self.voc, 
                                    self.num_class, self.num_char)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        img_path, target = self.samples[index]
        img = img_loader(img_path)
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.LongTensor(target)