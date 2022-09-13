import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import random
import os

def get_dataloader(root_dir, batch_size, train):
    if train:
        shuffle = True
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness= 0.5,
                contrast = 0.5,
                saturation = 0.5
            ),
            transforms.ToTensor()
        ])
    else:
        shuffle = False
        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    
    transform = None # TODO: Implement
    dataset = MaskDataset(root_dir, train, transform)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = shuffle, num_workers = 2)
    return dataloader

class MaskDataset(Dataset):
    def __init__(self, root_dir, train, transform):
        self.transform = transform
        self.samples = [] # image path & ground truth
        
        if train:
            target_dir = 'Train'
        else:
            target_dir = 'Test'

        pos_sample_dir = os.path.join(os.path.join(root_dir, target_dir), 'WithMask')
        neg_sample_dir = os.path.join(os.path.join(root_dir, target_dir), 'WithoutMask')

        for image_name in os.listdir(pos_sample_dir):
            if image_name.split('.')[-1] != 'png':
                continue
            image_path = os.path.join(pos_sample_dir,image_name)
            self.samples.append((image_path, 0))

        for image_name in os.listdir(neg_sample_dir):
            if image_name.split('.')[-1] != 'png':
                continue
            image_path = os.path.join(neg_sample_dir,image_name)
            self.samples.append((image_path, 1))

    def __len__(self): # 우리가 아는 일반적인 len function과 똑같은 역할을 함, 자동으로 연동됨
        return len(self.samples)

    def __getitem__(self, index):
        image_path, gt = self.samples[index]
        image = Image.open(image_path)

        image = self.transform(image)
        if random.random() < 0.5:
            # gaussian noise addition
            image += torch.randn(image.size())*0.05
        ret = {
            'x': image, 
            'y': torch.LongTensor([gt]).squeeze()
        }
        return ret


if __name__ == '__main__':

    test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor()
        ])
    ROOT_DIR = '/Users/admin/Downloads/Mask_Dataset_folder_copy'
    dataset = MaskDataset(ROOT_DIR, False, test_transform)

    print(len(dataset))

    import cv2
    import numpy as np
    for i, sample in enumerate(dataset):
        sample = sample['x']
        np_img = (sample * 255).numpy().astype(np.uint8).transpose(1,2,0)
        file_path = '{}.png'.format(i+1)
        cv2.imwrite(file_path, np_img)

        if i == 5:
            break