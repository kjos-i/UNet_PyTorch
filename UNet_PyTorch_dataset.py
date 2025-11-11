from torch.utils.data.dataset import Dataset
import torchvision.transforms.v2 as transforms
import torch
import os
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root_path, augmentation, change_size, scale_mask, limit):
        super().__init__()
        
        self.root_path = root_path
        self.limit = limit
        self.image_list = sorted([root_path + "/train_imgs/" + i for i in os.listdir(root_path + "/train_imgs/")])[:self.limit]
        self.mask_list = sorted([root_path + "/train_masks/" + i for i in os.listdir(root_path + "/train_masks/")])[:self.limit]
        self.augmentation = augmentation
        self.change_size = change_size

        self.augment = transforms.Compose([transforms.RandomVerticalFlip(p=0.25),
                                           transforms.RandomHorizontalFlip(p=0.25)])

        self.resize = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST)
        self.center_crop = transforms.CenterCrop(512)

        self.transform_img = transforms.Compose([transforms.ToImage(),
                                                transforms.ToDtype(torch.float32, scale=True)])
        self.transform_mask = transforms.Compose([transforms.ToImage(),
                                                transforms.ToDtype(torch.float32, scale=scale_mask)])         


    def __getitem__(self, index):
        img = Image.open(self.image_list[index]).convert("RGB")
        mask = Image.open(self.mask_list[index]).convert("L")
        
        if self.augmentation == 'augmentation':
            img, mask = self.augment(img, mask)

        if self.change_size == 'interpolation_nearest':
            img, mask = self.resize(img, mask) 

        if self.change_size == 'center_crop':
            img, mask = self.center_crop(img, mask)  

        return self.transform_img(img), self.transform_mask(mask)
        

    def __len__(self):
        if self.limit is None:
            self.limit = len(self.image_list)

        return min(len(self.image_list), self.limit)


