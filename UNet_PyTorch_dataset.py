from torch.utils.data.dataset import Dataset
import torchvision.transforms.v2 as transforms
import os
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root_path, transform, limit=None):
        super().__init__()
        
        self.root_path = root_path
        self.limit = limit
        self.image_list = sorted([root_path + "/train_imgs/" + i for i in os.listdir(root_path + "/train_imgs/")])[:self.limit]
        self.mask_list = sorted([root_path + "/train_masks/" + i for i in os.listdir(root_path + "/train_masks/")])[:self.limit]

        if transform == "transform_augmentation":
            self.transform = transforms.Compose([
                transforms.RandomVerticalFlip(p=0.25),
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
                ])

        if transform == "transform":
            self.transform = transforms.Compose([
                transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
                ])
        
        if self.limit is None:
            self.limit = len(self.image_list)


    def __getitem__(self, index):
        img = Image.open(self.image_list[index]).convert("RGB")
        mask = Image.open(self.mask_list[index]).convert("L")

        return self.transform(img, mask)
        

    def __len__(self):
        return min(len(self.image_list), self.limit)



