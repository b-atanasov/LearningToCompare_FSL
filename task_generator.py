import os
import random
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from scipy.special import binom


class FewShotDataset(Dataset):
    def __init__(self, image_roots, labels, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.image_roots = image_roots
        self.labels = labels

    def __len__(self):
        return len(self.image_roots)

    def __getitem__(self, idx):
        image_root = self.image_roots[idx]
        image = Image.open(image_root)
        image = image.convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels[idx]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


class TaskSampler:
    def __init__(self, args, train=True):
        self.metadataset_folder = args.train_folder if train else args.test_folder
        self.class_num = args.class_num
        self.train_num = args.sample_num_per_class
        self.test_num = args.batch_num_per_class if train else args.test_batch_num_per_class
        self.img_size = args.img_size
        
        random.seed(1)
        self.img_class_paths = self.__get_img_class_paths()
        self.__display_num_combinations()
        
    def __get_img_class_paths(self):
        img_class_paths = []
        for label in os.listdir(self.metadataset_folder):
            paths = os.path.join(self.metadataset_folder, label)
            if os.path.isdir(paths):
                img_class_paths.append(paths)

        random.shuffle(img_class_paths)
        return img_class_paths
    
    def __display_num_combinations(self):
        num_combinations = int(binom(len(self.img_class_paths), self.class_num))
        print(f'Number of {self.class_num}-combinations from a set of {len(self.img_class_paths)}',
              f'classes in {os.path.basename(self.metadataset_folder)}: {num_combinations}')
    
    def sample_task_data(self):
        class_folders = self.__sample_image_classes()
        train_roots, test_roots = self.__sample_images(class_folders)
        labels_index = self.__get_labels_index(class_folders)
        train_labels = self.__get_labels(train_roots, labels_index)
        test_labels = self.__get_labels(test_roots, labels_index)
        
        sample_dataloader = self.__get_data_loader(train_roots, train_labels, self.train_num, split="train", shuffle=False)
        batch_dataloader = self.__get_data_loader(test_roots, test_labels, self.test_num, split="test", shuffle=True)
        return sample_dataloader, batch_dataloader    
    
    def __sample_image_classes(self):
        class_folders = random.sample(self.img_class_paths, self.class_num)
        return class_folders
    
    def __sample_images(self, class_folders):
        samples = dict()
        train_roots = []
        test_roots = []
        
        for class_folder in class_folders:
            class_img_paths = [os.path.join(class_folder, img) for img in os.listdir(class_folder)]
            samples[class_folder] = random.sample(class_img_paths, self.train_num + self.test_num)

            train_roots += samples[class_folder][:self.train_num]
            test_roots += samples[class_folder][self.train_num:self.train_num + self.test_num]
        
        return train_roots, test_roots
    
    def __get_labels_index(self, class_folders):
        return dict(zip(class_folders, range(self.class_num)))
    
    def __get_labels(self, roots, labels_index):
        return [labels_index[os.path.dirname(root)] for root in roots]
    
    def __get_data_loader(self, image_roots, labels, num_per_class, split, shuffle):
        ds_transforms = transforms.Compose([transforms.RandomResizedCrop(self.img_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                 std=[0.229, 0.224, 0.225])])
        dataset = FewShotDataset(image_roots, labels, transform=ds_transforms)
        loader = DataLoader(dataset, batch_size=num_per_class*self.class_num, shuffle=shuffle,
                            num_workers=2, pin_memory=True)
        return loader
        