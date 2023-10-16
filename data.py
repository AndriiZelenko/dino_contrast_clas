from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforsm
import torch.nn as nn
import torch
from torchvision import transforms

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import display


import json
import os
import yaml
import os
import numpy as np


import yaml
import os
import numpy as np
import json

class weldingData:
    def __init__(self, dataset_config_path, training_config_path):
        training_config_file = open(training_config_path, 'r')
        self.training_config_data = yaml.safe_load(training_config_file)

        dataset_config_file = open(dataset_config_path, 'r')
        self.dataset_config_data = yaml.safe_load(dataset_config_file)
        
        self.images_path = self.training_config_data['images_folder']
        self.test_train_split_path = self.training_config_data['train_test_split_json']
        self.train_ratio = self.training_config_data['train_ratio']
        self.test_ratio = self.training_config_data['test_ratio']
        
        self.idx2class = self.get_idx2class(self.dataset_config_data)
        self.class2path = self.gen_class2path(self.images_path)
        self.path2class, self.idx2class, self.class2idx, self.folder2idx = self.gen_classifications(self.class2path)
    
        self.res_dict = None
            
    def gen_class2path(self, images_path):
        class2path = {}
        def gen_all_classes_s(path, data, depth =0):
            files = os.listdir(path)
            if files[0][-3:] == 'jpg':

                p = '='.join(path.split('/')[-depth:])
                data[p] = path
                depth = 0
            else:
                depth += 1
                for file in files:
                    if file == '.DS_Store' or file.split('.')[-1] == 'json':
                        continue

                    gen_all_classes_s(os.path.join(path, file), data, depth)
            return data
        all_classes = gen_all_classes_s(images_path, class2path)

        return class2path
    
    
    
    def get_idx2class(self, dataset_config_data):
        
        idx2class = {}
        for key, value in dataset_config_data.items():

            if type(value) == bool:
                if value:
                    idx2class[len(idx2class)] = key
            else:
                for key2, value2 in value.items():
                    if value2:
                        if key not in idx2class.values():
                            idx2class[len(idx2class)] = key
                
                            
        return idx2class
    
    
    def gen_classifications(self, class2path):
        

        path2class = {}
        idx2class = {}
        class2idx = {}
        for key, value in self.dataset_config_data.items():
            if type(value) == bool:
                if value:
                    idx2class[len(idx2class)] = key
                    class2idx[key] = len(idx2class) - 1  
                    path2class[class2path[key]] = class2idx[key]
            elif type(value) == dict:
                for key2, value2 in value.items():
                    if value2:
                        if key not in idx2class.values():
                            idx2class[len(idx2class)] = key
                        class2idx[key] = len(idx2class) - 1  
                        try:
                            path2class[class2path[key + '=' + key2]] = class2idx[key]
                        except KeyError: 
                            path2class[class2path[key2]] = class2idx[key]
        folder2idx = {}
        for key, value in path2class.items():
            k = key.split('/')[-1]
            folder2idx[k] = value
                            
        return path2class, idx2class, class2idx, folder2idx
        
    
    def gen_image2class(self, path2class):
        image2class = {}
        for key, value in path2class.items():
            images = os.listdir(key)
            for image in images:
                if image.split('.')[-1] not in ['jpg', 'png'] :
                    continue
                image2class[os.path.join(key, image)] = value
        return image2class
    
    
    def gen_train_test(self):
            
        test_dict = {}
        train_dict = {}


        images = list(self.image2class.keys())
        clas = list(self.image2class.values())
        
        for uq in np.unique(clas):
            idxs = np.where(clas == uq)[0]
            n_train = int(len(idxs) * self.train_ratio)
            np.random.shuffle(idxs)
            train_idxs = idxs[:n_train]
            test_idxs = idxs[n_train:]

            for i in train_idxs:
                train_dict[images[i]] = clas[i]
            for j in test_idxs:
                test_dict[images[j]] = clas[j]

        res_dict = {'train' : train_dict, 'test' : test_dict}
            
        return res_dict, train_dict, test_dict
    
    
    def save(self, path):
        
        self.path2class, self.idx2class, self.class2idx, self.folder2idx = self.gen_classifications(self.class2path)
        self.image2class = self.gen_image2class(self.path2class)
        if self.res_dict: 
            with open(path, "w") as json_file:
                json.dump(self.res_dict, json_file, indent=4)
        else: 
            self.res_dict, self.train_dict, self.test_dict = self.gen_train_test()


            self.res_dict = {'train': self.train_dict, 'test' : self.test_dict, 'idx2class' : self.idx2class}
            with open(path, "w") as json_file:
                json.dump(self.res_dict, json_file, indent=4)
        
    
    def load(self, path):
        file = open(path)
        data = json.load(file)
        train_dict = data['train']
        test_dict  = data['test']
        all_idx2classes = data['idx2class']
        all_class2idx = {value : key for key, value in all_idx2classes.items()}
        
        
        train_images = list(train_dict.keys())
        train_labels = list(train_dict.values())

        test_images = list(train_dict.keys())
        test_labels = list(test_dict.values())

        
        self.train_dict = {}
        self.test_dict = {}
        
        class2class = {}
        
        for key, value in self.folder2idx.items():
            for key2, value2 in all_class2idx.items():
                if key == key2:
                    class2class[int(value2)] = value
                    
        for key, value in train_dict.items():
            if value in list(class2class.keys()):
                self.train_dict[key] = class2class[value]
        for key, value in test_dict.items():
            if value in list(class2class.keys()):
                self.test_dict[key] = class2class[value]
        self.res_dict = {'train': self.train_dict, 'test' : self.test_dict, 'idx2class' : self.idx2class}
        self.image2class = {}
        self.image2class.update(self.train_dict)
        self.image2class.update(self.test_dict)
            

    
    def describe(self):
        text = f'Num images: {len(self.image2class)} \n' \
               f'Num train : {len(self.train_dict)} \n' \
               f'Num test  : {len(self.test_dict)} \n' \
               f'JSON path : {self.test_train_split_path} \n \n'
        text += '\n'
        text += 'VOCABULARY \n'
        for key , value in self.class2idx.items():
            text += f'{key} : {value} \n'
        text += '\n \n'
        text += 'CLASS DISTRIBUTION \n' 
        unique_classes = np.unique(list(self.image2class.values()))
        labels = list(self.image2class.values())
        for uq in unique_classes:
            num = len(np.where(np.array(labels) == uq)[0])
            text += f'{self.idx2class[uq]} : {num} \n'
        
        print(text)
        return text


class clasDataset(Dataset):
    def __init__(self, data_dict,idx2class, class2idx, transforms = None):
        super(clasDataset, self).__init__()
        
        self.data_dict = data_dict
        self.images = list(data_dict.keys())
        self.labels = list(data_dict.values())
        self.transforms = transforms
        self.class2idx = class2idx
        self.idx2class = idx2class
        
    
    
    def __len__(self):
        return len(self.data_dict)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx])
        label = self.labels[idx]
        
        if self.transforms:
            image = self.transforms(image)
            
        return image, label
    
    
def showFromDataset(dataset, idx2class, n = 5):
    i, _ = dataset[0]
    if type(i) == torch.Tensor:
        f = True
        t = transforms.Compose([transforms.ToPILImage()])
    else: 
        f = False
    rand = np.random.randint(0,len(dataset), n)
    if n >= 10:
        N = n // 2
        x = n // N
    else:
        x = 1
        N = n
    fig, axes = plt.subplots(x,N, figsize = (14,20))
    for idx, i in enumerate(rand):
        image, label = dataset[i]
        if f:
            image = t(image)
        axes.flat[idx].imshow(np.array(image))
        axes.flat[idx].set_title(idx2class[label])

    return fig

def showTensors(data):
    n = data.shape[0]
    t = transforms.Compose([transforms.ToPILImage()])
    fig, axes = plt.subplots(1,n, figsize = (10,14))
    for idx in range(n):
        image = t(data[idx])
        axes.flat[idx].imshow(image)
    return fig




def get_transforms(training_config_path):
    training_config_file = open(training_config_path, 'r')
    training_config_data = yaml.safe_load(training_config_file)
    mean = training_config_data['mean']
    std = training_config_data['std']
    size = training_config_data['size']

    train_transforms = transforms.Compose(
        [
            transforms.Resize((size, size), transforms.InterpolationMode.BILINEAR),
            transforms.ColorJitter(contrast=0.5),
            transforms.GaussianBlur(kernel_size = 5),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.RandomAffine(degrees = 40, shear = (0.5, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )

    test_transforms = transforms.Compose(
        [
            transforms.Resize((size, size), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ])

        
    return train_transforms , test_transforms


if __name__ == '__main__':
    data_path = '/home/andrii/adient/welding_data'
    dataset_config_path = '/home/andrii/adient/dataset_config.yaml'
    training_config_path = '/home/andrii/adient/training_conf.yaml'

    train_dataloader, test_dataloader, val_dataloader = get_dataloaders(data_path, dataset_config_path, training_config_path)

    train_dataset, train_dataset_clean, test_dataset, test_dataset_clean, val_dataset, val_dataset_clean, idx2class, class2idx = get_datasets(data_path, dataset_config_path,
                                                                                                                                              training_config_path)
    all_classes = get_all_classes(data_path)
    dir_pathes, idx2class, class2idx = get_data(dataset_config_path, all_classes)
    images_class = get_images_paths(dir_pathes)
    train_data, test_data, val_data = train_test_split(images_class, 0.7, 0.3, 0)
    r = np.random.randint(0, len(images_class), 15)

    for iidx in r:
        ip, ic = list(images_class.items())[iidx]
        print(f'{ip} | {ic}    ====>     {idx2class[ic]}')


    print('='*7 + 'TRAIN' + '='*7)
    print(f'U: {np.unique(list(train_data.values()))}  = {len(np.unique(list(train_data.values())))}')
    print('='*7 + 'TEST' + '='*7)
    print(f'U: {np.unique(list(test_data.values()))}  = {len(np.unique(list(test_data.values())))}')
    print('='*7 + 'VAL' + '='*7)
    print(f'U: {np.unique(list(val_data.values()))}  = {len(np.unique(list(val_data.values())))}')

    combined_array = np.concatenate((np.array(list(train_data.keys())), 
                                    np.array(list(test_data.keys())),
                                    np.array(list(val_data.keys()))))
    are_all_unique = len(combined_array) == len(np.unique(combined_array))
    if are_all_unique:
        print("All values in the arrays are different.")
    else:
        print("There are duplicate values in the arrays.")

    res_path = 'debug_output/data_tests/'
    if os.path.exists(res_path): 
        pass
    else: 
        rp = res_path.split('/')
        os.mkdir(rp[0])
        os.mkdir(f'{rp[0]}/{rp[1]}')

    for i, t in train_dataloader:
        break

    training_transformed = showTensors(i)
    training_transformed.savefig(os.path.join(res_path, 'training_transformed.png'))

    for i, t in test_dataloader:
        break

    testing_transformed = showTensors(i)
    testing_transformed.savefig(os.path.join(res_path, 'testing_transformed.png'))

    for i, t in train_dataset_clean:
        break

    train_clean = showFromDataset(train_dataset_clean, idx2class, 10)
    train_clean.savefig(os.path.join(res_path, 'training_clean.png'))

    for i, t in test_dataset_clean:
        break

    testing_clean = showFromDataset(test_dataset_clean, idx2class, 10)
    train_clean.savefig(os.path.join(res_path, 'testing_clean.png'))






