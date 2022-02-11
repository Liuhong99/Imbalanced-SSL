import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class IMBALANCECIFARIND(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='step', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False, weight_path = None):
        super(IMBALANCECIFARIND, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.weight_path = weight_path
        self.gen_imbalanced_data(imb_type)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, imb_type):
        if imb_type == 'exp':
            self.data = np.load('data/cifar_exp_data.npy').astype('uint8')
            self.targets = np.load('data/cifar_exp_label.npy').astype('int64')
        if imb_type == 'step':
            self.data = np.load('data/cifar_step_data.npy').astype('uint8')
            self.targets = np.load('data/cifar_step_label.npy').astype('int64')

        self.weights = np.load(self.weight_path).astype('float32') 

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list
        
    def __getitem__(self, index: int):
        
        img, target = self.data[index], self.targets[index]
        weight = self.weights[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, weight

class IMBALANCECIFAR(torchvision.datasets.CIFAR10):
    cls_num = 10

    def __init__(self, root, imb_type='step', imb_factor=0.01, rand_number=0, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super(IMBALANCECIFAR, self).__init__(root, train, transform, target_transform, download)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
        self.gen_imbalanced_data(imb_type)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, imb_type):
        if imb_type == 'exp':
            self.data = np.load('data/cifar_exp_data.npy').astype('uint8')
            self.targets = np.load('data/cifar_exp_label.npy').astype('int64')
        if imb_type == 'step':
            self.data = np.load('data/cifar_step_data.npy').astype('uint8')
            self.targets = np.load('data/cifar_step_label.npy').astype('int64')

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list