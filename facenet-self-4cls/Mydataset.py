import torch
import os
import numpy as np
import torchvision.transforms as transform
from torch.utils.data import Dataset,DataLoader
from PIL import Image
import random

class ImageClass():
    "Stores the paths to images for a given class"

    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths

    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'

    def __len__(self):
        return len(self.image_paths)


class Mydataset(Dataset):
    def __init__(self,path,data_aug=False):
        self.path = path
        self.data_aug = data_aug
        self.classes = os.listdir(self.path)
        self.classes.sort()
        self.transform = transform.Compose([
            transform.ToTensor(),
        ])
        self.dataset = self.get_dataset(self.path)
        self.image_paths_flat, self.labels_flat = self.get_image_paths_and_labels(self.dataset)

    def __len__(self):
        return len(self.image_paths_flat)

    def __getitem__(self, index):

        image_path = self.image_paths_flat[index]
        label = self.labels_flat[index]
        label = torch.tensor(label)
        image = Image.open(image_path)
        if self.data_aug:
            image = self.random_rotate_crop(image)
        image = image.resize((160,160),Image.ANTIALIAS)
        image = self.transform(image)
        return image,label

    def get_image_paths(self,facedir):
        image_paths = []
        if os.path.isdir(facedir):
            images = os.listdir(facedir)
            image_paths = [os.path.join(facedir, img) for img in images]
        return image_paths


    def get_dataset(self,path):
        dataset = []
        nrof_classes = len(self.classes)  # 总类别数量（有多少个人）
        for i in range(nrof_classes):
            class_name = self.classes[i]
            facedir = os.path.join(path, class_name)
            image_paths = self.get_image_paths(facedir)  # os.listdir + filter
            dataset.append(ImageClass(class_name, image_paths))
            # dataset.append(class_name + ', ' + str(len(image_paths)) + ' images')

        return dataset

    def get_image_paths_and_labels(self,dataset):
        image_paths_flat = []
        labels_flat = []
        for i in range(len(dataset)):
            image_paths_flat += dataset[i].image_paths
            labels_flat += [i] * len(dataset[i].image_paths)
        return image_paths_flat, labels_flat



    def random_rotate_crop(self,image):
        if random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image = image.rotate(np.random.randint(-10, 10))
        w,h = image.size[0],image.size[1]
        a = random.randint(1,10)
        image = image.crop((a, a, w-a, h-a))
        return image


if __name__ == '__main__':
    path = r"D:\facenet-self\data\casia_maxpy_mtcnnpy2_182"
    mydataset = Mydataset(path,data_aug=True)
    print(len(mydataset),mydataset.dataset)
    dataload = DataLoader(mydataset,batch_size=12,shuffle=True)
    img = next(iter(dataload))[0]
    print(img.shape)
    for image,label in dataload:
        # print("71image,label",image.shape,label)
        image = image[0]
        image = transform.ToPILImage()(image)
        image.show()