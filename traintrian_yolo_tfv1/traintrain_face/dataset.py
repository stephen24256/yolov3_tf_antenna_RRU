from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import os

tf = transforms.Compose([
    transforms.Resize(112),
    transforms.ToTensor(),
])


class MyDataset(Dataset):

    def __init__(self, main_dir):

        self.dataset = []
        face = os.listdir(main_dir)
        face.sort(key=lambda x:int(x.split(".")[0]))
        for face_dir in face:
            for face_filename in os.listdir(os.path.join(main_dir, face_dir)):
                self.dataset.append([os.path.join(main_dir, face_dir, face_filename), int(face_dir)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        image_data = tf(Image.open(data[0]))
        label_data = data[1]
        return image_data, label_data


if __name__ == '__main__':
    mydataset = MyDataset("data")
    dataset = DataLoader(mydataset,2,shuffle=True)
    for data in dataset:
        print(data[0].shape)
        print(data[1].shape,data[1])
        exit()
