import os
from PIL import Image
from sklearn.utils import shuffle
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms

class LoadDataset(Dataset):
    def __init__(self, dir, mean, std):
        self.dir = dir
        self.list_files = []
        self.transforms = transforms.Compose([
            # transforms.Grayscale(),
            # transforms.Resize(256),
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        for root, dir, files in os.walk(dir):
            for file in files:
                if '.jpg' in file:
                    self.list_files.append(os.path.join(root, file))
        # self.list_files = self.list_files[:16000]


    def __len__(self):
        return len(self.list_files)


    def __getitem__(self, index):
        file = self.list_files[index]
        img = Image.open(file).convert('RGB')
        img = self.transforms(img)

        return img


if __name__ == "__main__":
    dir = 'data/'
    ds = LoadDataset(dir, 0.5, 0.5)
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    for batch in loader:
        print(batch.shape)
        save_image(batch, 'sample.jpg')
        break