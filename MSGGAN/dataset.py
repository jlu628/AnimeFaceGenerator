import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from torchvision import transforms
import random

class LoadDataset(Dataset):
    def __init__(self, dir, mean=0.5, std=0.5, length=None):
        self.dir = dir
        self.list_files = []
        self.transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        for root, dir, files in os.walk(dir):
            for file in files:
                if '.jpg' in file:
                    self.list_files.append(os.path.join(root, file))
        if length:
            random.shuffle(self.list_files)
            self.list_files = self.list_files[:length]


    def __len__(self):
        return len(self.list_files)


    def __getitem__(self, index):
        file = self.list_files[index]
        img = Image.open(file).convert('RGB')
        img = self.transforms(img)

        return img


if __name__ == "__main__":
    ds = LoadDataset('data/train', 0.5, 0.5)
    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=2)
    num_batches = len(ds)/32

    val_ds = LoadDataset('data/val', 0.5, 0.5)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=2)

    dl = {
        "ds": ds,
        "loader": loader,
        "val_ds": val_ds,
        "val_loader": val_loader
    }
    import pickle
    with open('DataLoader.p', 'wb') as fh:
        pickle.dump(dl, fh)
    # for batch in loader:
    #     print(batch.shape)
    #     save_image(batch, 'sample.jpg')
    #     break