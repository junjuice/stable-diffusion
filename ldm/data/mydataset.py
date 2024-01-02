from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import requests
import tqdm
import shutil
import glob
from typing import Sequence

def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm.tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

class MydatasetBase(Dataset):
    def __init__(self, size: int = 256, path="./data/mydataset", transform=transforms.Compose([])):
        url = "https://huggingface.co/datasets/junjuice0/28k/resolve/main/data.zip"
        path = path.removesuffix("/") + "/"
        if not os.path.isdir(path+"data"):
            os.makedirs(path, exist_ok=True)
            print("Downloading data file...")
            download(url, path+"data.zip")
            print("Extracting data files...")
            shutil.unpack_archive(path+"data.zip", extract_dir=path)
            os.remove(path+"data.zip")

        self.size = size
        self.data = glob.glob(path+"/data/*.png")
        self.transform = transforms.Compose([
            transform,
            transforms.RandomCrop((size, size))
        ])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        path = self.data[index]
        example = {
            "image": None,
            "caption": None
        }
        image = Image.open(path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        image = self.transform(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        with open(path.replace(".png", ".txt")) as f:
            example["caption"] = f.readlines()[1]
        return example

class MydatasetTrain(MydatasetBase):
    def __init__(self, size: int = 256, path="./data/mydataset", transform=transforms.Compose([])):
        super().__init__(size, path, transform)
        length = len(self)
        indices = range(length)[:int(length*0.95)]
        self = Subset(self, indices)

class MydatasetValidation(MydatasetBase):
    def __init__(self, size: int = 256, path="./data/mydataset", transform=transforms.Compose([])):
        super().__init__(size, path, transform)
        length = len(self)
        indices = range(length)[int(length*0.95):]
        self = Subset(self, indices)