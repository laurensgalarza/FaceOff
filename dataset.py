from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from facenet_pytorch import MTCNN

class SiameseDataset(Dataset):
    def __init__(self, makeup_dir, no_makeup_dir, mtcnn, image_size=112):
        self.makeup_dir = makeup_dir
        self.no_makeup_dir = no_makeup_dir
        self.mtcnn = mtcnn
        self.people = list(set(os.listdir(makeup_dir)).intersection(os.listdir(no_makeup_dir)))
        
        # self.makeup_files = os.listdir(makeup_dir)
        # self.no_makeup_files = os.listdir(no_makeup_dir)
        # self.common_files = list(set(self.makeup_files).intersection(set(self.no_makeup_files)))

    def __len__(self):
        return len(self.people)

    # def load_face(self, image_path):
    #     img = Image.open(image_path).convert("RGB")
    #     face = self.mtcnn(img)
    #     if face is None:
    #         raise ValueError(f"No face detected in {image_path}")
    #     return face

    def __getitem__(self, idx):
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            person = self.people[idx]
            img0_path = os.path.join(self.makeup_dir, person)
            img1_path = os.path.join(self.no_makeup_dir, person)
            label = 0.0
        else:
            img0_name = random.choice(self.people)
            img1_name = random.choice([f for f in self.people if f != img0_name])
            img0_path = os.path.join(self.makeup_dir, img0_name)
            img1_path = os.path.join(self.no_makeup_dir, img1_name)
            label = 1.0

        img0 = Image.open(img0_path).convert("RGB")
        img1 = Image.open(img1_path).convert("RGB")

        img0_tensor = self.mtcnn(img0)
        img1_tensor = self.mtcnn(img1)

        if img0_tensor is None or img1_tensor is None:
            return self.__getitem__((idx + 1) % len(self))
        
        return img0_tensor, img1_tensor, torch.tensor([label], dtype=torch.float32)
        # try:
        #     img0 = self.load_face(img0_path)
        #     img1 = self.load_face(img1_path)
        # except Exception as e:
        #     print(f"Skipping pair due to face detection failure: {e}")
        #     return self.__getitem__((idx + 1) % len(self))



        # return img0, img1, label