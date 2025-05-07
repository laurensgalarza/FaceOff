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
    def __init__(self, makeup_dir, no_makeup_dir, mtcnn, transform=None, image_size=100):
        self.makeup_dir = makeup_dir
        self.no_makeup_dir = no_makeup_dir
        self.transform = transform
        self.mtcnn = mtcnn
        
        self.makeup_files = os.listdir(makeup_dir)
        self.no_makeup_files = os.listdir(no_makeup_dir)
        self.common_files = list(set(self.makeup_files).intersection(set(self.no_makeup_files)))

    def __len__(self):
        return len(self.common_files)

    def load_face(self, image_path):
        img = Image.open(image_path).convert("RGB")
        face = self.mtcnn(img)
        if face is None:
            raise ValueError(f"No face detected in {image_path}")
        return face

    def __getitem__(self, idx):
        should_get_same_class = random.randint(0, 1)

        if should_get_same_class:
            filename = self.common_files[idx]
            img0_path = os.path.join(self.makeup_dir, filename)
            img1_path = os.path.join(self.no_makeup_dir, filename)
            label = 1
        else:
            img0_name = random.choice(self.makeup_files)
            img1_name = random.choice([f for f in self.no_makeup_files if f != img0_name])
            img0_path = os.path.join(self.makeup_dir, img0_name)
            img1_path = os.path.join(self.no_makeup_dir, img1_name)
            label = 0

        try:
            img0 = self.load_face(img0_path)
            img1 = self.load_face(img1_path)
        except Exception as e:
            print(f"Skipping pair due to face detection failure: {e}")
            return self.__getitem__((idx + 1) % len(self))

        return img0, img1, label

# class SiameseDataset(Dataset):
#     def __init__(self, imageFolderDataset, transform=None):
#         self.imageFolderDataset  = imageFolderDataset
#         self.transform = transform
#         self.mtcnn = MTCNN(image_size=100, margin=10, post_process=True)

#     def __getitem__(self, index):
#         while True:
#             img0_tuple = random.choice(self.imageFolderDataset.imgs)

#             # We need approximately 50% of images to be in the same class
#             should_get_same_class = random.randint(0,1)
#             if should_get_same_class:
#                 while True:
#                     # Look until the same class image is found
#                     img1_tuple = random.choice(self.imageFolderDataset.imgs)
#                     if img0_tuple[1] == img1_tuple[1]:
#                         break
#             else:
#                 while True:
#                     # Look until a different class image is found
#                     img1_tuple = random.choice(self.imageFolderDataset.imgs)
#                     if img0_tuple[1] != img1_tuple[1]:
#                         break
            
#             img0 = Image.open(img0_tuple[0])
#             img1 = Image.open(img1_tuple[0])

#             img0 = img0.convert("RGB")
#             img1 = img1.convert("RGB")

#             # if self.transform is not None:
#             #     img0 = self.transform(img0)
#             #     img1 = self.transform(img1)

#             img0_face = self.mtcnn(img0)
#             img1_face = self.mtcnn(img1)

#             if img0_face is None or img1_face is None:
#                 continue  # Retry another pair if detection failed

#             return img0_face, img1_face, torch.torch.tensor([int(img1_tuple[1] != img0_tuple[1])], dtype=torch.float32)
    
    # def __len__(self):
    #     return len(self.imageFolderDataset.imgs)