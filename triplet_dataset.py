from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from torch.utils.data import Dataset

def _valid_images(folder):
    return [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

class TripletDataset(Dataset):
    def __init__(self, makeup_dir, no_makeup_dir, mtcnn, image_size=112):
        self.makeup_dir = makeup_dir
        self.no_makeup_dir = no_makeup_dir
        self.mtcnn = mtcnn
        self.image_size = image_size
        # only people with both makeup and no makeup are included
        self.people = list(set(_valid_images(makeup_dir)).intersection(_valid_images(no_makeup_dir)))

    def __len__(self):
        return len(self.people) * 2

    def __getitem__(self, idx):
        # Anchor and Positive: same person
        anchor_person = self.people[idx % len(self.people)]
        anchor_path = os.path.join(self.no_makeup_dir, anchor_person) # anchor image is a face with no makeup
        positive_path = os.path.join(self.makeup_dir, anchor_person) # same face but with makeup

        # Negative: different person
        negative_person = np.random.choice([p for p in self.people if p != anchor_person])
        negative_path = os.path.join(self.makeup_dir, negative_person) # negative image is a different face with makeup

        # Uses MTCNN to crop and align faces
        anchor = self.mtcnn(Image.open(anchor_path).convert("RGB"))
        positive = self.mtcnn(Image.open(positive_path).convert("RGB"))
        negative = self.mtcnn(Image.open(negative_path).convert("RGB"))

        if anchor is None or positive is None or negative is None:
            print(f"[MTCNN Warning] Face not detected in: {anchor_path if anchor is None else ''} {positive_path if positive is None else ''} {negative_path if negative is None else ''}")
            return self.__getitem__((idx + 1) % len(self))

        return anchor, positive, negative

