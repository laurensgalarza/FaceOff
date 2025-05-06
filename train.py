import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from PIL import Image
import random

class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.image_paths = {
            cls: [os.path.join(root_dir, cls, img) for img in os.listdir(os.path.join(root_dir, cls))]
            for cls in self.classes
        }

    def __len__(self):
        return 10000
    
    def __getitem__(self, idx):
        anchor_class = random.choice(self.classes)
        negative_class = random.choice([cls for cls in self.classes if cls != anchor_class])

        anchor_img = Image.open(random.choice(self.image_paths[anchor_class])).convert("RGB")
        positive_img = Image.open(random.choice(self.image_paths[anchor_class])).convert("RGB")
        negative_img = Image.open(random.choice(self.image_paths[negative_class])).convert("RGB")

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)

        return anchor_img, positive_img, negative_img
    
class FaceOff(nn.Module):
    def __init__(self, embedding_size=128):
        super(FaceOff, self).__init__()
        resnet = models.resnet18(pretrained=True)
        resnet.fc = nn.Linear(resnet.fc.in_features, embedding_size)
        self.backbone = nn.Sequential(resnet)
    
    def forward(self, x):
        x = self.backbone(x)
        return nn.functional.normalize(x)
    
def train():
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    dataset = TripletFaceDataset("dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceOff().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.TripletMarginLoss(margin=1.0)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for i, (anchor, positive, negative) in enumerate(dataloader):
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            anchor_emb = model(anchor)
            positive_emb = model(positive)
            negative_emb = model(negative)

            loss = criterion(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {epoch}, Step {i}, Loss: {loss.item():.4f}")
            
        print(f"Epoch {epoch} done. Avg loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), "face_model.pth")
    print("Model saved as face_model.pth")
    return model

if __name__ == "__main__":
    model = train()
