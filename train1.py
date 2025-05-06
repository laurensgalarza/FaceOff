import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import FaceOff, ContrastiveLoss
from dataset import SiameseDataset
from facenet_pytorch import MTCNN


# Initialize MTCNN for face detection + alignment
mtcnn = MTCNN(image_size=100, margin=20)

def train():
    
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])
    folder_dataset = datasets.ImageFolder(root="./archive")
    # siamese_dataset = SiameseDataset(folder_dataset, transform=None)
    # dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=64, num_workers=2)
    # Create dataset and dataloader
    siamese_dataset = SiameseDataset("archive/with_makeup", "archive/no_makeup", mtcnn)
    dataloader = DataLoader(siamese_dataset, shuffle=True, batch_size=32, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = FaceOff().to(device)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

    for epoch in range(10):
        for i, (img0, img1, label) in enumerate(dataloader):
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            output1, output2 = net(img0, img1)
            loss = criterion(output1, output2, label)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Epoch {epoch} Batch {i} Loss {loss.item():.4f}")

    torch.save(net.state_dict(), './output/siamese_model.pth')

if __name__ == "__main__":
    train()