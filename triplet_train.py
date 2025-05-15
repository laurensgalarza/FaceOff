import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from model import FaceOff
from triplet_dataset import TripletDataset
from facenet_pytorch import MTCNN

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initializes mtcnn to crop and align faces
    mtcnn = MTCNN(image_size=112, margin=10, post_process=True, device=device)

    # loads dataset of triplets
    full_dataset = TripletDataset("archive/with_makeup", "archive/no_makeup", mtcnn) 
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16, num_workers=0)
    # val_loader = DataLoader(val_dataset, shuffle=False, batch_size=16, num_workers=0)

    # Loads the model
    model = FaceOff().to(device)
    criterion = nn.TripletMarginLoss(margin=0.5) # defines loss function that guides how the model learns
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # adjust learning rates of parameters during training
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) # adjusts learning rate during training

    # each epoch sets the model to training mode and iterates over batchest of triplets from the dataset
    for epoch in range(30):
        model.train()
        total_loss = 0.0
        for i, (anchor, positive, negative) in enumerate(train_loader):
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)
            
            # gets embeddings of each image
            anchor_out = model.forward_once(anchor)
            positive_out = model.forward_once(positive)
            negative_out = model.forward_once(negative)
            
            # computes triplet loss using the embeddings
            loss = criterion(anchor_out, positive_out, negative_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/30 | Train Loss: {avg_train_loss:.4f}")

    torch.save(model.state_dict(), "./output/triplet_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
