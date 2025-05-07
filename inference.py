from model import FaceOff
from dataset import SiameseDataset
from torchvision import transforms, datasets
import torch
import torch.utils.data
import torchvision.utils
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
from facenet_pytorch import MTCNN

def imshow(img, text=None):
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(60, 8, text, style='italic', fontweight='bold',
                 bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def main():
    # Load the trained model
    model_path = "./output/siamese_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaceOff().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transform
    transformation = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    mtcnn = MTCNN(image_size=100, margin=0)

    # Load test dataset
    # folder_dataset_test = datasets.ImageFolder(root="./archive/")
    siamese_dataset = SiameseDataset(
        # imageFolderDataset=folder_dataset_test,
        makeup_dir="./archive/with_makeup",
        no_makeup_dir="./archive/no_makeup",
        transform=transformation,
        mtcnn=mtcnn
    )

    test_dataloader = torch.utils.data.DataLoader(
        siamese_dataset,
        num_workers=2,  # <-- change to 0 if debugging
        batch_size=1,
        shuffle=True
    )

    # Run inference
    dataiter = iter(test_dataloader)
    x0, _, _ = next(dataiter)

    for i in range(10):
        img0, img1, label = next(dataiter)
        
        output1, output2 = model(img0.to(device), img1.to(device))
        euclidean_distance = F.pairwise_distance(output1, output2)

        concatenated = torch.cat((img0, img1), 0)
        threshold = 1.0  # You can tune this based on validation set
        prediction = "Same" if euclidean_distance.item() < threshold else "Different"

        imshow(torchvision.utils.make_grid(concatenated),   
            f'Dissimilarity: {euclidean_distance.item():.2f} | Predicted: {prediction} | Ground Truth: {"Same" if label.item() == 0 else "Different"}')

if __name__ == "__main__":
    main()

