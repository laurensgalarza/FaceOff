import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from model import FaceOff 

def show_result(img1, img2, distance, threshold):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img1)
    axes[0].set_title("Image 1")
    axes[0].axis("off")
    
    axes[1].imshow(img2)
    axes[1].set_title("Image 2")
    axes[1].axis("off")

    plt.suptitle(f"Dissimilarity: {distance:.2f} | Predicted: {'Same' if distance < threshold else 'Different'}")
    plt.show()

def infer(image_path1, image_path2, model_path="./output/siamese_model.pth", threshold=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceOff().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    mtcnn = MTCNN(image_size=112, margin=10)

    # Load and preprocess images
    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")

    face1 = mtcnn(img1)
    face2 = mtcnn(img2)

    if face1 is None or face2 is None:
        print("Face detection failed on one or both images.")
        return

    face1 = face1.unsqueeze(0).to(device)
    face2 = face2.unsqueeze(0).to(device)

    with torch.no_grad():
        output1, output2 = model(face1, face2)
        distance = F.pairwise_distance(output1, output2).item()

    show_result(img1, img2, distance, threshold)

if __name__ == '__main__':
    # Replace with actual file paths
    infer("archive/no_makeup/5.jpg", "archive/with_makeup/5.jpg")
