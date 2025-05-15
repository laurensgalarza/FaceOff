import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from model import FaceOff

def infer(image_path1, image_path2, model_path="./output/triplet_model.pth", threshold=1.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load trained model
    model = FaceOff().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # MTCNN face detector
    mtcnn = MTCNN(image_size=112, margin=10, post_process=True, device=device)

    # Load and align both images
    img1 = Image.open(image_path1).convert("RGB")
    img2 = Image.open(image_path2).convert("RGB")

    face1 = mtcnn(img1)
    face2 = mtcnn(img2)

    if face1 is None or face2 is None:
        print(f"[MTCNN Warning] Face not detected in: {image_path1 if face1 is None else ''} {image_path2 if face2 is None else ''}")
        return

    face1 = face1.unsqueeze(0).to(device)
    face2 = face2.unsqueeze(0).to(device)

    # Compute embeddings and distance
    with torch.no_grad():
        emb1 = model.forward_once(face1)
        emb2 = model.forward_once(face2)
        distance = F.pairwise_distance(emb1, emb2).item()

    prediction = "Same" if distance < threshold else "Different"
    print(f"Dissimilarity: {distance:.2f} → Prediction: {prediction}")

    # Visualize
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img1)
    axes[0].set_title("Image 1")
    axes[0].axis("off")
    axes[1].imshow(img2)
    axes[1].set_title("Image 2")
    axes[1].axis("off")
    plt.suptitle(f"Dissimilarity: {distance:.2f} → {prediction}")
    plt.show()

if __name__ == '__main__':
    # Replace with actual file paths
    # 15 and 6
    # 3 and 3
    # 44 and 44 probably because they have different lighting
    # 34 and 34 probably because the dataset doesn't have a variety of skin colors
    infer("archive/no_makeup/30.png", "archive/with_makeup/30.png")
