from facenet_pytorch import MTCNN
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms

# Initialize MTCNN
mtcnn = MTCNN(image_size=112, margin=10, post_process=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# Path to a folder of images
image_folder = 'archive/no_makeup'  # or 'archive/no_makeup'

# List image files
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

selected_files = {
    "2.jpg",
    "3.jpg",
    "11.jpg",
    "5.jpg",
    "6.jpg",
    "7.jpg",
    "8.jpg",
    "9.jpg",
    "10.jpg",
    "40.png"
}
# Show aligned faces
for i, filename in enumerate(selected_files):  # Show first 10 for example
    path = os.path.join(image_folder, filename)
    print(f"{filename}")
    img = Image.open(path).convert("RGB")

    aligned = mtcnn(img)

    if aligned is not None:
        aligned_pil = transforms.ToPILImage()(aligned)
        plt.subplot(2, 5, i + 1)
        plt.imshow(aligned_pil)
        plt.title(f"{filename}")
        plt.axis('off')
    else:
        print(f"[MTCNN Warning] Face not detected in: {filename}")
        Image.open(path).show()

plt.suptitle("MTCNN-Aligned Faces", fontsize=16)
plt.tight_layout()
plt.show()
