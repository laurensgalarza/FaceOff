import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, UnidentifiedImageError
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt
from model import FaceOff 
import os

def show_result(img1, img2, distance, threshold):
    """
    Displays the input images side by side, along with the similarity score

    Args:
        img1 (PIL.Image): First face image
        img2 (PIL.Image): Second face image
        distance (float): Euclidean distance used to determine how similar the face tensors are
        threshold (float): Cutoff used to determine whether the distance indicates the same or a different person
    """
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
    """
    Runs inference on two face images using trained network and displays the result.
    
    Loads a trained model, detects faces in the provided images, computes their similarity, and shows the prediction.

    Args:
        image_path1 (string): Path to the first image file
        image_path2 (string): Path to the second image file
        model_path (string, optional): Path to the trained Siamese model file. Defaults to "./output/siamese_model.pth"
        threshold (float, optional): Threshold for determining if the images are of the same person. Defaults to 1.0
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Find model
    model = FaceOff().to(device)
    
    #Return if model file not found
    if not os.path.exists(model_path):
        print(f"Model file not found at path: {model_path}")
        return
    
    # Load model
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    model.eval()
    
    mtcnn = MTCNN(image_size=112, margin=10)

    # Load and preprocess images
    try:
        img1 = Image.open(image_path1).convert("RGB")
        img2 = Image.open(image_path2).convert("RGB")
    # Errors:
    # File not found 
    except FileNotFoundError as e:
        print(f"Image file not found: {e}")
        return
    # File found but is not a valid image
    except UnidentifiedImageError as e:
        print(f"Failed to open, not a valid image file: {e}")
        return
    #Other
    except Exception as e:
        print(f"An error occurred when opening images: {e}")
        return        
        
    #Detect the faces    
    face1 = mtcnn(img1)
    face2 = mtcnn(img2)
    
    #Return if no faces detected
    if face1 is None or face2 is None:
        print("Face detection failed on one or both images.")
        return

    face1 = face1.unsqueeze(0).to(device)
    face2 = face2.unsqueeze(0).to(device)

    #Run inference
    with torch.no_grad():
        try:
            output1, output2 = model(face1, face2)
            distance = F.pairwise_distance(output1, output2).item()
        except Exception as e:
            print(f"Inference failed: {e}")
            return

    show_result(img1, img2, distance, threshold)

if __name__ == '__main__':
    # Replace with actual file paths
    infer("archive/no_makeup/5.jpg", "archive/with_makeup/5.jpg")
