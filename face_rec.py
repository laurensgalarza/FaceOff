import os
import cv2
import cv2.data
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from train import FaceOff

model = FaceOff()
model.load_state_dict(torch.load("face_model.pth"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print (f"No face found in: {image_path}")
        return None
    
    x, y, w, h = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (112, 112))
    face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    face_tensor = transform(face_pil).unsqueeze(0)

    with torch.no_grad():
        embedding = model(face_tensor).squeeze(0)

    return embedding

def compare_faces(img_folder):
    embeddings = {}
    for filename in os.listdir(img_folder):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(img_folder, filename)
        emb = extract_embedding(path)
        if emb is not None:
            embeddings[filename] = emb

    keys = list(embeddings.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            name1 = keys[i]
            name2 = keys[j]
            dist = np.linalg.norm(embeddings[name1] - embeddings[name2])
            print(f"Distacne between {name1} and {name2}: {dist:.4f}")


if __name__ == "__main__":
    compare_faces("dataset")