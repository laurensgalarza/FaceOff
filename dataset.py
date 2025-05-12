from torch.utils.data import Dataset
import random
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import MTCNN


class SiameseDataset(Dataset):  
    """
    Custom dataset for training the neural network to compare faces with or without makeup.
    Loads image pairs and determines whether the pair represents the same person (0.0) or different people (1.0)
    """
    def __init__(self, makeup_dir, no_makeup_dir, mtcnn, image_size=112):
        """
        makeup_dir (string): Directory containing images of faces with makeup on.
        no_makeup_dir (string): Directory containing images of faces without makeup.
        mtcnn (MTCNN): Multi-task Cascaded Convolutional Networks algorithm. Detects faces and facial landmarks.
        image_size (int, default=112): Size of the cropped and aligned face image.
        """
        self.makeup_dir = makeup_dir
        self.no_makeup_dir = no_makeup_dir
        self.mtcnn = mtcnn
        
        # Get list of filenames (representing people) that exist in both directories
        self.people = list(set(os.listdir(makeup_dir)).intersection(os.listdir(no_makeup_dir)))
        
        if not self.people:
            raise ValueError("Intersection is empty, no matching files present. Please make sure that image file names in makeup_dir are matching those in no_makeup_dir.")
        
        # self.makeup_files = os.listdir(makeup_dir)
        # self.no_makeup_files = os.listdir(no_makeup_dir)
        # self.common_files = list(set(self.makeup_files).intersection(set(self.no_makeup_files)))

    def __len__(self):
        """
        Returns the amount of people (matching files) found in the directories
        """
        return len(self.people)

    # def load_face(self, image_path):
    #     img = Image.open(image_path).convert("RGB")
    #     face = self.mtcnn(img)
    #     if face is None:
    #         raise ValueError(f"No face detected in {image_path}")
    #     return face

    def __getitem__(self, idx):
        """ 
        Detects an image pair of the face with and without makeup, then aligns and crops to the image_size.
        
        Returns:
            img0_tensor (Tensor): First face tensor (image cropped and aligned with MTCNN)
            img1_tensor (Tensor): Second face tensor
            label (Tensor): Float tensor, 0.0. if same person and 1.0 if different people
        """
        # attempts are used for infinite recursion prevention
        attempts_allowed = 5 
        
        for attempt in range(attempts_allowed):
            try: 
                # decide if returning pair of faces from the same person, or different people
                should_get_same_class = random.randint(0, 1)

                if should_get_same_class:
                    person = self.people[idx]
                    img0_path = os.path.join(self.makeup_dir, person)
                    img1_path = os.path.join(self.no_makeup_dir, person)
                    label = 0.0
                else:
                    img0_name = random.choice(self.people)
                    img1_name = random.choice([f for f in self.people if f != img0_name])
                    img0_path = os.path.join(self.makeup_dir, img0_name)
                    img1_path = os.path.join(self.no_makeup_dir, img1_name)
                    label = 1.0
                
                #Load images and convert to RGB
                img0 = Image.open(img0_path).convert("RGB")
                img1 = Image.open(img1_path).convert("RGB")
                
                #MTCNN aligns and crops images
                img0_tensor = self.mtcnn(img0)
                img1_tensor = self.mtcnn(img1)

                if img0_tensor is not None and img1_tensor is not None:
                    return img0_tensor, img1_tensor, torch.tensor([label], dtype=torch.float32)
                
                else:
                    print(f"Face detection failed on attempt {attempt+1}. Retrying current input...")
                
            except Exception as e:
                print(f"Failed to load image pair on attempt {attempt+1}: {e}")
                
            #Change index for next attempt
            idx = (idx + 1) % len(self)
            
        raise RuntimeError("Failed to retrieve a valid image pairs after multiple attempts")
            # try:
            #     img0 = self.load_face(img0_path)
            #     img1 = self.load_face(img1_path)
            # except Exception as e:
            #     print(f"Skipping pair due to face detection failure: {e}")
            #     return self.__getitem__((idx + 1) % len(self))



            # return img0, img1, label