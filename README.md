# FaceOff

FaceOff is a facial recognition system that compares two images and predicts whether they depict the same person, even when one includes makeup. It uses a Siamese neural network trained to focus on facial features that stay consistent despite alterations resulting from makeup. This was completed as our final project for a Computer Vision course.

## How It Works

Face pairs are passed into a Siamese model. During training, it:

- Minimizes the distance between feature tensors of the same person
- Maximizes the distance between different people

This helps the model learn implicitly which facial features are consistent across makeup changes — such as face shape, eye position, or bone structure.

## Tech Stack

- Python
- PyTorch
- OpenCV
- matplotlib
- torchvision
- facenet-pytorch
- NumPy
- Pillow

## Running the Project

### Terminal Command to install libraries used:
pip install torch torchvision pillow matplotlib opencv-python numpy facenet-pytorch

### When running the program for the first time, you should run every file in this order:
dataset.py
model.py
train1.py
inference.py

The command for running the files should just be: python file_name

### Notes
train1.py might take a while and it should go up to 30 epochs.
After running train1.py, there should be a file added to the output folder named "siamese_model.pth"
This file is the saved weights of the trained Siamese neural network. 
It contains all of the learned parameters.

inference.py is used for testing. 
To test different pictures, just change the file paths at the bottom of inference.py.
You do not need to run every file again to test different pictures because the trained 
neural network is already saved. 

## Future Implementation Considerations

- Include a wider range of facial variations beyond makeup, such as lighting, expression, angle, ethnicity, etc.
- Integrate data featuring exaggerated makeup and gender-nonconforming people to challenge the model’s assumptions and expand its training scope.
- Consider further edge cases where face shape may be altered by heavy cosmetics or accessories.
