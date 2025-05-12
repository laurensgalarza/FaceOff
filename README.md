# FaceOff

Terminal Command to install libraries used:

pip install torch torchvision pillow matplotlib opencv-python numpy facenet-pytorch

When running the program for the first time, you should run every file in this order:
dataset.py
model.py
train1.py
inference.py

The command for running the files should just be: python file_name

train1.py might take a while and it should go up to 30 epochs.
After running train1.py, there should be a file added to the output folder named "siamese_model.pth"
This file is the saved weights of the trained Siamese neural network. 
It contains all of the learned parameters.

inference.py is used for testing. 
To test different pictures, just change the file paths at the bottom of inference.py.
You do not need to run every file again to test different pictures because the trained 
neural network is already saved. 