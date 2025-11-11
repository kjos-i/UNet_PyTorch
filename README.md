# UNet implementation using PyTorch

This is a simple UNet implementation to run on your own computer or Google Colab
<br>

U-Net: Convolutional Networks for Biomedical Image Segmentation <br>
Olaf Ronneberger, Philipp Fischer, Thomas Brox - Submitted on 18 May 2015 <br>

UNet implementation for binary segmentation (background, foreground) <br>
Image and mask size should be 512 x 512, see CHANGE_SIZE variable  <br>
Mask values should be binary (background=0 and foreground=1), see SCALE_MASK variable <br>
<br>

## Quick start to run on your own computer 

1. Make sure to have a separate test dataset. Arrange your training dataset (training and validation combined) like this: 
<img width="220" height="153" alt="Skjermbilde 2025-11-07 203630" src="https://github.com/user-attachments/assets/bc13a00d-c9f6-48e9-92b9-e4bf5317411a" />

2. Download the four .py files to one folder: <br>
UNet_PyTorch_training.py <br>
UNet_PyTorch_model.py <br>
UNet_PyTorch_dataset.py <br>
UNet_PyTorch_utils.py <br>

3. Open the UNet_PyTorch_training.py file in an editor (for example VS Code) and set these variables: <br>
AUGMENTATION = None # Options: None or "augmentation" (vertical and horizontal flips, each p=0.25) <br>
CHANGE_SIZE = None # Options: None, "center_crop" or "interpolation_nearest" <br>
SCALE_MASK = True # Scales pixel values from range [0, 255] to range [0.0, 1.0]
LIMIT: The nr of images included from dataset, to include all set value to None <br>
BATCH_SIZE <br>
PIN_MEMORY = True # Enables allocation of page-locked memory on the CPU for data fetched by DataLoader
EPOCHS <br>
INPUT_CHANNELS: Number of channels in input images, dataset class converts to RGB image (3 channels) <br>
NUM_CLASSES: For this inplementation the number of classes should be 1 <br>
LEARNING_RATE: For AdamW optimizer learning rate (LR), PyTorch default 0.001 <br>
WEIGHT_DECAY: For AdamW optimizer, PyTorch default 0.01 <br>
LR_S_STEP_SIZE: For LR scheduler StepLR, nr of epochs before applying gamma decay <br>
LR_S_GAMMA: For LR scheduler StepLR (LR * gamma = new LR), PyTorch default 0.1 <br>
THRESHOLD: Threshold for binary class prediction <br>
DICE_INCLUDE_BACKGROUND: TorchMetrics dice score calculation, default True <br>
ROOT_PATH_SAVE: Path where results folder will be created <br>
CHECKPOINT_PATH: Path to checkpoint file with saved model, optimizer and scheduler parameters (optional) <br>

4. Check that the Python environment has all necessary packages installed, otherwise install them
5. Run the UNet_PyTorch_training.py file
6. The results will be stored in a new folder at the specified save path

## Quick start to run using Google Colab

1. Make sure to have a separate test dataset. Arrange your training dataset (training and validation combined) like this on Google Drive: 
<img width="441" height="238" alt="Skjermbilde 2025-11-07 213123" src="https://github.com/user-attachments/assets/1ad221b3-15d9-45e2-813f-2218b8f8c08e" />

2. Download the three .py files:
UNet_PyTorch_model.py <br>
UNet_PyTorch_dataset.py <br>
UNet_PyTorch_utils.py <br>

3. Get a copy of the UNet_PyTorch_colab.ipynb file, store it in Google Drive - My Drive - Colab Notebooks and open and run it block by block in Google Colab
4. Remember to set the variables as explained under "Quick start to run on your own computer - point 3"
5. The results will be stored in a new folder at the specified save path
