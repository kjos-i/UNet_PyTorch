# UNet implementation using PyTorch

This is a simple UNet implementation to run on your own computer or Google Colab


## Quick start to run on your own computer

1. For this UNet implementation, the image masks must be binary with background = 0 and class label = 1

2. Make sure to have a separate test dataset. Arrange your training dataset (training and validation combined) like this: 

<img width="220" height="153" alt="Skjermbilde 2025-11-07 203630" src="https://github.com/user-attachments/assets/bc13a00d-c9f6-48e9-92b9-e4bf5317411a" />

3. Download the four .py files to one folder:

<img width="213" height="139" alt="Skjermbilde 2025-11-07 210211" src="https://github.com/user-attachments/assets/3f6fb4fd-ad77-494e-a4e1-a65f4b97a413" />

4. Open the UNet_PyTorch_training.py file in an editor (for example VS Code) and set these variables:

<img width="1244" height="429" alt="Skjermbilde 2025-11-07 210921" src="https://github.com/user-attachments/assets/f6d7a6ee-5748-44ec-83b6-3894e11579e3" />

5. Check that the Python environment has all necessary packages installed, otherwise install them

6. Run the UNet_PyTorch_training.py file

7. The results will be stored in a new folder at the specified save path


## Quick start to run using Google Colab

1. For this UNet implementation, the image masks must be binary with background = 0 and class label = 1

2. Make sure to have a separate test dataset. Arrange your training dataset (training and validation combined) like this on Google Drive: 

<img width="441" height="238" alt="Skjermbilde 2025-11-07 213123" src="https://github.com/user-attachments/assets/1ad221b3-15d9-45e2-813f-2218b8f8c08e" />

3. Download the three .py files:

<img width="284" height="115" alt="Skjermbilde 2025-11-07 213642" src="https://github.com/user-attachments/assets/e6cb66ae-c862-44b6-ad23-2b9bc05da1ac" />

4. Get a copy of the UNet_PyTorch_colab.ipynb file, store it in Google Drive - My Drive - Colab Notebooks and open and run it in Google Colab

5. Set variables as explained

