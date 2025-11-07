import torch
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchmetrics
from torchmetrics.segmentation import DiceScore

from tqdm import tqdm
import pandas as pd
import os
import seaborn as sns
sns.set_theme()

from UNet_PyTorch_dataset import MyDataset
from UNet_PyTorch_model import UNet
from UNet_PyTorch_matplotlib import val_image_mask, train_loss_iou_dice_acc_graph

"""
UNet set up for segmentation of binary images (background = 0, class label = 1)
"""

# SET VARIABLES
ROOT_PATH_DATASET = "C:/Users/kjosi/Python/carvana_dataset" # Path to main dataset folder  
TRANSFORM = "transform" # OR "transform_augmentation", but make sure it works on dataset!!
LIMIT = 10 # The nr of images included from dataset, to include all set value to None
BATCH_SIZE = 1
EPOCHS = 10
LEARNING_RATE = 0.0001 # For AdamW optimizer learning rate (LR), PyTorch default 0.001
WEIGHT_DECAY = 0.01 # For AdamW optimizer, PyTorch default 0.01
LR_S_STEP_SIZE = 5 # For LR scheduler StepLR, nr of epochs before applying gamma decay
LR_S_GAMMA = 0.1 # For LR scheduler StepLR (LR * gamma = new LR), PyTorch default 0.1
THRESHOLD = 0.5 # Threshold for binary class prediction
DICE_INCLUDE_BACKGROUND = True # TorchMetrics dice score calculation, default True
ROOT_PATH_SAVE = "C:/Users/kjosi/Python" # Path where results folder will be created
CHECKPOINT_PATH = None # Path to checkpoint file with saved model, optimizer and scheduler parameters


# SET DEVICE, RUN ON CUDA IF AVAILABLE 
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("\n")
print("#" * 50)
print(f"Running on {device}")
print("#" * 50)
print("\n")

torch.cuda.empty_cache()


# LOAD DATASET
ROOT_PATH_DATASET = ROOT_PATH_DATASET
TRANSFORM = TRANSFORM
LIMIT = LIMIT
dataset = MyDataset(root_path=ROOT_PATH_DATASET, transform=TRANSFORM, limit=LIMIT)
generator = torch.Generator().manual_seed(25)

train_dataset, val_dataset = random_split(dataset, [0.8, 0.2], generator=generator)


LEARNING_RATE = LEARNING_RATE
BATCH_SIZE = BATCH_SIZE

train_dataloader = DataLoader(dataset=train_dataset,
                              pin_memory=False,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
val_dataloader = DataLoader(dataset=val_dataset,
                              pin_memory=False,
                              batch_size=BATCH_SIZE,
                              shuffle=True)


# SET MODEL COST FUNCTION, OPTIMIZER AND LEARNING RATE SCHEDULER
in_channels = 3
num_classes = 1 

model = UNet(in_channels, num_classes).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = lr_scheduler.StepLR(optimizer, step_size=LR_S_STEP_SIZE, gamma=LR_S_GAMMA) 

if CHECKPOINT_PATH != None:
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

model_name = model.__class__.__name__
optimizer_name = optimizer.__class__.__name__
criterion_name = criterion.__class__.__name__
scheduler_name = scheduler.__class__.__name__


# SAVE RESULTS
ROOT_PATH_SAVE = ROOT_PATH_SAVE
save_directory = "UNet_training_results"
save_path = os.path.join(ROOT_PATH_SAVE, save_directory)

if not os.path.exists(save_path):
    os.mkdir(save_path)
else:
    x = 1
    while os.path.exists(save_path):
        new_save_directory = (f"{save_directory}_{x + 1}")
        save_path = os.path.join(ROOT_PATH_SAVE, new_save_directory)
        x += 1
    os.mkdir(save_path)   


# TRAIN AND EVALUATE
EPOCHS = EPOCHS
train_losses = []
train_dices = []
train_ious = []
train_dices = []
train_accs = []
val_losses = []
val_ious = []
val_dices = []
val_accs = []
lr_updates = []

for epoch in tqdm(range(EPOCHS), "EPOCHS"): 
    model.train()
    train_loss_sum = 0 
    train_iou_sum = 0
    train_dice_sum = 0
    train_acc_sum = 0
    nr_of_train_loss_items = 0

    jaccardindex = torchmetrics.JaccardIndex(task='binary').to(device) 
    dice_segmentation = DiceScore(num_classes=1, include_background=DICE_INCLUDE_BACKGROUND).to(device)
    accuracy = torchmetrics.Accuracy(task='binary').to(device)

    print("-" * 50)
    print(f"Beginning of epoch {epoch + 1}")
    print("-" * 50)

    for idx, img_mask in enumerate(tqdm(train_dataloader, "BATCH TRAIN", position=0, leave=True)):
        img = img_mask[0].float().to(device)
        mask = img_mask[1].float().to(device)
        
        mask_pred = model(img)
        optimizer.zero_grad()

        loss = criterion(mask_pred, mask)

        prediction = torch.sigmoid(mask_pred)
        prediction[prediction < PROBABILITY] = 0
        prediction[prediction >= PROBABILITY] = 1
        
        prediction_int = prediction.long()
        mask_int = mask.long()

        iou = jaccardindex(prediction_int, mask_int)
        dice = dice_segmentation(prediction_int, mask_int)
        acc = accuracy(prediction_int, mask_int)
        
        train_loss_sum += loss.item() 
        train_iou_sum += iou.item()
        train_dice_sum += dice.item()
        train_acc_sum += acc.item()
        nr_of_train_loss_items += 1

        loss.backward()
        optimizer.step()
    
    scheduler.step()
    updated_lr = scheduler.get_last_lr()
    lr_updates.append(updated_lr)

    train_loss = train_loss_sum / len(train_dataloader)
    train_iou = train_iou_sum / len(train_dataloader)
    train_dice = train_dice_sum / len(train_dataloader)
    train_acc = train_acc_sum / len(train_dataloader)

    train_losses.append(train_loss)
    train_ious.append(train_iou)
    train_dices.append(train_dice)
    train_accs.append(train_acc)
    
    print(f"End of training part of epoch {epoch + 1}")
    print(f"Updated learning rate: {updated_lr}")
    print(f"Batch size {BATCH_SIZE}, length train_dataset {len(train_dataset)}")
    print(f"Final index plus one {idx + 1}, length train_dataloader {len(train_dataloader)}")
    print(f"Total number of images {(idx + 1) * BATCH_SIZE}")
    print(f"Items added to train_loss_sum per epoch {nr_of_train_loss_items}")
    print("-" * 5)
   
    
    model.eval()
    val_loss_sum = 0
    val_iou_sum = 0
    val_dice_sum = 0
    val_acc_sum = 0
    nr_of_val_loss_items = 0
    with torch.no_grad():
        for idx, img_mask in enumerate(tqdm(val_dataloader, "BATCH VAL", position=0, leave=True)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device) 
            
            mask_pred = model(img)
            loss = criterion(mask_pred, mask)

            prediction = torch.sigmoid(mask_pred)
            prediction[prediction < PROBABILITY] = 0
            prediction[prediction >= PROBABILITY] = 1

            prediction_int = prediction.long()
            mask_int = mask.long()
            
            iou = jaccardindex(prediction_int, mask_int)
            dice = dice_segmentation(prediction_int, mask_int)
            acc = accuracy(prediction_int, mask_int)

            val_loss_sum += loss.item()
            val_iou_sum += iou.item()
            val_dice_sum += dice.item()
            val_acc_sum += acc.item()
            nr_of_val_loss_items += 1

            # SAVE IMAGES FOR SOME OF THE PREDICTIONS
            if idx == 0 and epoch > EPOCHS - 5:      
                val_image_mask(save_path, epoch, idx, img, mask, prediction)

        val_loss = val_loss_sum / len(val_dataloader)
        val_iou = val_iou_sum / len(val_dataloader)
        val_dice = val_dice_sum / len(val_dataloader)
        val_acc = val_acc_sum / len(val_dataloader)

        val_losses.append(val_loss)
        val_ious.append(val_iou)
        val_dices.append(val_dice)
        val_accs.append(val_acc)


        print(f"End of validation part of epoch {epoch + 1}")
        print(f"Batch size {BATCH_SIZE}, length train_dataset {len(val_dataset)}")
        print(f"Final index plus one {idx + 1}, length val_dataloader {len(val_dataloader)}")
        print(f"Total number of images {(idx + 1) * BATCH_SIZE}")
        print(f"Items added to val_loss_sum per epoch {nr_of_val_loss_items}")
        
    print("-" * 50)
    print(f"End of epoch {epoch + 1}")
    print("-" * 50)
    print(f"Training Loss EPOCH {epoch + 1}: {train_loss:.4f}")
    print(f"Training IoU EPOCH {epoch + 1}: {train_iou:.4f}")
    print("-" * 50)
    print(f"Validation Loss EPOCH {epoch + 1}: {val_loss:.4f}")
    print(f"Validation IoU EPOCH {epoch + 1}: {val_iou:.4f}")
    print("-" * 50)


    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()}
    torch.save(checkpoint, (os.path.join(save_path, f"checkpoint_{epoch + 1}.pth")))


# PANDAS DATAFRAME WITH TRAINING RESULTS
epochs_list = list(range(1, EPOCHS + 1))
lists = [epochs_list, train_losses, val_losses, train_ious, val_ious, 
         train_dices, val_dices, train_accs, val_accs, lr_updates]
titles = ["Epoch", "Train loss", "Val loss", "Train IoU", "Val IoU", 
          "Train dice", "Val dice", "Train acc", "Val acc", "Updated LR"]
training_results_dictionary = dict(zip(titles, lists))
df = pd.DataFrame(training_results_dictionary)
df.to_csv((os.path.join(save_path, "training_dataframe.csv")), index=False)
print("-" * 5)
print(df)  
print("-" * 50) 


# PANDAS DATAFRAME WITH GENERAL INFO
info_names = ["Model", "Criterion", "Optimizer", "Weight decay", "LR scheduler", "LR start", "LR step size", 
              "LR gamma", "Input channels", "Number of classes", "Length dataset", "Training images", 
              "Validation images", "Limit", "Batch size", "Epochs", "Probability", "Dice include background"]
info_items = [model_name, criterion_name, optimizer_name, WEIGHT_DECAY, scheduler_name, LEARNING_RATE, 
              LR_S_STEP_SIZE, LR_S_GAMMA, in_channels, num_classes, len(dataset), len(train_dataset), 
              len(val_dataset), LIMIT, BATCH_SIZE, EPOCHS, PROBABILITY, DICE_INCLUDE_BACKGROUND]

info_dict = {"Item": info_names, "Info": info_items}
info_df = pd.DataFrame(info_dict)
info_df.to_csv((os.path.join(save_path, "training_info.csv")), index=False)


# GRAPH SHOWING TRAINING AND VALIDATION LOSS AND DICE
loss_iou_dice_acc_graph = train_loss_iou_dice_acc_graph(save_path, epochs_list, train_losses, val_losses, 
                                  train_ious, val_ious, train_dices, val_dices, train_accs, val_accs)


torch.cuda.empty_cache()




