import torch
import os
import random
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import seaborn as sns
sns.set_theme()


# Graphs showing training and validation results over epochs for loss, IoU, Dice and accuracy
def train_loss_iou_dice_acc_graph(save_path, epochs_list, train_losses, val_losses, 
                                  train_ious, val_ious, train_dices, val_dices, train_accs, val_accs):
    fig, axes = plt.subplots(2, 2, constrained_layout=True, figsize=(10, 8))
    ax1, ax2, ax3, ax4 = axes.flatten()
    ax1.plot(epochs_list, train_losses, label='Training Loss', lw=3)
    ax1.plot(epochs_list, val_losses, label='Validation Loss', lw=3)
    ax2.plot(epochs_list, train_ious, label='Training IoU', lw=3)
    ax2.plot(epochs_list, val_ious, label='Validation IoU', lw=3)
    ax3.plot(epochs_list, train_dices, label='Training Dice', lw=3)
    ax3.plot(epochs_list, val_dices, label='Validation Dice', lw=3)
    ax4.plot(epochs_list, train_accs, label='Training Accuracy', lw=3)
    ax4.plot(epochs_list, val_accs, label='Validation Accuracy', lw=3)
    
    ax1.set_title("Loss over epochs", fontsize=14); ax2.set_title('IoU over epochs', fontsize=14)
    ax3.set_title("Dice over epochs", fontsize=14); ax4.set_title('Accuracy over epochs', fontsize=14)
    ax1.set_ylabel("Loss"); ax2.set_ylabel("IoU"); ax3.set_ylabel("Dice"); ax4.set_ylabel("Accuracy")
    ax1.set_xlabel("Epochs"); ax2.set_xlabel("Epochs"); ax3.set_xlabel("Epochs"); ax4.set_xlabel("Epochs")
    ax2.set_ylim(0, 1); ax3.set_ylim(0, 1); ax4.set_ylim(0, 1)
    ax1.legend(); ax2.legend(); ax3.legend(); ax4.legend()
    fig.suptitle("Training and validation graphs", fontsize=16)
    plt.savefig(os.path.join(save_path, "training_loss_iou_dice_acc_graph.png"))
    plt.show()


# Examples from validation of image, mask and prediction
def val_image_mask(save_path, epoch, idx, img, mask, prediction):
    """
    1: Split: Split batch dimension
    2: Squeeze: Remove batch dimension
    3: Resize: Restore size to original image size
    4: Permute: Channel in last position
    5: Send to cpu
    6: Create and save plot (first example from each batch)
    """
    img = (torch.split(img, 1, dim=0)) 
    mask = (torch.split(mask, 1, dim=0))
    prediction = (torch.split(prediction, 1, dim=0))
    
    img = torch.squeeze(img[0], dim=0)
    mask = torch.squeeze(mask[0], dim=0)
    prediction = torch.squeeze(prediction[0], dim=0)
    
    img = torch.permute(img, (1, 2, 0))
    mask = torch.permute(mask, (1, 2, 0))
    prediction = torch.permute(prediction, (1, 2, 0))
    
    img, mask, prediction = img.detach().cpu(), mask.detach().cpu(), prediction.detach().cpu()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5), constrained_layout=True) 
    ax1.imshow(img)
    ax1.set_title(f"Real Image, epoch {epoch}, index {idx}")
    ax1.set_axis_off()
    ax2.imshow(mask)
    ax2.set_title(f"Segmented Mask, epoch {epoch}, index {idx}")
    ax2.set_axis_off()
    ax3.imshow(prediction)
    ax3.set_title(f"Predicted Mask, epoch {epoch}, index {idx}")
    ax3.set_axis_off()
    fig.suptitle("Examples from validation", fontsize=16)
    plt.savefig(os.path.join(save_path, f"image_mask_val_{epoch}_{idx}.png"))
