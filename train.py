import os 
import random
import time 
import numpy as np 
import torch 
torch.set_num_threads(3)
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms
#from torchsummary import summary
from dataset import FaceData
from models import AutoEncoderConv
from datetime import datetime
import matplotlib.pyplot as plt 
from tqdm import tqdm 
from utils import save_images
from argparse import ArgumentParser as AP 
print(f"Imports complete")

get_timestamp = lambda : datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
get_extended_num = lambda num : (6-len(str(num)))*'0' + str(num)

parser = AP()
parser.add_argument("N_epochs_")
parser.add_argument("autosave")
args = parser.parse_args()

AUTO_SAVE = int(args.autosave)
N_Epochs = int(args.N_epochs_)
BATCH_SIZE = 4

# Load data
composed_transforms = transforms.ToTensor()
train_data = FaceData('./Data/Trump_Faces/train',transform=composed_transforms)
val_data = FaceData('./Data/Trump_Faces/val', transform=composed_transforms)

# Making data-loader elements
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False)
print(f"Data loaders created\n")


# Initialize the model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
model = AutoEncoderConv(1).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0003)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
loss_fn = nn.L1Loss()


# Train the model
print(f"Training started at:")
os.system("date")
model_autosave_filename = f"{model.model_name}_{get_timestamp()}.pt"
val_images_save_dir = f"{model_autosave_filename.replace('.pt', '')}"
os.system(f'mkdir ./{val_images_save_dir}')
print(f'Saving image progress to {val_images_save_dir}')
epoch_val_loss_list = []
epoch_val_acc_list = []
epoch_learning_rate_list = []
epoch_train_loss_list = []
num_batches = len(train_loader)

if AUTO_SAVE:
    print(f"Autosave model filename = {model_autosave_filename}\n")
else:
    print()
for epoch in range(N_Epochs):
    model.train()
    train_loss_list = []
    val_loss_list = []
    val_acc_list = []
    correct, total = 0, 0
    train_correct, train_total = 0, 0
    bef_time = time.time()
    
    # Training loop
    for idx, batch in tqdm(enumerate(train_loader), ascii='.>=', disable=True):
        X = batch
        X = X.to(device)
        model.zero_grad()
        out = model(X)
        loss = loss_fn(out, X)
        loss.backward()
        optimizer.step()
        #scheduler.step() # Scheduler step for batch
        train_loss_list.append(loss.detach().numpy())
        print(f'Epoch ({epoch+1}/{N_Epochs}) | Batch ({idx+1}/{len(train_loader)}) | train_loss = {np.mean(train_loss_list):.8f}', end='\r')
    print(f"Epoch ({epoch+1}/{N_Epochs}): train_loss = {np.mean(train_loss_list):0.6f}", end=' ')
    
    # Validation loop
    model.eval()
    with torch.no_grad():
        for val_idx, batch in enumerate(val_loader):
            X = batch 
            X = X.to(device)
            out = model(X)
            loss = loss_fn(out, X)
            val_acc = 1-abs(torch.mean((X-out)/X).detach().item())
            val_acc_list.append(val_acc)
            val_loss_list.append(loss.detach().numpy())
            # save images to disk to monitor progress
            if val_idx < 10:
                save_images(out, X, f"{val_images_save_dir}/val_{get_extended_num(epoch+1)}_{get_extended_num(val_idx)}.jpg")
                
    aft_time = time.time()
    print(f"| val_loss = {np.mean(val_loss_list):0.6f} | val_acc = {np.mean(val_acc_list):0.4f} | Time: {aft_time-bef_time:.1f}s", end = ' ')
    
    # Model autosave
    if epoch != 0:
        if AUTO_SAVE and min(epoch_val_loss_list) > np.mean(val_loss_list):
            torch.save(model, model_autosave_filename)
            print(f"| Model saved.")
        elif not AUTO_SAVE or min(epoch_val_loss_list) < np.mean(val_loss_list):
            print(f"| Model not saved.")
    else:
        if AUTO_SAVE:
            torch.save(model, model_autosave_filename)
            print(f"| Model saved.")
        else:
            print(f"| Model not saved.")
    # Save for metrics
    epoch_val_acc_list.append(np.mean(val_acc_list))
    epoch_val_loss_list.append(np.mean(val_loss_list))
    epoch_train_loss_list.append(np.mean(train_loss_list))
    epoch_learning_rate_list.append(optimizer.param_groups[0]['lr'])
    scheduler.step() # Scheduler step for epoch
print(f"Training complete at:")
os.system('date')


# Out of sample accuracy
#if AUTO_SAVE:
#    net = model
#    net = torch.load(model_autosave_filename)
#    print(f"Best saved model loaded.")
#else:
#    net = model
#correct, total = 0, 0
#net.eval()
#with torch.no_grad():
#    for batch in test_loader:
#        X, y = batch 
#        X = X.to(device)
#        y = y.to(device)
#        out = net(X)
#        if out.detach().numpy()[0][0] <= 0.5:
#            out_ = 0
#        else:
#            out_ = 1
#        if out_ == y.detach().numpy()[0]:
#            correct += 1
#            total += 1
#        else:
#            total += 1
#        loss = loss_fn(out.view(y.shape[0]).float(), y.float())
#print(f"\nOut of sample accuracy = {correct/total*100:.2f}")


# Plot metrics
# 1.Losses
plt.style.use(f"fivethirtyeight")
epochs_X = [i+1 for i in range(len(epoch_val_acc_list))]
plt.plot(epochs_X, epoch_val_loss_list, color='b', linewidth=1, label='val_loss')
plt.plot(epochs_X, epoch_train_loss_list, color='r', linewidth=1, label='train_loss')
loss_minima_at_epoch = epoch_val_loss_list.index(min(epoch_val_loss_list)) + 1
plt.axvline(x = loss_minima_at_epoch, color='m', linewidth=1, label='val_loss_minima')
plt.axhline(y = min(epoch_val_loss_list), color='m', linewidth=1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title(f"Losses")
plt.show()
# 2.Accuracy
plt.plot(epochs_X, epoch_val_acc_list, color='g', linewidth=1, label='val_acc')
acc_maxima_at_epoch = epoch_val_acc_list.index(max(epoch_val_acc_list)) + 1
plt.axvline(x = acc_maxima_at_epoch, color='m', linewidth=1, label='val_acc_maxima')
plt.axvline(x = loss_minima_at_epoch, color='b', linewidth=1, label='val_loss_minima')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title("Accuracy")
plt.show()
# 3.Learning rate
plt.plot(epochs_X, epoch_learning_rate_list, color='b', linewidth=1, label='learning_rate')
plt.legend()
plt.title(f"Learning rate")
plt.show()


# Save the model
if not AUTO_SAVE:
    save_model_flag = input("Save model [y/n]: ")
    if save_model_flag.lower() == 'y':
        save_file_name = f"{model.model_name}_{get_timestamp()}.pt"
        torch.save(model, save_file_name) 
        print(f"Model saved as {save_file_name}")
        # Rename the model
        renamed_save_filename = f"{save_file_name.replace('.pt', '')}" + \
                f"_valLoss{epoch_val_loss_list[-1]:.6f}_valAcc{epoch_val_acc_list[-1]*100:.2f}.pt"
        rename_flag = input(f"Rename the model to {renamed_save_filename}? [y/n]: ")
        if rename_flag.lower() == 'y':
            os.system(f"mv {save_file_name} {renamed_save_filename}")
        else:
            pass
    else:
        pass

elif AUTO_SAVE:
    print(f"Model saved as {model_autosave_filename}")
    # Rename the model
    val_loss_filename = min(epoch_val_loss_list)
    val_acc_filename = epoch_val_acc_list[epoch_val_loss_list.index(val_loss_filename)]*100
    renamed_save_filename = f"{model_autosave_filename.replace('.pt','')}" + \
            f"_valLoss{val_loss_filename:.6f}_valAcc{val_acc_filename:.2f}.pt"
    rename_flag = input(f"Rename the model to {renamed_save_filename}? [y/n]: ")
    if rename_flag.lower() == 'y':
        os.system(f"mv {model_autosave_filename} {renamed_save_filename}")
    else:
        pass

else:
    pass
