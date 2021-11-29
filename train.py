import os 
import time 
import torch 
import torch.nn as nn 
import torch.optim as optim 
import torchvision
from torchvision import transforms 
import numpy as np
from models import Encoder, Decoder
from dataset import FaceData
from datetime import datetime
print(f"Imports complete")

BATCH_SIZE = 8
N_EPOCHS = 30


# Some helper functions
get_timestamp = lambda : datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def save_image_grid(images, columns, save_path):
    '''
    Saves images as grid
    '''
    grid = torchvision.utils.make_grid(images, nrow=columns, padding=4)
    torchvision.utils.save_image(grid, save_path)


# initialize log dirs
RUN_TIMESTAMP = get_timestamp()
logs_path = f"./logs/RUN_{RUN_TIMESTAMP}"
save_models_path = f"{logs_path}/saved_models"
gen_images_path = f"{logs_path}/gen_images"
metrics_log_path = f"{logs_path}/metrics.csv"
os.system(f"mkdir -p {logs_path}")
os.system(f"mkdir -p {save_models_path}")
os.system(f"mkdir -p {gen_images_path}")
os.system(f"touch {metrics_log_path}")
print(f"Saving logs to {logs_path}")


# Load datasets
my_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    #transforms.RandomRotation(45),
])
person1_data = FaceData("./Data/Person1", 512, transform=my_transforms)
person2_data = FaceData("./Data/Person2", 512, transform=my_transforms)
dataloader_p1 = torch.utils.data.DataLoader(person1_data, batch_size=BATCH_SIZE, shuffle=True)
dataloader_p2 = torch.utils.data.DataLoader(person2_data, batch_size=BATCH_SIZE, shuffle=True)
print(f"Data loaded.")


# Initialize models and hyperparams
BASE_LR = 0.003
LOAD_TRAINED_MODELS = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')

# Load models
encoder = Encoder().to(device)
decoder1 = Decoder().to(device)
decoder2 = Decoder().to(device)
if LOAD_TRAINED_MODELS:
    encoder = Encoder()
    decoder1 = Decoder()
    decoder2 = Decoder()
    encoder = torch.load('./encoder*')
    decoder1 = torch.load('./decoder1*')
    decoder2 = torch.load('./decoder2*')
    encoder = encoder.to(device)
    decoder1 = decoder1.to(device)
    decoder2 = decoder2.to(device)
else:
    encoder = Encoder().to(device)
    decoder1 = Decoder().to(device)
    decoder2 = Decoder().to(device)
# Optimizers
optim_enc = optim.Adam(encoder.parameters(), lr=BASE_LR)
optim_dec1 = optim.Adam(decoder1.parameters(), lr=BASE_LR)
optim_dec2 = optim.Adam(decoder2.parameters(), lr=BASE_LR)
loss_fn = nn.L1Loss()
print(f"Running on {device}\n")


# Metrics
train_loss_p1 = []
train_loss_p2 = []
test_images_dir_path = './test_images' # None if you dont wanna test
if test_images_dir_path:
    test_transforms = transforms.ToTensor()
    test_data = FaceData(test_images_dir_path, 8, test_transforms)
    test_loader = torch.utils.data.DataLoader(test_data, shuffle=False)
metrics_log = open(metrics_log_path, 'w')
metrics_log.write("EPOCHS,BATCHES,LOSSES_P1,LOSSES_P2\n")
metrics_log.close()


# Train models
for epoch_idx in range(N_EPOCHS):
    epoch_clock_bef = time.time()
    batch_losses_p1, batch_losses_p2 = [], []
    log = ''
    for batch_idx, data in enumerate(zip(dataloader_p1, dataloader_p2)):
        # optim zero grad
        optim_enc.zero_grad()
        optim_dec1.zero_grad()
        optim_dec2.zero_grad()
        # load images
        img1, img2 = data
        img1 = img1.to(device)
        img2 = img2.to(device)
        # Train person1 networks
        out1 = decoder1(encoder(img1))
        loss1 = loss_fn(out1, img1)
        loss1.backward()
        optim_enc.step()
        optim_dec1.step()
        # Train person2 networks
        out2 = decoder2(encoder(img2))
        loss2 = loss_fn(out2, img2)
        loss2.backward()
        optim_enc.step()
        optim_dec2.step()
        # batch metrics
        batch_losses_p1.append(loss1.item())
        batch_losses_p2.append(loss2.item())
        avg_loss_p1, avg_loss_p2 = np.average(batch_losses_p1), np.average(batch_losses_p2)
        log = log + f"{epoch_idx+1},{batch_idx+1},{loss1.item()},{loss2.item()}\n"
        print(f"Epoch: ({epoch_idx+1}/{N_EPOCHS}) Batch: {batch_idx+1}", end=' ')
        print(f"[loss_p1: {avg_loss_p1:.6f} | loss_p2: {avg_loss_p2:.6f}]", end='\r')

    # save models per epoch
    encoder = encoder.to(cpu)
    torch.save(encoder, f'{save_models_path}/epoch_{epoch_idx+1}_encoder.pt')
    encoder = encoder.to(device)
    decoder1 = decoder1.to(cpu)
    torch.save(decoder1, f'{save_models_path}/epoch_{epoch_idx+1}_decoder1.pt')
    decoder1 = decoder1.to(device)
    decoder2 = decoder2.to(cpu)
    torch.save(decoder2, f'{save_models_path}/epoch_{epoch_idx+1}_decoder2.pt')
    decoder2 = decoder2.to(device)
    
    # epoch metrics
    train_loss_p1.append(avg_loss_p1)
    train_loss_p2.append(avg_loss_p2)
    with open(metrics_log_path, 'a') as write_log:
        write_log.write(log)

    # save test images
    if test_images_dir_path:
        images = []
        for img in test_loader:
            img = img.to(device)
            images.append(img.to(cpu)[0]) # truth
            images.append(decoder1(encoder(img)).to(cpu)[0]) # D1(E(x))
            images.append(decoder2(encoder(img)).to(cpu)[0]) # D2(E(x))
        save_image_grid(images, 3, f"{gen_images_path}/epoch_{epoch_idx+1}_test.jpg")
        
    elapsed_time = (time.time()-epoch_clock_bef)/60 # Minutes
    print(f"Epoch: {epoch_idx+1} | loss_p1: {avg_loss_p1:.8f} | loss_p2: {avg_loss_p2:.8f}", end=' ')
    print(f"| time elapsed: {elapsed_time:.2f} Minutes. | Models saved.")
