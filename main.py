# main module for simulations

# import packages and modules
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # avoid issues with pytorch

import torch
import gc

import pandas as pd 

from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from torch.utils.data import DataLoader, ConcatDataset
from tqdm.auto import tqdm

from preprocessing.utils import *
from preprocessing.dataset import *
from model.CNN import *
from postprocessing.save_outputs import *
from postprocessing.report import *

# set the device where operations are performed
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("CUDA Device Count: ", torch.cuda.device_count())
    print("CUDA Device Name: ", torch.cuda.get_device_name(0))
else:
    device = 'cpu'
    
print(f'Using device: {device}')

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# load training .csv file, splitting into training and validation sets
# avoiding that the same image appears in different sets 

path = r'data'

df = pd.read_csv(os.path.join(path, 'train.csv'), sep=',', header=0)

split = 0.85
random_state = 42

df_train = df.sample(frac=split, random_state=random_state)
df_val = df.drop(df_train.index)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

# create training dataset and dataloader to compute the statistics for normalization
batch_size = 16
num_workers = 0
pin_memory = False

# downsize image dimensions
resize = 200

# first transform - to extract statistics before normalization
transform_stats = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize),
    transforms.ToTensor()
])

# create dataset and dataloader to compute the statistics for normalization
train_dataset_stats = DealerDataset(path, df=df_train, transform=transform_stats)
train_loader_stats = DataLoader(train_dataset_stats, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=pin_memory)

mean = torch.zeros(3)
std = torch.zeros(3)
n_images = 0

for x, y in train_loader_stats:
    batch_size_current = x.size(0)

    mean += x.mean(dim=[0, 2, 3]) * batch_size_current
    std += x.std(dim=[0, 2, 3]) * batch_size_current

    n_images += batch_size_current

mean /= n_images
std /= n_images

# second transform with normalization - original dataset
transform_player = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

# augmented transform to double the dataset size
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.20,
        hue=0.03
    ),

    transforms.RandomAffine(
        degrees=5,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
        shear=3
    ),

    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])

# create training and validation datasets
train_dataset_original = DealerDataset(path, df=df_train, transform=transform_player)
train_dataset_augmented = DealerDataset(path, df=df_train, transform=transform_train)
train_dataset = ConcatDataset([
    train_dataset_original,
    train_dataset_augmented
])

val_dataset = DealerDataset(path, df=df_val, transform=transform_player)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          num_workers=num_workers, pin_memory=pin_memory)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        num_workers=num_workers, pin_memory=pin_memory)

# model architecture hyperparameters
init_hidden_channels = 4
kernel_size = 3
padding = 1
n_down = 6

# load model and loss function
model = UNOplayer(
    init_hidden_channels=init_hidden_channels, 
    kernel_size=kernel_size, 
    padding=padding, 
    n_down=n_down
    ).to(device)

# assign larger weight to the positive class (active player) to address class imbalance
loss_f = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))

# check that model size is below 12e6 parameters
get_model_size(model) 

# training hyperparameters
learning_rate = 1e-2
num_epochs = 300

# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# scheduler for decreasing the learning rate
# every (step_size) epochs with given factor (gamma)
step_size = 5     # set to None to remove the scheduler
gamma = 0.75      # set to None to remove the scheduler

# create the learning rate scheduler if step_size and gamma are defined
scheduler = None
if step_size is not None and gamma is not None:
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

# save normalization mean and std for later use in testing
save_normalization(mean, std, init_hidden_channels, kernel_size, 
                   n_down, batch_size, learning_rate, augmentation=True)

# initialize training, validation losses and metrics
train_losses, val_losses = [], []
metrics = []

# initial values
best_loss = float('inf')
best_model_state = None
count = 0
best_epoch = 0

early_stop = 15 # stop training if the validation loss does not improve for (early_stop) epochs

# set to True to run the training loop, False to skip training and load existing trained model and losses
train_loop = False

if train_loop:
    pbar = tqdm(range(1, num_epochs + 1), desc="Epochs")
    
    # start training loop
    for epoch in pbar:
        
        train_loss = train_ap(model, loss_f, train_loader, optimizer, device=device)
        val_loss, val_metrics = eval_ap(model, loss_f, val_loader, device=device)

        train_losses.append(train_loss)
        val_losses.append(val_loss) 
        metrics.append(val_metrics)
        
        # check for improvement in validation loss and save the best model state
        if val_loss <= best_loss:
            best_loss = val_loss
            best_model_state = deepcopy(model.state_dict())
            best_epoch = epoch
            count = 0
        else:
            count += 1

        pbar.set_postfix({
            't_loss': f'{train_loss:.4f}',
            'v_loss': f'{val_loss:.4f}',
            'v_f1': f'{val_metrics["f1_score"]:.4f}',
            'best': f'{best_loss:.4f}',
            'no_impr': count
        })

        if count >= early_stop:
            print(f'Early stopping at epoch {epoch}. Best epoch was {best_epoch} with val loss {best_loss:.4f}.')
            break

        # update learning rate
        if scheduler is not None:
            scheduler.step()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # save best model state - losses - metrics - 
    save_model(best_model_state, optimizer, epoch, best_loss, init_hidden_channels, 
            kernel_size, n_down, batch_size, learning_rate)

    save_losses(train_losses, val_losses, init_hidden_channels, 
                kernel_size, n_down, batch_size, learning_rate)

    save_metrics(metrics, init_hidden_channels, kernel_size, 
                n_down, batch_size, learning_rate)

# start testing
# create test dataset and dataloader with the same normalization as training and validation
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())
])
    
# create test dataset and dataloader
test_dataset = DealerTestDataset(r'src\data', transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, 
                         num_workers=num_workers, pin_memory=pin_memory)

# best model after augmentation
model_path = 'outputs/models/best_model_epoch70_init4_kernel3_ndown6_batch16_lr0.01.pth'
model = load_model(model, model_path=model_path, device=device)

# predict active players on the test set
# predict the probability of each crop to be active player 
# then assign the active player as the one with the highest probability for each image
image_ids, active_players = predict_active_players(model, test_loader, device=device)

# store the test predictions in a specific format for submission
submission = pd.DataFrame({'image_id': image_ids, 'active_player': active_players})

# save the test predictions in a .csv file
test_predictions = save_test_predictions(image_ids, active_players, init_hidden_channels, kernel_size,
                                         n_down, batch_size, learning_rate)