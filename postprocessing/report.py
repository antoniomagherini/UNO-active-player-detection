# deep learning model helper
# functions for visualizing the data preprocessing steps, loading the trained model,
# and plotting the test predictions and Grad-CAM heatmaps for model explainability.

import torch
import skimage.io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms

from src.utils_dl import *

def plot_cropping(path='src/data/train_images/L1000772.jpg'):
    '''
    Identify the cropping regions for the 4 players in the image and plot them.

    :param path: path to image file, defaults to 'src/data/train_images/L1000772.jpg'
    :type path: str, optional
    :return: image array
    :rtype: numpy.ndarray
    '''
    
    img = skimage.io.imread(path)
    
    # compute the coordinates of the cropping regions
    M, N, C = np.shape(img)

    M13, M23 = M//3, 2*M//3
    M15, M45 = M//5, 4*M//5
    N15, N45 = N//5, 4*N//5

    # plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(img, cmap='gray')

    lw = 3

    # player 1
    ax.hlines([M23, M], N15, N45, color='b', ls='--', lw=lw, label='Player 1')
    ax.vlines([N15, N45], M23, M, color='b', ls='--', lw=lw)

    # player 2
    ax.hlines([0, M45], N45, N, color='k', ls='--', lw=lw, label='Player 2')
    ax.vlines([N45, N], 0, M45, color='k', ls='--', lw=lw)

    # player 3
    ax.hlines([0, M13], N15, N45, color='g', ls='--', lw=lw, label='Player 3')
    ax.vlines([N15, N45], 0, M13, color='g', ls='--', lw=lw)

    # player 4
    ax.hlines([M15, M], 0, N15, color='m', ls='--', lw=lw, label='Player 4')
    ax.vlines([0, N15], M15, M, color='m', ls='--', lw=lw)

    ax.set_xlim(0, N)
    ax.set_ylim(M, 0)

    ax.set_title('Cropping regions for identifying the 4 players', 
                 loc = 'center', fontsize=16)
    plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.2), ncols=4)
    plt.show()

    return img

def plot_preprocessing(players, player1, player2, player3, player4):
    '''
    Plot the original crops of the 4 players and the crops after preprocessing (reshaping and padding).

    :param players: list of original crops of the 4 players
    :type players: list of numpy.ndarray
    :param player1: preprocessed crop of player 1
    :type player1: numpy.ndarray
    :param player2: preprocessed crop of player 2
    :type player2: numpy.ndarray
    :param player3: preprocessed crop of player 3
    :type player3: numpy.ndarray
    :param player4: preprocessed crop of player 4
    :type player4: numpy.ndarray
    :return: None
    :rtype: None
    '''

    print(f'Shapes before padding: p1 = {np.shape(players[0])}, \
    p2 = {np.shape(players[1])}')
    print(f'Shapes before padding: p1 = {np.shape(player1)}, \
    p2 = {np.shape(player2)}')

    fig, axs = plt.subplots(2, 4, figsize=(12, 5))

    axs[0, 0].imshow(players[0], cmap='gray')
    axs[0, 0].set_title('Player 1', fontsize=12)

    axs[0, 1].imshow(players[1], cmap='gray')
    axs[0, 1].set_title('Player 2', fontsize=12)

    axs[0, 2].imshow(players[2], cmap='gray')
    axs[0, 2].set_title('Player 3', fontsize=12)

    axs[0, 3].imshow(players[3], cmap='gray')
    axs[0, 3].set_title('Player 4', fontsize=12)

    axs[1, 0].imshow(player1, cmap='gray')
    axs[1, 0].set_title('Player 1', fontsize=12)
    axs[1, 1].imshow(player2, cmap='gray')
    axs[1, 1].set_title('Player 2', fontsize=12)
    axs[1, 2].imshow(player3, cmap='gray')
    axs[1, 2].set_title('Player 3', fontsize=12)
    axs[1, 3].imshow(player4, cmap='gray')
    axs[1, 3].set_title('Player 4', fontsize=12)

    fig.text(0.5, 0.94, 'Original crops - before preprocessing',
        ha='center', va='center', fontsize=16)

    fig.text(0.5, 0.48, 'Crops after preprocessing',
        ha='center', va='center', fontsize=16)

    plt.show()
    return None

# parameters for computing the statistics of the training dataset for normalization
path = 'src/data'
resize = 200
batch_size = 16
num_workers = 0
pin_memory = False

def train_dataset_statistics(path=path, resize=resize):
    '''
    Load training dataset and compute the stastistics for normalization

    :param path: path to data folder, defaults to path
    :type path: str, optional
    :param resize: resize dimension for the images, defaults to resize
    :type resize: int, optional
    :return: tuple of mean and std for normalization, and the training and validation dataframes
    :rtype: tuple
    '''
    # split the csv file into training and validation sets
    # avoiding that the same image appears in different sets 

    df = pd.read_csv(os.path.join(path, 'train.csv'), sep=',', header=0)

    split = 0.85
    random_state = 42

    df_train = df.sample(frac=split, random_state=random_state)
    df_val = df.drop(df_train.index)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)

    print(f'Number of training images: {len(df_train)}, number of validation images: {len(df_val)}')
    print(f'Number of training samples: {4*len(df_train)}, number of validation samples: {4*len(df_val)}\n')

    # first transform - before normalization
    transform_stats = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

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

    print('Training dataset mean for normalization:', mean)
    print('Training dataset std for normalization:', std)

    return mean, std, df_train, df_val

def augmentation(mean, std, df_train, df_val, path=path, resize=resize):
    '''
    Apply data augmentation to the training dataset by doubling the original training samples.
    Transformations include color jitter, random affine and gaussian blur.

    :param mean: mean for normalization
    :type mean: torch.Tensor
    :param std: standard deviation for normalization
    :type std: torch.Tensor
    :param df_train: training dataframe
    :type df_train: pd.DataFrame
    :param df_val: validation dataframe
    :type df_val: pd.DataFrame
    :param path: path to data folder, defaults to path
    :type path: str, optional
    :param resize: resize dimension for the images, defaults to resize
    :type resize: int, optional
    :return: tuple of augmented training and validation datasets
    :rtype: tuple
    '''
    # second transform with normalization - original dataset
    transform_player = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    # augmented transform
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(resize),
        
        # new transformations
        transforms.ColorJitter(brightness=0.25, contrast=0.25, 
                               saturation=0.20, hue=0.03),
        transforms.RandomAffine(degrees=5, translate=(0.03, 0.03),
            scale=(0.95, 1.05), shear=3),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),

        transforms.ToTensor(),
        transforms.Normalize(mean=mean.tolist(), std=std.tolist())
    ])

    train_dataset_original = DealerDataset(path, df=df_train, transform=transform_player)
    train_dataset_augmented = DealerDataset(path, df=df_train, transform=transform_train)

    train_dataset = ConcatDataset([
        train_dataset_original,
        train_dataset_augmented
    ])

    val_dataset = DealerDataset(path, df=df_val, transform=transform_player)

    print(f'Original training samples: {len(train_dataset_original)}')
    print(f'Augmented training samples: {len(train_dataset_augmented)}')
    print(f'Total training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}\n')

    print(f'Example image shape: {train_dataset[0][0].shape}\t')
    print(f'Example image pixel values: {train_dataset[0][0]}\n')
    print(f'Example image class: {train_dataset[0][1]}')

    return train_dataset, val_dataset

def unnormalize(img, mean, std):
    '''
    Unnormalize images before plotting them to avoid clipping warnings.

    :param img: image to unnormalize
    :type img: torch.Tensor
    :param mean: mean for normalization
    :type mean: torch.Tensor
    :param std: standard deviation for normalization
    :type std: torch.Tensor
    :return: unnormalized image
    :rtype: torch.Tensor
    '''
    img = img.detach().cpu().clone()

    mean_t = torch.as_tensor(mean).view(-1, 1, 1)
    std_t = torch.as_tensor(std).view(-1, 1, 1)

    img = img * std_t + mean_t
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    # mean = torch.as_tensor(mean, dtype=img.dtype, device=img.device).view(3, 1, 1)
    # std = torch.as_tensor(std, dtype=img.dtype, device=img.device).view(3, 1, 1)

    return img

def plot_original_augmented(train_dataset, mean, std, idx=22):
    '''
    Plot original and augmented images from training dataset.

    :param train_dataset: training dataset containing original and augmented images
    :type train_dataset: torch.utils.data.Dataset
    :param mean: mean for normalization
    :type mean: torch.Tensor
    :param std: standard deviation for normalization
    :type std: torch.Tensor
    :param idx: index of the image to plot, defaults to 22
    :type idx: int, optional
    '''
    
    cropped_original = unnormalize(train_dataset[idx][0], mean, std)
    cropped_augmented = unnormalize(train_dataset[idx + 276][0], mean, std)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(cropped_original)
    ax[0].set_title(f'Original image', fontsize=14)

    ax[1].imshow(cropped_augmented)
    ax[1].set_title(f'Augmented image', fontsize=14)

    plt.show()
    return None

def load_test_dataset(mean, std, path=path, resize=resize):
    '''
    Load test dataset with the same transformations as the training dataset.
    
    :param mean: mean for normalization
    :type mean: torch.Tensor
    :param std: standard deviation for normalization
    :type std: torch.Tensor
    :param df_test: dataframe containing test dataset information
    :type df_test: pandas.DataFrame
    :return: test dataset
    :rtype: torch.utils.data.Dataset
    '''
    transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(resize),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean.tolist(), std=std.tolist())])
    
    test_dataset = DealerTestDataset(path=path, transform=transform_test)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=pin_memory)
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    idx1 = 56
    idx2 = 101
    ax[0].imshow(unnormalize(test_dataset[idx1][1][0], mean, std))
    ax[0].set_title(f'Test image {idx1}', fontsize=14)
    ax[1].imshow(unnormalize(test_dataset[idx2][1][0], mean, std))
    ax[1].set_title(f'Test image {idx2}', fontsize=14)
    plt.show()
    return test_dataset, test_loader

def load_model(model, model_path='src/outputs/models/best_model_epoch70_init4_kernel3_ndown6_batch16_lr0.01.pth', device='cpu'):
    '''
    Load the trained model checkpoints from the specified path and set it to evaluation mode.

    :param model: model architecture to load the weights into
    :type model: UNOplayer model
    :param model_path: path to the model checkpoint, defaults to 'outputs/models/best_model_epoch70_init4_kernel3_ndown6_batch16_lr0.01.pth'
    :type model_path: str, optional
    :param device: device to load the model on, defaults to 'cpu'
    :type device: str, optional
    :return: loaded model
    :rtype: UNOplayer model
    '''
    checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval() # set the model to evaluation mode

    # print the keys in the checkpoint to see what is available
    print(f'Checkpoint keys: {checkpoint.keys()}')

    print(f'epoch: {checkpoint["epoch"]}')
    print(f'loss: {checkpoint["loss"]:.3e}')
    print(f'init_hidden_channels: {checkpoint["init_hidden_channels"]}')
    print(f'kernel_size: {checkpoint["kernel_size"]}')
    print(f'n_down: {checkpoint["n_down"]}')
    print(f'learning_rate: {checkpoint["learning_rate"]}')
    print(f'batch_size: {checkpoint["batch_size"]}')

    return model

def plot_test_predictions(model_predictions, path_images='src/data/test_images', path_labels='src/data/test.csv'):
    '''
    Plot random test images with their predicted and true labels.

    :param model_predictions: predicted active players for the test dataset
    :type model_predictions: list or array-like
    :param path_images: path to the directory containing test images, defaults to 'src/data/test_images'
    :type path_images: str, optional
    :param path_labels: path to the .csv file containing test labels, defaults to 'src/data/test.csv'
    :type path_labels: str, optional
    '''
    test_labels = pd.read_csv(path_labels)['active_player']

    test_images = sorted(os.listdir(path_images))
    
    nrows, ncols = 4, 5
    fig, ax = plt.subplots(nrows, ncols, figsize=(20, 12), sharex=True, sharey=True)
    
    # pick random images within the dataset
    img_id = np.random.randint(0, len(test_labels) - 1, nrows * ncols).astype(int)
    
    ax = ax.flatten()
    for i, img in enumerate(img_id):
        name = test_images[img]
        img_g = skimage.io.imread(os.path.join(path_images, name))
        ax[i].imshow(img_g)
        ax[i].set_title(f"Image {name.split('.')[0]}\nPred: {model_predictions[img]}, True: {test_labels[img]}", fontsize=12)
        ax[i].axis('off')

    plt.show()
    return None

def compute_gradcam_for_player(model, player_tensor, activations, gradients, target_label=1, device='cpu'):
    '''
    Compute Grad-CAM for a given player image.

    :param model: trained model to compute Grad-CAM for
    :type model: UNOplayer model
    :param player_tensor: tensor representing the player image, shape [C, H, W]
    :type player_tensor: torch.Tensor
    :param activations: dictionary to store activation values
    :type activations: dict
    :param gradients: dictionary to store gradient values
    :type gradients: dict
    :param target_label: target class label for Grad-CAM computation, defaults to 1
    :type target_label: int, optional
    :param device: device to perform computation on, defaults to 'cpu'
    :type device: str, optional
    :return: Grad-CAM heatmap, prediction probability, and logits
    :rtype: tuple
    '''
    x = player_tensor.unsqueeze(0).to(device)

    model.zero_grad(set_to_none=True)

    logit = model(x) # shape: [1, 1]
    prob = torch.sigmoid(logit)[0, 0]

    # For binary classification:
    # logit supports class 1.
    # -logit supports class 0.
    if target_label == 1:
        score = logit[0, 0]
    else:
        score = -logit[0, 0]

    score.backward()

    acts = activations["value"]      # [1, C, H, W]
    grads = gradients["value"]       # [1, C, H, W]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1, keepdim=True)
    cam = F.relu(cam)

    cam_min, cam_max = cam.min(), cam.max()
    if cam_max > cam_min:
        cam = (cam - cam_min) / (cam_max - cam_min)
    else:
        cam = torch.zeros_like(cam)

    cam = F.interpolate(
        cam,
        size=player_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()
    prob = prob.detach().cpu().item()
    logit = logit.detach().cpu().item()

    return cam, prob, logit

def plot_explanation_heatmap(model, test_dataset, mean, std, model_layer=(-1, -1), image_index=82, device='cpu'):
    '''
    Plot original player crops and Grad-CAM heatmaps for both binary classes.

    :param model: trained model to compute Grad-CAM for
    :type model: UNOplayer model
    :param test_dataset: dataset containing test images and labels
    :type test_dataset: torch.utils.data.Dataset
    :param mean: mean for normalization
    :type mean: torch.Tensor    
    :param std: standard deviation for normalization
    :type std: torch.Tensor
    :param model_layer: tuple specifying the layer to compute Grad-CAM on, defaults to (-1, -1) which corresponds to the last convolutional layer
    :type model_layer: tuple, optional
    :param image_index: index of the test image to analyze, defaults to 82
    :type image_index: int, optional
    :param device: device to perform computation on, defaults to 'cpu'
    :type device: str, optional
    :return: None
    :rtype: None
    '''
    model.eval()

    # (-1, -1) is the last convolutional layer before the final classifier
    target_layer = model.features[model_layer[0]].conv[model_layer[1]]

    activations = {}
    gradients = {}

    def forward_hook(module, input, output):
        activations['value'] = output

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # get one test image and its 4 player crops
    image_id, players = test_dataset[image_index]   # players shape: [4, C, H, W]

    player_names = ['p1', 'p2', 'p3', 'p4']

    cams_class0 = []
    cams_class1 = []
    probs = []
    logits = []

    for p in range(4):
        # compute Grad-CAM for both classes (active vs non-active) for each player crop
        cam0, prob0, logit0 = compute_gradcam_for_player(model, players[p], activations, gradients, 
                                                         target_label=0, device=device)
        cam1, prob1, logit1 = compute_gradcam_for_player(model, players[p], activations, gradients, 
                                                         target_label=1, device=device)

        cams_class0.append(cam0)
        cams_class1.append(cam1)

        # prob/logit are the same model prediction, regardless of target_label
        probs.append(prob1)
        logits.append(logit1)

    predicted_active_player = player_names[int(np.argmax(probs))]

    fig, ax = plt.subplots(3, 4, figsize=(12, 6))

    for p in range(4):
        img = unnormalize(players[p], mean, std)

        # in case img is a torch tensor
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu()
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.permute(1, 2, 0)
            img = img.clamp(0, 1)

        # row 1: original crop
        ax[0, p].imshow(img)
        ax[0, p].set_title(f'{player_names[p]} crop\nP(active) = {probs[p]:.3f}', fontsize=11)
        ax[0, p].axis('off')

        # row 2: Grad-CAM for class 0
        ax[1, p].imshow(img)
        ax[1, p].imshow(cams_class0[p], cmap='jet', alpha=0.45)
        ax[1, p].set_title('Grad-CAM class 0', fontsize=11)
        ax[1, p].axis('off')

        # row 3: Grad-CAM for class 1
        ax[2, p].imshow(img)
        ax[2, p].imshow(cams_class1[p], cmap='jet', alpha=0.45)
        ax[2, p].set_title('Grad-CAM class 1', fontsize=11)
        ax[2, p].axis('off')

    plt.suptitle(f'Image {image_id} | Predicted active player: {predicted_active_player}', fontsize=16)

    plt.tight_layout()
    plt.show()

    # remove hooks to avoid accumulating them if you rerun the function
    forward_handle.remove()
    backward_handle.remove()

    return None