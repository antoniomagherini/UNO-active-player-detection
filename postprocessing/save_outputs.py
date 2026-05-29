# deep learning model helper
# functions for saving various outputs during training and evaluation, 
# like model checkpoints, losses, metrics, test predictions, and normalization parameters for reproducibility.

import os
import json

import torch
import pandas as pd


def get_unique_filepath(filepath):
    '''
    If filepath already exists, append _1, _2, ... before the extension.

    :param filepath: name of the file to save
    :type filepath: str
    :return: unique filepath
    :rtype: str
    '''
    if os.path.exists(filepath):
        base, ext = os.path.splitext(filepath)
        count = 1

        while os.path.exists(f'{base}_{count}{ext}'):
            count += 1

        filepath = f'{base}_{count}{ext}'

    return filepath

def save_model(state_dict, optimizer, epoch, loss, init_hidden_channels, kernel_size, 
               n_down, batch_size, learning_rate):
    '''
    Save the model state dict, optimizer state dict, epoch, loss, and hyperparameters to a .pth file

    :param state_dict: model state dict
    :type state_dict: torch.nn.Module.state_dict
    :param optimizer: optimizer state dict
    :type optimizer: torch.optim.Optimizer.state_dict
    :param epoch: current epoch
    :type epoch: int
    :param loss: current loss
    :type loss: float
    :param init_hidden_channels: initial number of hidden channels
    :type init_hidden_channels: int
    :param kernel_size: size of the convolutional kernel
    :type kernel_size: int
    :param n_down: number of downsampling layers
    :type n_down: int
    :param batch_size: size of the training batch
    :type batch_size: int
    :param learning_rate: initial learning rate for the optimizer
    :type learning_rate: float
    :return: None
    :rtype: None
    '''
    folder = 'outputs/models'
    os.makedirs(folder, exist_ok=True)

    filename = f'best_model_epoch{epoch}_init{init_hidden_channels}_kernel{kernel_size}_ndown{n_down}_batch{batch_size}_lr{learning_rate}.pth'
    filepath = os.path.join(folder, filename)
    
    # avoid overwriting existing files
    filepath = get_unique_filepath(filepath)

    torch.save({
        'model_state_dict': state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss,
        'init_hidden_channels': init_hidden_channels,
        'kernel_size': kernel_size,
        'n_down': n_down,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }, filepath)

    return None

def save_losses(train_losses, val_losses, init_hidden_channels, 
                kernel_size, n_down, batch_size, learning_rate):
    '''
    Save the training and validation losses to a .csv file

    :param train_losses: list of training losses
    :type train_losses: list of float
    :param val_losses: list of validation losses
    :type val_losses: list of float
    :param init_hidden_channels: initial number of hidden channels
    :type init_hidden_channels: int
    :param kernel_size: size of the convolutional kernel
    :type kernel_size: int
    :param n_down: number of downsampling layers
    :type n_down: int
    :param batch_size: size of the training batch
    :type batch_size: int
    :param learning_rate: initial learning rate for the optimizer
    :type learning_rate: float
    '''
    folder = 'outputs/losses'
    os.makedirs(folder, exist_ok=True)

    save_losses = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    
    df = pd.DataFrame(save_losses)

    filename = f'losses_init{init_hidden_channels}_kernel{kernel_size}_ndown{n_down}_batch{batch_size}_lr{learning_rate}.csv'
    filepath = os.path.join(folder, filename)
    filepath = get_unique_filepath(filepath)

    df.to_csv(filepath, index=False)

    return None

def save_metrics(metrics, init_hidden_channels, kernel_size, n_down, batch_size, learning_rate):
    '''
    Save the training and validation metrics to a .csv file

    :param metrics: list of dictionaries containing the metrics for each epoch
    :type metrics: list of dict
    :param init_hidden_channels: initial number of hidden channels
    :type init_hidden_channels: int
    :param kernel_size: size of the convolutional kernel
    :type kernel_size: int
    :param n_down: number of downsampling layers
    :type n_down: int
    :param batch_size: size of the training batch
    :type batch_size: int
    :param learning_rate: initial learning rate for the optimizer
    :type learning_rate: float
    '''
    folder = 'outputs/metrics'
    os.makedirs(folder, exist_ok=True)

    save_metrics = {
        'accuracy': [metric['accuracy'] for metric in metrics],
        'precision': [metric['precision'] for metric in metrics],
        'recall': [metric['recall'] for metric in metrics],
        'f1-score': [metric['f1_score'] for metric in metrics],
    }

    df = pd.DataFrame(save_metrics)

    filename = f'metrics_init{init_hidden_channels}_kernel{kernel_size}_ndown{n_down}_batch{batch_size}_lr{learning_rate}.csv'
    filepath = os.path.join(folder, filename)
    filepath = get_unique_filepath(filepath)

    df.to_csv(filepath, index=False)

    return None

def save_test_predictions(image_ids, active_players, init_hidden_channels, kernel_size, 
                          n_down, batch_size, learning_rate):
    '''
    Save the test active-player predictions to a .csv file

    :param image_ids: list of image ids
    :type image_ids: list of str
    :param active_players: list of predicted active players
    :type active_players: list of int
    :param init_hidden_channels: initial number of hidden channels
    :type init_hidden_channels: int
    :param kernel_size: size of the convolutional kernel
    :type kernel_size: int
    :param n_down: number of downsampling layers
    :type n_down: int
    :param batch_size: size of the training batch
    :type batch_size: int
    :param learning_rate: initial learning rate for the optimizer
    :type learning_rate: float
    '''
    folder = 'outputs/test_predictions'
    os.makedirs(folder, exist_ok=True)

    submission = pd.DataFrame({
        'image_id': image_ids,
        'active_player': active_players
    })

    filename = f'test_ap_init{init_hidden_channels}_kernel{kernel_size}_ndown{n_down}_batch{batch_size}_lr{learning_rate}.csv'
    filepath = os.path.join(folder, filename)
    filepath = get_unique_filepath(filepath)

    submission.to_csv(filepath, index=False)

    return submission

def save_normalization(mean, std, init_hidden_channels, kernel_size, n_down, batch_size, learning_rate, augmentation=False):
    '''
    Save normalization mean and std to a .json file for reproducibility

    :param mean: mean of the training dataset
    :type mean: list or np.ndarray
    :param std: standard deviation of the training dataset
    :type std: list or np.ndarray
    :param init_hidden_channels: initial number of hidden channels
    :type init_hidden_channels: int
    :param kernel_size: size of the convolutional kernel
    :type kernel_size: int
    :param n_down: number of downsampling layers
    :type n_down: int
    :param batch_size: size of the training batch
    :type batch_size: int
    :param learning_rate: initial learning rate for the optimizer
    :type learning_rate: float
    :param augmentation: whether data augmentation was used during training, deafaults to False
    :type augmentation: bool, optional
    '''
    folder = 'outputs/normalization'
    os.makedirs(folder, exist_ok=True)

    normalization = {
        'mean': mean.tolist() if hasattr(mean, 'tolist') else mean,
        'std': std.tolist() if hasattr(std, 'tolist') else std
    }

    filename = f'norm_init{init_hidden_channels}_kernel{kernel_size}_ndown{n_down}_batch{batch_size}_lr{learning_rate}_aug{augmentation}.json'
    filepath = os.path.join(folder, filename)
    filepath = get_unique_filepath(filepath)

    with open(filepath, 'w') as f:
        json.dump(normalization, f, indent=4)

    return filepath