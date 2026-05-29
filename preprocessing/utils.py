# deep learning model helper
# functions for loading the training and validation datasets, defining the CNN model architecture,
# and training and evaluation loops for the active player prediction task.

import os
import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn

from tqdm.auto import tqdm

def crop_rotate_roi(image):
    '''
    Crop and rotate regions of interest for center card and players

    :param image: input image
    :type image: numpy.array
    :return: central card crop and list of player crops
    :rtype: tuple
    '''
    M, N, _ = image.shape
    
    # thirds 
    M13, M23 = M//3, 2*M//3
    N13, N23 = N//3, 2*N//3

    # fifths
    M15, M45 = M//5, 4*M//5
    N15, N45 = N//5, 4*N//5
    
    # cropping and rotate - dealer always on right side of cards
    central = image[M13:M23, N13:N23]
    
    player1 = image[M23:, N15:N45]
    player2 = np.rot90(image[:M45+1, N45:], -1)
    player3 = np.rot90(image[:M13+1, N15:N45], 2)
    player4 = np.rot90(image[M15:, :N15], 1)

    return central, [player1, player2, player3, player4]

def reshape_players(players):
    '''
    Reshape player image to be compatible with CNN input

    :param player: input player images
    :type player: numpy.array
    :return: reshaped player images
    :rtype: tuple
    '''
    pl1, pl2, pl3, pl4 = players

    shape2 = np.shape(pl2)
    shape3 = np.shape(pl3)
    shape4 = np.shape(pl4)

    pl2_new = np.zeros_like(pl1)
    pl3_new = np.zeros_like(pl1)
    pl4_new = np.zeros_like(pl1)

    pl2_new[:shape2[0], :shape2[1]] = pl2
    pl3_new[:shape3[0], :shape3[1]] = pl3
    pl4_new[:shape4[0], :shape4[1]] = pl4

    return pl1, pl2_new, pl3_new, pl4_new
         
def get_model_size(model):
    '''
    Get total model parameters and size in MB

    :param model: model to evaluate
    :type model: torch.nn.Module
    :return: print total number of parameters and model size in MB
    :raises ValueError: if the model has more than 12 million parameters
    '''
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = (param_size + buffer_size) / (1024 ** 2)

    total_p = sum(p.numel() for p in model.parameters())

    if total_p < 12e6: # assignment contraint
        print(f'All good! Number of parameters is {total_p:.3e}, below the 12e6 limit.\n\
Model size is {model_size:.2f} MB.') 
    else:
        raise ValueError(f'Model has {total_p:.3e} parameters, which exceeds the 12e6 limit.')

def move_targets_to_device(targets, device='cpu'):
    '''
    Move the target tensors (dictionary) to the specified device

    :param targets: dictionary of target tensors
    :type targets: dict
    :param device: device to move the tensors to, defaults to 'cpu'
    :type device: str, optional
    '''
    return {k: v.to(device) for k, v in targets.items()}

def compute_metrics(predictions, targets, threshold=0.5):
    '''
    Compute metrics for the given inputs, targets and predictions

    :param predictions: model predictions
    :type predictions: torch.Tensor
    :param targets: dataset labels
    :type targets: torch.Tensor
    :param threshold: classification threshold, defaults to 0.5
    :type threshold: float, optional
    :return: accuracy, precision, recall, f1_score
    :rtype: tuple
    '''
    probs = torch.sigmoid(predictions) # model outputs are logits, apply sigmoid to get probabilities
    predictions = (probs >= threshold).float()

    tp = ((predictions == 1) & (targets == 1)).sum().item()
    tn = ((predictions == 0) & (targets == 0)).sum().item()
    fp = ((predictions == 1) & (targets == 0)).sum().item()
    fn = ((predictions == 0) & (targets == 1)).sum().item()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

def train_ap(model, loss_f, train_loader, optimizer, device='cpu'):
    '''
    Train the model

    :param model: model to train
    :type model: torch.nn.Module
    :param loss_f: loss function
    :type loss_f: torch.nn.Module
    :param train_loader: training data loader
    :type train_loader: torch.utils.data.DataLoader
    :param optimizer: optimizer
    :type optimizer: torch.optim.Optimizer
    :param device: device to use, defaults to 'cpu'
    :type device: str
    :return: average training loss
    :rtype: float
    '''
    model.to(device)
    model.train()

    losses = []
    
    for inputs, targets in tqdm(train_loader, desc='Train batches', leave=False):
        inputs = inputs.to(device)
        targets = targets.to(device)  #move_targets_to_device(targets, device)
        
        optimizer.zero_grad(set_to_none=True)
        
        predictions = model(inputs)
        loss = loss_f(predictions, targets)

        loss.backward()
        optimizer.step()
        
        losses.append(loss.detach().cpu().item())

    losses = np.array(losses).mean()
    return losses

def eval_ap(model, loss_f, val_loader, device='cpu'):
    '''
    Evaluate the model

    :param model: model to evaluate
    :type model: torch.nn.Module
    :param loss_f: loss function
    :type loss_f: torch.nn.Module
    :param val_loader: validation data loader
    :type val_loader: torch.utils.data.DataLoader
    :param device: device to use, defaults to 'cpu'
    :type device: str, optional
    :return: average validation loss and metrics
    :rtype: tuple
    '''
    model.to(device)
    model.eval()

    losses = []
    accuracies, precisions, recalls, f1_scores = [], [], [], []

    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="Val batches", leave=False):
            inputs = inputs.to(device)
            targets = targets.to(device)  #move_targets_to_device(targets, device)

            predictions = model(inputs)
            loss = loss_f(predictions, targets)   

            losses.append(loss.detach().cpu().item())
            
            accuracy, precision, recall, f1_score = compute_metrics(predictions, targets)
            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1_score)

    losses = np.array(losses).mean()
    metrics = {
        'accuracy': np.array(accuracies).mean(),
        'precision': np.array(precisions).mean(),
        'recall': np.array(recalls).mean(),
        'f1_score': np.array(f1_scores).mean()
    }
    return losses, metrics

def predict_active_players(model, test_loader, device='cpu'):
    '''
    Predict the active player for a single image in the test set using the trained model.

    :param model: trained model to use for prediction
    :type model: torch.nn.Module
    :param test_loader: test data loader
    :type test_loader: torch.utils.data.DataLoader
    :param device: device to use, defaults to 'cpu'
    :type device: str, optional
    :return: lists of image IDs and predicted active players
    :rtype: tuple
    '''
    model.eval()

    image_ids_all = []
    active_players_all = []

    player_decoding = {
        0: 'p1',
        1: 'p2',
        2: 'p3',
        3: 'p4',
    }

    with torch.no_grad():
        for image_ids, players in test_loader:
            players = players.to(device)

            B, P, C, H, W = players.shape

            players_flat = players.view(B*P, C, H, W) # flatten the first and second dimensions to pass through the model

            logits = model(players_flat)              # [B*4, 1]
            probs = torch.sigmoid(logits)            # [B*4, 1]

            probs = probs.view(B, P)                 # [B, 4] - reshape back to separate the players

            pred_idx = torch.argmax(probs, dim=1)    # [B]

            pred_players = [player_decoding[i.item()] for i in pred_idx]

            image_ids_all.extend(image_ids)
            active_players_all.extend(pred_players)

    return image_ids_all, active_players_all

def plot_losses(train_losses, val_losses):
    '''
    Plot the training and validation losses. 

    :param train_losses: list of training losses for each epoch
    :type train_losses: list
    :param val_losses: list of validation losses for each epoch
    :type val_losses: list
    :return None:
    '''   
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_losses, color='navy', linewidth=2.5, ls='-', label='training')
    ax.plot(val_losses, color='crimson', linewidth=2.5, ls='--', label='validation')
    
    # identify min validation loss
    min_val_loss = np.min(val_losses)
    min_val_epoch = np.argmin(val_losses)
    ax.scatter(min_val_epoch, min_val_loss, marker='o',  color='green', label='min validation loss')
    ax.set_yscale('log')
    
    ax.set_title(f'Evolution of training and validation losses', fontsize=18)
    ax.set_xlabel('Epochs (-)', fontsize=16)
    ax.set_ylabel(f'Binary Cross Entropy loss (-)', fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    ax.legend()

    ax.set_xlim(0, len(train_losses)+1)
    
    plt.show()

def plot_metrics(metrics):
    '''
    Plot the training and validation metrics. 

    :param metrics: list of dictionaries containing the metrics for each epoch
    :type metrics: list
    '''   
    fig, ax = plt.subplots()

    colors = ['navy', 'crimson', 'green', 'orange']
    for i, metric in enumerate(metrics[0].keys()):
        metric_values = [m[metric] for m in metrics]
        ax.plot(metric_values, color=colors[i], linewidth=2.5, ls='-', label=metric)

    ax.set_title(f'Evolution of validation metrics', fontsize=18)
    ax.set_xlabel('Epochs (-)', fontsize=16)
    ax.set_ylabel(f'(-)', fontsize=16)
    ax.legend()
    
    plt.show()