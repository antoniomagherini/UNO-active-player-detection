# dataset classes for loading the training, validation and test datasets of the active player prediction task

import os
import torch
import skimage.io

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset

from preprocessing.utils import crop_rotate_roi, reshape_players

class DealerDataset(Dataset):
    '''
    Class for loading the training and validation datasets of the active player prediction task

    :param Dataset: dataset class
    :type Dataset: torch.utils.data.Dataset
    '''
    def __init__(self, path, df=None, transform=None):
        '''
        Load dataframes and set paths for images and labels

        :param path: path to the dataset groundtruth, containing 'train_images' and 'train.csv'
        :type path: str
        :param df: dataframe containing labels, defaults to None
        :type df: pd.DataFrame, optional
        :param transform: transform to apply to the images, defaults to None
        :type transform: torchvision.transforms, optional
        '''
        self.path = path
        self.path_images = os.path.join(path, 'train_images')
        self.path_labels = os.path.join(path, 'train.csv')

        if df is None:
            self.df = pd.read_csv(self.path_labels, sep=',', header=0)
        else:
            self.df = df.reset_index(drop=True)

        self.transform = transform

        self.player_encoding = {
            'p1': 0,
            'p2': 1,
            'p3': 2,
            'p4': 3,
        }

    def __len__(self):
        return 4 * len(self.df)

    def __getitem__(self, idx):
        '''
        Get the player crop and target for the given index

        :param idx: index of the player crop to retrieve
        :type idx: int
        :return: player crop and target
        :rtype: tuple
        '''
        row_idx = idx // 4
        player_idx = idx % 4

        row = self.df.iloc[row_idx]

        image_id = row['image_id']
        active_player = row['active_player']

        image_path = os.path.join(self.path_images, image_id + '.jpg')
        image = skimage.io.imread(image_path)

        _, players = crop_rotate_roi(image)
        players = reshape_players(players)

        player = players[player_idx]
        player = np.ascontiguousarray(player)

        if self.transform is not None:
            player = self.transform(player)

        target = 1 if player_idx == self.player_encoding[active_player] else 0
        target = torch.tensor([target], dtype=torch.float32)

        return player, target
    
class DealerTestDataset(Dataset):
    '''
    Class for loading the test dataset of the active player prediction task

    :param Dataset: dataset class
    :type Dataset: torch.utils.data.Dataset
    '''
    def __init__(self, path, transform=None):
        '''
        Load paths for test images

        :param path: path to the test dataset
        :type path: str
        :param transform: transform to apply to the images, defaults to None
        :type transform: torchvision.transforms, optional
        '''
        self.path = path
        self.path_images = os.path.join(path, 'test_images')
        self.transform = transform

        self.image_names = sorted(os.listdir(self.path_images))

        self.player_decoding = {
            0: 'p1',
            1: 'p2',
            2: 'p3',
            3: 'p4',
        }

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        '''
        Get the player crops for the given index

        :param idx: index of the image to retrieve
        :type idx: int
        :return: image ID and player crops
        :rtype: tuple
        '''
        image_name = self.image_names[idx]
        image_id = os.path.splitext(image_name)[0]

        image_path = os.path.join(self.path_images, image_name)
        image = skimage.io.imread(image_path)

        if image.shape[-1] == 4:
            image = image[:, :, :3]

        _, players = crop_rotate_roi(image)
        players = reshape_players(players)

        players_out = []

        for player in players:
            player = np.ascontiguousarray(player)

            if self.transform is not None:
                player = self.transform(player)

            players_out.append(player)

        players_out = torch.stack(players_out)

        return image_id, players_out