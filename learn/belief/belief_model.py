import os
import cv2
import torch
import shutil
import imageio
import numpy as np
from typing import Optional

from torch import nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader

from util.util import *
from learn.belief.models import *
from learn.belief.custom_dataset import CustomDataset

import numpy as np
import cv2


import numpy as np
import cv2


def create_mpe_image(
    coords,
    colors,
    stars,
    width=800,
    height=800,
    radius=5,
    world_range=10,
    background=(255, 255, 255),
    output_path=None,
):
    """
    Create an RGB image from world coordinates.

    Coordinate system:
        x, y ∈ [-world_range, world_range]

    (0,0) is centered in the image.

    """

    # create background
    img = np.full((height, width, 3), background, dtype=np.uint8)

    # color map (RGB)
    color_map = {
        "r": (255, 0, 0),
        "g": (0, 255, 0),
        "b": (0, 0, 255),
        "y": (255, 255, 0),
        "w": (255, 255, 255),
        "k": (0, 0, 0),
        "c": (0, 255, 255),
        "m": (255, 0, 255),
    }

    for (x, y), c, s in zip(coords, colors, stars):

        if c not in color_map:
            raise ValueError(f"Unknown color code: {c}")

        # world -> image transform
        px = int(((x + world_range) / (2 * world_range)) * width)
        py = int(((world_range - y) / (2 * world_range)) * height)

        rgb = color_map[c]

        # RGB -> BGR
        bgr = (rgb[2], rgb[1], rgb[0])

        # ---- DRAW STAR ----
        if s == 1:

            star_size = radius * 2

            cv2.line(
                img,
                (px - star_size, py),
                (px + star_size, py),
                bgr,
                2,
            )

            cv2.line(
                img,
                (px, py - star_size),
                (px, py + star_size),
                bgr,
                2,
            )

            cv2.line(
                img,
                (px - star_size, py - star_size),
                (px + star_size, py + star_size),
                bgr,
                2,
            )

            cv2.line(
                img,
                (px - star_size, py + star_size),
                (px + star_size, py - star_size),
                bgr,
                2,
            )

        # ---- DRAW NORMAL DOT ----
        else:
            cv2.circle(
                img,
                center=(px, py),
                radius=radius,
                color=bgr,
                thickness=-1,
            )

    # convert to RGB before returning
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #if output_path is not None:
    #    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return img

COLOR_SCALE = np.array([
    [255, 0,   0],
    [242, 13,  0],
    [228, 27,  0],
    [215, 40,  0],
    [201, 54,  0],
    [188, 67,  0],
    [174, 81,  0],
    [161, 94,  0],
    [147, 108, 0],
    [134, 121, 0],
    [120, 135, 0],
    [107, 148, 0],
    [93,  162, 0],
    [80,  175, 0],
    [67,  188, 0],
    [53,  202, 0],
    [40,  215, 0],
    [26,  229, 0],
    [13,  242, 0],
    [0,   0, 255]
], dtype=np.uint8)

def boat_plot_data(
    pos,
    hdg,
    scale=10,
    color=None,
) -> dict:
    '''
    returns dictionary w/ plottind data for Renderer2D

    output
    ------
    dict
        plotting data keys:[lines,points]
    '''

    boat_points = np.array([
        [0.0, 0.25],
        [1.5, 0.25],
        [2.0, 0.0],
        [1.5, -0.25],
        [0.0, -0.25]
    ])

    boat_lines = np.array([
        (0,1), (1,2), (2,3), (3,4), (4,0)
    ])

    points = []
    lines = []
    colors = []

    for i,(pos,hdg) in enumerate(zip(pos,hdg)):

        dcm = np.array([
            [np.cos(hdg), -np.sin(hdg)],
            [np.sin(hdg),  np.cos(hdg)]
        ])
        transformed_vertices = np.dot(boat_points*scale, dcm.T) + pos*scale
        for j in range(len(boat_points)):
            points.append(transformed_vertices[j])
            lines.append((boat_lines[j][0]+(5*i),boat_lines[j][1]+(5*i)))
            if color is not None:
                colors.append(color[i])
            else:
                colors.append('k')

    plot_data = {}
    plot_data['lines'] = lines
    plot_data['points'] = points
    plot_data['colors'] = colors

    return plot_data

class BeliefModel(pl.LightningModule):

    def __init__(self,
                 lr: float=1e-4,
                 seed: int=1,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=1.0,
                 log_dir: str='logs',
                 train_batch: int=512,
                 val_batch: int=256,
                 test_batch: int=256,
                 num_workers: int=8,
                 model_name: str='NN2CNN',
                 noise_val: Optional[float]=None,
                 data_dir: str='data',
                 data_name: str='sinusoidal_data',
                 input_channels: int=1,
                 output_channels: int=1,
                 lr_schedule: list=[],
                 output_noise: float=0.0,
                 window_size: int=61,
                 num_frames: int=10,
                 num_classes: int=2,
                 **kwargs,
        ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}
        self.__build_model()

    def __build_model(self):
        #if self.hparams.model_name == 'bayesianCNN':
        #self.model = NN2CNN(self.hparams.input_channels,self.hparams.output_channels)
        if self.hparams.model_name == 'predator_prey_NN':
            self.model = predator_prey_NN(self.hparams.input_channels,self.hparams.output_channels)
            self.loss_func = self.model.loss
            self.val_loss_func = self.model.val_loss
            self.vae = False
        if self.hparams.model_name == 'predator_prey_VAE_NN':
            self.model = predator_prey_VAE_NN(self.hparams.input_channels,self.hparams.output_channels)
            self.loss_func = self.model.loss #torch.nn.MSELoss()
            self.val_loss_func = self.model.val_loss
            self.vae = True
        if self.hparams.model_name == 'drones_NN':
            self.model = drones_NN(self.hparams.input_channels,self.hparams.output_channels)
            self.loss_func = self.model.loss
            self.val_loss_func = self.model.val_loss
            self.vae = False
        if self.hparams.model_name == 'drones_VAE_NN':
            self.model = drones_VAE_NN(self.hparams.input_channels,self.hparams.output_channels)
            self.loss_func = self.model.loss #torch.nn.MSELoss()
            self.val_loss_func = self.model.val_loss
            self.vae = True
        if self.hparams.model_name in ['football_NN', 'predator_prey_swarm_NN', 'market_game_large_NN']:
            self.model = football_NN(self.hparams.input_channels,self.hparams.output_channels)
            self.loss_func = self.model.loss
            self.val_loss_func = self.model.val_loss
            self.vae = False
        if self.hparams.model_name == 'fire_NN':
            self.model = fire_NN(
                self.hparams.input_channels,
                self.hparams.output_channels,
                window_size=self.hparams.window_size,
                num_frames=self.hparams.num_frames,
                num_classes=self.hparams.num_classes,
            )
            self.loss_func = self.model.loss
            self.val_loss_func = self.model.val_loss
            self.vae = False
        if self.hparams.model_name == 'fire_CNN':
            self.model = fire_CNN(
                self.hparams.input_channels,
                self.hparams.output_channels,
                window_size=self.hparams.window_size,
                num_frames=self.hparams.num_frames,
                num_classes=self.hparams.num_classes,
            )
            self.loss_func = self.model.loss
            self.val_loss_func = self.model.val_loss
            self.vae = False
        if self.hparams.model_name == 'market_game_NN':
            self.model = market_game_NN(self.hparams.input_channels,self.hparams.output_channels)
            self.loss_func = self.model.loss
            self.val_loss_func = self.model.val_loss
            self.vae = False
        if self.hparams.model_name == 'football_VAE_NN':
            self.model = football_VAE_NN(self.hparams.input_channels,self.hparams.output_channels)
            self.loss_func = self.model.loss
            self.val_loss_func = self.model.val_loss
            self.vae = True
        
    def sample_stochastic(self, x, sample_size=40):
        inputs = x.repeat(sample_size,1)
        output, mu, logvar = self.model(inputs,stochastic=True)

        return output


    def train_forward(self, x):
        if self.vae:
            output, mu, logvar = self.model(x)
            return output, mu, logvar
        else:
            output = self.model(x)
            return output 

    def training_step(self, batch, batch_idx):
        data, target, filepath = batch
        if self.vae:
            output, mu, logvar = self.model(data)
            train_loss, kl = self.loss_func(output,target,mu,logvar)
        else:
            output = self.model(data)
            train_loss = self.loss_func(output, target)

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if self.vae:
            self.log('train_kl_loss', kl, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return train_loss

    def validation_step(self, batch, batch_idx):
        data, target, filepath = batch

        if self.vae:
            output, mu, logvar = self.model(data)
        else:
            output = self.model(data)

        val_loss = self.val_loss_func(output,target)
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return val_loss

    def test_step(self, batch, batch_idx):
        data, target, filepath = batch
        self.all_filepaths.extend(filepath)

        error = 0

        if self.vae:
            output, mu, logvar = self.train_forward(data)
        else:
            output = self.train_forward(data)

        #avg_error = np.sum(np.linalg.norm(output.detach().cpu().numpy() - target.squeeze().detach().cpu().numpy(),axis=1))/len(data)
        test_loss = self.val_loss_func(output,target.squeeze())
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.all_predictions.extend(output.detach().cpu().numpy())
        self.all_ground_truths.extend(target.squeeze().detach().cpu().numpy())
        self.all_inputs.extend(data.detach().cpu().numpy())

    def test_save(self):
        #print(self.all_filepaths)
        folderlist = self.test_dataset.get_folderlist()
        folder_idxs = np.random.choice(len(folderlist), size=5, replace=False)
        #generate and save images
        save_dir = os.path.join(self.hparams.log_dir, 'test_outputs')
        mkdir(save_dir)
        print('Saving test output images to: ', save_dir)
        for folder_idx in folder_idxs:
            folder = folderlist[folder_idx]
            file_names = [s for i, s in enumerate(self.test_dataset.get_filelist()) if folder in s]
            if len(file_names) > 0:
                file_names = sorted(file_names, key=lambda s: int(s.split("_")[1]))
                file_idxs = [self.all_filepaths.index(s) for s in file_names if s in self.all_filepaths]
                #for i in file_idxs:
                #    print(self.all_filepaths[i])
                imgs = []
                for idx in file_idxs:
                    pred = self.all_predictions[idx]
                    gt = self.all_ground_truths[idx]
                    data = self.all_inputs[idx]
                    offset = data[-4:-2]
                    poses = np.array([
                            - offset,
                        data[-2:]-offset,
                        pred[0:2]  - offset,
                        pred[2:4] - offset,
                        gt[0:2] - offset,
                        gt[2:4] - offset,
                        data[-4:-2] - offset
                    ])
                    #print(np.linalg.norm(np.array([gt[0],gt[1]]) - np.array([pred[0],pred[1]])))
                    clrs = ['b','k','g','g','r','r','k']
                    stars = [0,0,0,0,0,0,1]
                    img = create_mpe_image(poses,clrs,stars)
                    imgs.append(img)

                save_file = os.path.join(save_dir, f'sim_{folder_idx}.gif')
                save_cv2_images_as_gif(imgs, save_file, fps=0.005)
                #cv2.imwrite(save_file, img)
            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if stage=='fit':
            kwargs = {
                    'data_dir':self.hparams.data_dir,
                    'data_name':self.hparams.data_name,
                    'flag':'train',
                    'seed':self.hparams.seed,
                }
            self.train_dataset = CustomDataset(**kwargs)
            kwargs['flag'] = 'val'
            self.val_dataset = CustomDataset(**kwargs)
        if stage=='test':
            kwargs = {
                    'data_dir':self.hparams.data_dir,
                    'data_name':self.hparams.data_name,
                    'flag':'test',
                    'seed':self.hparams.seed,
                }
            self.test_dataset = CustomDataset(**kwargs)
            self.all_predictions = []
            self.all_ground_truths = []
            self.all_inputs = []
            self.all_filepaths = []
            self.all_sims = []
        pass

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.hparams.train_batch,
                                                   shuffle=True,
                                                   **self.kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                 batch_size=self.hparams.val_batch,
                                                 shuffle=False,
                                                 **self.kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.hparams.test_batch,
                                                  shuffle=False,
                                                  **self.kwargs)
        return test_loader