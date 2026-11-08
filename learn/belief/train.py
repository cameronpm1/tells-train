import os
import sys
import yaml
import glob
import torch
import pprint
import shutil
from munch import munchify
from datetime import datetime
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from learn.belief.belief_model import BeliefModel

def mkdir(folder):
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)


def train(config_filepath, belief_dir: str = None, lr: float = None):
    config_filepath = config_filepath
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        cfg.data_name,
                        cfg.model_name,
                        cfg.label])
    cfg['log_dir'] = log_dir

    if lr is not None:
        print('Overriding config lr with:', lr)
        cfg['lr'] = lr

    model = BeliefModel(**cfg)

    if belief_dir is not None:
        checkpoint_filepath = glob.glob(os.path.join(belief_dir, '*.ckpt'))[-1]
        print('Continuing training from checkpoint:', checkpoint_filepath)
        ckpt = torch.load(checkpoint_filepath)
        model.load_state_dict(ckpt['state_dict'])

    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        dirpath=log_dir + "/lightning_logs/checkpoints/{epoch}_{val_loss}",
        verbose=True,
        monitor='val_loss',
        mode='min')

    # optional early stop (config early_stop_patience, in epochs): the best checkpoint above is still the one kept
    callbacks = [checkpoint_callback]
    if cfg.get('early_stop_patience'):
        callbacks.append(EarlyStopping(monitor='val_loss', mode='min', patience=int(cfg.early_stop_patience)))

    #setup tensorboard logger
    now = datetime.now()
    dt_string = now.strftime('%d'+'_'+'%m'+'_'+'%Y'+'_'+'%H:%M:%S')
    tb_log_dir = log_dir + '/tb_logs/'
    if not os.path.exists(tb_log_dir):
        mkdir(tb_log_dir)
    logger = TensorBoardLogger(tb_log_dir, name=dt_string)

    # define trainer
    trainer = Trainer(logger=logger,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      accelerator='cuda',
                      default_root_dir=log_dir,
                      val_check_interval=1.0,
                      callbacks=callbacks
                      )

    trainer.fit(model)

if __name__=="__main__":
    torch.set_float32_matmul_precision('high')

    cfg_filepath = 'configs/rf_data1_32x32/bayesianCNN/config1.yaml'
    ckpt_filepath = 'logs/_rf_data1_32x32_bayesianCNN_1/lightning_logs/checkpoints/{epoch}_{val_loss}/'
    learn(cfg_filepath)
    eval(cfg_filepath,ckpt_filepath)