from omegaconf import DictConfig
from typing import List, Optional, Tuple
import hydra
from hydra.utils import instantiate

import torch
import wandb

from seed_everything import _init_fn, seedEverything
from dataset import *
from transforms import *
from training import *
from model import *

# envirionment
print("pytorch version: {}".format(torch.__version__))
print("GPU available: {}".format(torch.cuda.is_available()))
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"num of GPU: {torch.cuda.device_count()}")

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    # Experiment name
    print(f"Experiement: {cfg.name}")

    # seed
    seedEverything(cfg.seed)

    # Dataset
    print(f"Instantiating Train Dataset.. <{cfg.dataset.train._target_}>")
    train_dataset = instantiate(
        cfg.dataset.train, mode=cfg.dataset.train.mode, transform=train_transform
    )
    print(f"Instantiating Valid Dataset.. <{cfg.dataset.valid._target_}>")
    val_dataset = instantiate(
        cfg.dataset.valid, mode=cfg.dataset.valid.mode, transform=val_transform
    )

    # DataLoader
    print(f"Instantiating Train DataLoader.. <{cfg.dataloader._target_}>")
    train_loader = instantiate(
        cfg.dataloader,
        dataset=train_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=_init_fn,
    )
    print(f"Instantiating Valid DataLoader.. <{cfg.dataloader._target_}>")
    val_loader = instantiate(
        cfg.dataloader,
        dataset=val_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=_init_fn,
    )

    # Model
    print(f"Instantiating Model.. <{cfg.model._target_}>")
    model = instantiate(cfg.model)
    model = model.to(device)

    # Loss
    print(f"Instantiating loss.. <{cfg.loss._target_}>")
    criterion = instantiate(cfg.loss)

    # Optimizer
    print(f"Instantiating Optimizer.. <{cfg.optimizer._target_}>")
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Scheduler
    print(f"Instantiating Scheduler.. <{cfg.scheduler._target_}>")
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # Train
    instantiate(cfg.logger)
    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        device,
        cfg.training.max_epoch,
        cfg.paths.output_dir,
        cfg.training.val_every,
        cfg.training.log_step,
        cfg.logger.name,
    )
    wandb.finish()


if __name__ == "__main__":
    main()
