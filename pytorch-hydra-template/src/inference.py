from omegaconf import DictConfig
from typing import List, Optional, Tuple
import hydra
from hydra.utils import instantiate

import torch

from seed_everything import _init_fn, seedEverything
from dataset import *
from transforms import *
from testing import *
from model import *

# envirionment
print("pytorch version: {}".format(torch.__version__))
print("GPU available: {}".format(torch.cuda.is_available()))
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"num of GPU: {torch.cuda.device_count()}")

device = "cuda" if torch.cuda.is_available() else "cpu"


@hydra.main(version_base=None, config_path="../configs", config_name="inference")
def main(cfg: DictConfig) -> None:
    # Experiment name
    print(f"Model weight: '{cfg.training.save_dir}/{cfg.weight}'")

    # seed
    seedEverything(cfg.seed)

    # Dataset
    print(f"Instantiating Test Dataset.. <{cfg.dataset.test._target_}>")
    test_dataset = instantiate(
        cfg.dataset.test, mode=cfg.dataset.test.mode, transform=test_transform
    )

    # DataLoader
    print(f"Instantiating Test DataLoader.. <{cfg.dataloader._target_}>")
    test_loader = instantiate(
        cfg.dataloader,
        dataset=test_dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=False,
        num_workers=cfg.dataloader.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=_init_fn,
    )

    # Model
    model_path = os.path.join(cfg.training.save_dir, cfg.weight)
    print(f"Instantiating Model.. <{model_path}>")
    model = instantiate(cfg.model)
    model = model.to(device)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint.state_dict()
    model.load_state_dict(state_dict)
    model.eval()

    # sample_submisson.csv 열기
    submission = pd.read_csv("../submission/sample_submission.csv", index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append(
            {
                "image_id": file_name,
                "PredictionString": " ".join(str(e) for e in string.tolist()),
            },
            ignore_index=True,
        )

    # submission.csv로 저장
    submission.to_csv(
        f"../submission/{cfg.weight.split('/')[0]}_{cfg.weight.split('/')[1][:-3]}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
