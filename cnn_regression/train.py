# import fire

import hydra
from omegaconf import DictConfig

from cnn_regression.utils.utils import train


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
