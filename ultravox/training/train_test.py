import tempfile

from ultravox.training import config_base
from ultravox.training import train


def test_train():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = config_base.get_train_config(
            [
                "--config",
                "training/configs/test_train.yaml",
                "--device",
                "cpu",
                "--output_dir",
                tmpdir,
            ]
        )
    train.train(config)
