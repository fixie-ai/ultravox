from ultravox.training import config_base


def test_can_create_train_config():
    # override args to [], otherwise pytest arguments will be used
    args = config_base.get_train_config([])
    assert isinstance(args, config_base.TrainConfig)
