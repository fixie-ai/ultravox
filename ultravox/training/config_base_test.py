from ultravox.training import config_base


def test_can_create_train_config():
    args = config_base.get_train_args()
    assert isinstance(args, config_base.TrainConfig)
