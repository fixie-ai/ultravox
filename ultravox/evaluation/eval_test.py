from ultravox.evaluation import eval


def test_eval():
    eval.main(["--config_path", "training/configs/test_eval.yaml"])
