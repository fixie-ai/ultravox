import comet
import torch

from ultravox.evaluation import eval_types
from ultravox.training import ddp_utils


comet_model = None


def comet_eval(sample: eval_types.Sample) -> eval_types.CometResult:
    global comet_model
    if comet_model is None:
        with ddp_utils.run_on_master_first():
            model_path = comet.download_model("Unbabel/eamt22-cometinho-da")
        comet_model = comet.load_from_checkpoint(model_path)

    score = comet_model.predict(
        [
            {
                "src": sample.question,
                "mt": sample.generated_answer,
                "ref": sample.expected_answer,
            }
        ],
        gpus=1 if torch.cuda.is_available() else 0,
    )
    return eval_types.CometResult(score=score.system_score, reason="Not implemented")
