from ultravox.data import types

# English only
BBA_BASE_CONFIG = types.DatasetConfig(
    name="bigbenchaudio",
    path="fixie-ai/big_bench_audio",
    splits=[
        types.DatasetSplitConfig(
            name="train", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(
        metric="bigbench",
    ),
    user_template=types.QA_USER_TEMPLATE,
    transcript_template="{{transcript}}",
    assistant_template="{{official_answer}}",
)

configs = [
    BBA_BASE_CONFIG,
]
