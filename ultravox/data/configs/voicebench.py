from ultravox.data import types

SYSTEM_PROMPT = """
You are a friendly and helpful character. You love to answer questions for people.
"""
DUMMY_ASSISTANT_TEMPLATE = "I'm sorry, I don't know the answer to that question."

# Datasets with reference answers (use yes/no evaluation)
VB_BBH_CONFIG = types.DatasetConfig(
    name="voicebench-bbh",
    path="hlt-lab/voicebench",
    subset="bbh",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(
        metric="voicebench_bbh",
        extra_kwargs_map={"id": "id"},
    ),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template="{{reference}}",
    system_prompt_template=SYSTEM_PROMPT,
)

VB_MMSU_CONFIG = types.DatasetConfig(
    name="voicebench-mmsu",
    path="hlt-lab/voicebench",
    subset="mmsu",
    splits=[
        types.DatasetSplitConfig(
            name="law", num_samples=51, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="engineering", num_samples=107, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="other", num_samples=546, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="biology", num_samples=172, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="business", num_samples=236, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="economics", num_samples=280, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="health", num_samples=406, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="philosophy", num_samples=305, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="psychology", num_samples=317, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="history", num_samples=104, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="chemistry", num_samples=167, split=types.DatasetSplit.TEST
        ),
        types.DatasetSplitConfig(
            name="physics", num_samples=383, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(metric="voicebench_mcq"),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template="{{reference}}",
    system_prompt_template=SYSTEM_PROMPT,
)

VB_OPENBOOKQA_CONFIG = types.DatasetConfig(
    name="voicebench-openbookqa",
    path="hlt-lab/voicebench",
    subset="openbookqa",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=455, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(metric="voicebench_mcq"),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template="{{reference}}",
    system_prompt_template=SYSTEM_PROMPT,
)

VB_SD_QA_CONFIG = types.DatasetConfig(
    name="voicebench-sd-qa",
    path="hlt-lab/voicebench",
    subset="sd-qa",
    splits=[
        types.DatasetSplitConfig(
            name="usa", num_samples=553, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(
        metric="voicebench_yes_no", args={"evaluator": "gpt-4o-mini"}
    ),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template="{{reference}}",
    system_prompt_template=SYSTEM_PROMPT,
)
# The following splits could be used to create separate test sets in the future
# types.DatasetSplitConfig(name="aus", num_samples=553, split=types.DatasetSplit.TEST)
# types.DatasetSplitConfig(name="gbr", num_samples=553, split=types.DatasetSplit.TEST)
# types.DatasetSplitConfig(name="ind_n", num_samples=553, split=types.DatasetSplit.TEST)
# types.DatasetSplitConfig(name="ind_s", num_samples=553, split=types.DatasetSplit.TEST)
# types.DatasetSplitConfig(name="irl", num_samples=553, split=types.DatasetSplit.TEST)
# types.DatasetSplitConfig(name="kenya", num_samples=553, split=types.DatasetSplit.TEST)
# types.DatasetSplitConfig(name="nga", num_samples=553, split=types.DatasetSplit.TEST)
# types.DatasetSplitConfig(name="nzl", num_samples=553, split=types.DatasetSplit.TEST)
# types.DatasetSplitConfig(name="phl", num_samples=553, split=types.DatasetSplit.TEST)
# types.DatasetSplitConfig(name="zaf", num_samples=553, split=types.DatasetSplit.TEST)

# Datasets without reference answers (use scalar evaluation)
VB_ADVBENCH_CONFIG = types.DatasetConfig(
    name="voicebench-advbench",
    path="hlt-lab/voicebench",
    subset="advbench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=520, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(metric="voicebench_harm"),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template=DUMMY_ASSISTANT_TEMPLATE,
    system_prompt_template=SYSTEM_PROMPT,
)

VB_ALPACAEVAL_CONFIG = types.DatasetConfig(
    name="voicebench-alpacaeval",
    path="hlt-lab/voicebench",
    subset="alpacaeval",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=199, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(
        metric="voicebench_scalar", args={"evaluator": "gpt-4o-mini"}
    ),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template=DUMMY_ASSISTANT_TEMPLATE,
    system_prompt_template=SYSTEM_PROMPT,
)

VB_ALPACAEVAL_FULL_CONFIG = types.DatasetConfig(
    name="voicebench-alpacaeval-full",
    path="hlt-lab/voicebench",
    subset="alpacaeval_full",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=636, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(
        metric="voicebench_scalar", args={"evaluator": "gpt-4o-mini"}
    ),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template=DUMMY_ASSISTANT_TEMPLATE,
    system_prompt_template=SYSTEM_PROMPT,
)

VB_ALPACAEVAL_SPEAKER_CONFIG = types.DatasetConfig(
    name="voicebench-alpacaeval-speaker",
    path="hlt-lab/voicebench",
    subset="alpacaeval_speaker",
    splits=[
        types.DatasetSplitConfig(
            name="en_AU_Wavenet_A_1.0_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_AU_Wavenet_B_1.0_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_IN_Wavenet_A_1.0_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_IN_Wavenet_B_1.0_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_GB_Wavenet_A_1.0_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_GB_Wavenet_B_1.0_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_US_Wavenet_A_1.0_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_US_Wavenet_C_1.0_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_US_Wavenet_A_1.5_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_US_Wavenet_A_2.0_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
        types.DatasetSplitConfig(
            name="en_US_Wavenet_A_0.5_0.0_0.0",
            num_samples=636,
            split=types.DatasetSplit.TEST,
        ),
    ],
    eval_config=types.EvalConfig(
        metric="voicebench_scalar", args={"evaluator": "gpt-4o-mini"}
    ),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template="",
    system_prompt_template=SYSTEM_PROMPT,
)

VB_COMMONEVAL_CONFIG = types.DatasetConfig(
    name="voicebench-commoneval",
    path="hlt-lab/voicebench",
    subset="commoneval",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=200, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(
        metric="voicebench_scalar", args={"evaluator": "gpt-4o-mini"}
    ),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template=DUMMY_ASSISTANT_TEMPLATE,
    system_prompt_template=SYSTEM_PROMPT,
)

VB_IFEVAL_CONFIG = types.DatasetConfig(
    name="voicebench-ifeval",
    path="hlt-lab/voicebench",
    subset="ifeval",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=345, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(
        metric="voicebench_ifeval",
        extra_kwargs_map={
            "id": "id",
            "key": "key",
            "instruction_id_list": "instruction_id_list",
            "kwargs": "kwargs",
        },
    ),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template=DUMMY_ASSISTANT_TEMPLATE,
    system_prompt_template=SYSTEM_PROMPT,
)

VB_MTBENCH_CONFIG = types.DatasetConfig(
    name="voicebench-mtbench",
    path="hlt-lab/voicebench",
    subset="mtbench",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=46, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(
        metric="voicebench_scalar", args={"evaluator": "gpt-4o-mini"}
    ),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template=DUMMY_ASSISTANT_TEMPLATE,
    system_prompt_template=SYSTEM_PROMPT,
)

VB_WILDVOICE_CONFIG = types.DatasetConfig(
    name="voicebench-wildvoice",
    path="hlt-lab/voicebench",
    subset="wildvoice",
    splits=[
        types.DatasetSplitConfig(
            name="test", num_samples=1000, split=types.DatasetSplit.TEST
        ),
    ],
    eval_config=types.EvalConfig(
        metric="voicebench_scalar", args={"evaluator": "gpt-4o-mini"}
    ),
    user_template=types.AUDIO_PLACEHOLDER,
    transcript_template="{{prompt}}",
    assistant_template=DUMMY_ASSISTANT_TEMPLATE,
    system_prompt_template=SYSTEM_PROMPT,
)

configs = [
    # Reference-based datasets (Yes/No evaluation)
    VB_BBH_CONFIG,
    VB_MMSU_CONFIG,
    VB_OPENBOOKQA_CONFIG,
    VB_SD_QA_CONFIG,
    # Open-ended datasets (Scalar evaluation)
    VB_ADVBENCH_CONFIG,
    VB_ALPACAEVAL_CONFIG,
    VB_ALPACAEVAL_FULL_CONFIG,
    VB_ALPACAEVAL_SPEAKER_CONFIG,
    VB_COMMONEVAL_CONFIG,
    VB_IFEVAL_CONFIG,
    VB_MTBENCH_CONFIG,
    VB_WILDVOICE_CONFIG,
]
