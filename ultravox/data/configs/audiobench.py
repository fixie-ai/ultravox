from ultravox.data import types

INSTRUCTION_MQA_USER_TEMPLATE = (
    f"{types.AUDIO_PLACEHOLDER} \n\n{{{{instruction}}}} \n\n{{{{choices}}}}"
)
INSTRUCTION_USER_TEMPLATE = f"{types.AUDIO_PLACEHOLDER} \n\n{{{{instruction}}}}"

# English only
AB_CN_COLLEGE_LISTEN_MCQ_CONFIG = types.DatasetConfig(
    name="audiobench-cn-college-listen-mcq",
    path="fixie-ai/cn_college_listen_mcq_test",
    splits=[
        types.DatasetSplitConfig(name="test", num_samples=2_270),
    ],
    eval_config=types.EvalConfig(metric="audiobench_binary"),
    user_template=INSTRUCTION_MQA_USER_TEMPLATE,
    transcript_template="{{instruction}}",
    assistant_template="{{answer}}",
)

AB_DREAM_TTS_MCQ_CONFIG = types.DatasetConfig(
    name="audiobench-dream-tts-mcq",
    path="fixie-ai/dream_tts_mcq_test",
    splits=[
        types.DatasetSplitConfig(name="test", num_samples=1_910),
    ],
    eval_config=types.EvalConfig(metric="audiobench_binary"),
    user_template=INSTRUCTION_MQA_USER_TEMPLATE,
    transcript_template="{{instruction}}",
    assistant_template="{{answer}}",
)

AB_SLUE_P2_SQA5_CONFIG = types.DatasetConfig(
    name="audiobench-slue-p2-sqa5",
    path="fixie-ai/slue_p2_sqa5_test",
    splits=[
        types.DatasetSplitConfig(name="test", num_samples=408),
    ],
    eval_config=types.EvalConfig(metric="audiobench_scalar"),
    user_template=INSTRUCTION_USER_TEMPLATE,
    transcript_template="{{instruction}}",
    assistant_template="{{answer}}",
)

AB_PUBLIC_SG_SPEECH_QA_CONFIG = types.DatasetConfig(
    name="audiobench-public-sg-speech-qa",
    path="fixie-ai/public_sg_speech_qa_test",
    splits=[
        types.DatasetSplitConfig(name="test", num_samples=688),
    ],
    eval_config=types.EvalConfig(metric="audiobench_scalar"),
    user_template=INSTRUCTION_USER_TEMPLATE,
    transcript_template="{{instruction}}",
    assistant_template="{{answer}}",
)


configs = [
    AB_CN_COLLEGE_LISTEN_MCQ_CONFIG,
    AB_SLUE_P2_SQA5_CONFIG,
    AB_DREAM_TTS_MCQ_CONFIG,
    AB_PUBLIC_SG_SPEECH_QA_CONFIG,
]
