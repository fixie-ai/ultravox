from ultravox.data import types

ML_BASE_CONFIG = types.DatasetConfig(
    name="multilingual_librispeech",
    path="fixie-ai/multilingual_librispeech",
    transcript_template="{{transcript}}",
    assistant_template="{{transcript}}",
)

ML_NL_CONFIG = types.DatasetConfig(
    name="multilingual_librispeech-nl",
    base="multilingual_librispeech",
    subset="dutch",
    splits=[types.DatasetSplitConfig(name="train", num_samples=37_533)],
)

ML_PT_CONFIG = types.DatasetConfig(
    name="multilingual_librispeech-pt",
    base="multilingual_librispeech",
    subset="portuguese",
    splits=[types.DatasetSplitConfig(name="train", num_samples=37_533)],
)

ML_NL_TRANS_CONFIG = types.DatasetConfig(
    name="multilingual_librispeech-nl-transcription",
    base="multilingual_librispeech-nl",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

ML_PT_TRANS_CONFIG = types.DatasetConfig(
    name="multilingual_librispeech-pt-transcription",
    base="multilingual_librispeech-pt",
    user_template=types.TRANSCRIPTION_USER_TEMPLATE,
)

ML_NL_CONT_CONFIG = types.DatasetConfig(
    name="multilingual_librispeech-nl-continuation",
    base="multilingual_librispeech-nl",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

ML_PT_CONT_CONFIG = types.DatasetConfig(
    name="multilingual_librispeech-pt-continuation",
    base="multilingual_librispeech-pt",
    user_template=types.CONTINUATION_USER_TEMPLATE,
    assistant_template=types.CONTINUATION_ASSISTANT_TEMPLATE,
)

configs = [
    ML_BASE_CONFIG,
    ML_NL_CONFIG,
    ML_PT_CONFIG,
    ML_NL_TRANS_CONFIG,
    ML_PT_TRANS_CONFIG,
    ML_NL_CONT_CONFIG,
    ML_PT_CONT_CONFIG,
]
