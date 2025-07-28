import platform

import numpy as np
import pytest

from ultravox.data.aug import AugRegistry


@pytest.mark.parametrize("gain_db", [10, -10])
def test_gain(gain_db):
    augmentation_config = AugRegistry.get_config("gain", {"gain_db": gain_db})
    augmentation = AugRegistry.create_augmentation(augmentation_config)
    audio = np.random.randn(16000) * 0.01  # white noise with amplitude 0.01
    augmented_audio = augmentation(audio)
    assert augmented_audio.shape == audio.shape
    assert np.isclose(
        np.square(augmented_audio).mean() / np.square(audio).mean(),
        10 ** (gain_db / 10),
        atol=0.2,
    )


def test_grouped_augmentation():
    aug_group_config = AugRegistry.get_config(
        "my_aug",
        override_args=dict(
            children=[
                AugRegistry.get_config("gain", {"gain_db": 10}),
                AugRegistry.get_config("gain", {"gain_db": -20}),
            ]
        ),
    )
    augmentation = AugRegistry.create_augmentation(aug_group_config)
    audio = np.random.randn(16000) * 0.01  # white noise with amplitude 0.01
    augmented_audio = augmentation(audio)
    assert augmented_audio.shape == audio.shape
    assert np.isclose(
        np.square(augmented_audio).mean() / np.square(audio).mean(),
        10 ** (-10 / 10),
        atol=0.1,
    )


def test_all_registered_augmentations():
    """
    Test that all registered augmentations can be created and applied to audio data.
    This ensures that every augmentation in the registry is working as expected.
    """
    # Get all registered augmentation configs
    registered_configs = list(AugRegistry._configs.keys())

    # Skip the null augmentation as it's just a shell
    if "null" in registered_configs:
        registered_configs.remove("null")

    # Ensure we have augmentations to test
    assert len(registered_configs) > 0, "No augmentations found in registry"

    # Create sample audio for testing (white noise)
    sample_audio = np.clip(np.random.randn(16000) * 0.01, -1, 1)

    # Test each augmentation type
    for aug_name in registered_configs:

        if not platform.system() == "Linux" and aug_name in [
            "amr_wb",
            "random_amr_compression",
        ]:
            print(f"Skipping {aug_name} on non-Linux platform")
            continue

        config = AugRegistry.get_config(aug_name)

        # Create the augmentation instance
        augmentation = AugRegistry.create_augmentation(config)

        # Apply the augmentation to our sample audio
        try:
            augmented_audio = augmentation(sample_audio)

            # Check that the output has the same shape as the input
            if not (augmented_audio.shape == sample_audio.shape):
                if aug_name.startswith("amr") or aug_name == "random_amr_compression":
                    print(
                        f"Warning: Augmentation {aug_name} changed audio shape from {sample_audio.shape} to {augmented_audio.shape}"
                    )
                else:
                    raise ValueError(
                        f"Augmentation {aug_name} changed audio shape from {sample_audio.shape} to {augmented_audio.shape}"
                    )

            # Check that the output is not identical to the input (except for null augmentation)
            # This might not be true for all augmentations, so we'll just print a warning
            if np.allclose(augmented_audio[: len(sample_audio)], sample_audio):
                print(f"Warning: Augmentation {aug_name} did not modify the audio data")

        except Exception as e:
            pytest.fail(f"Augmentation {aug_name} failed with error: {str(e)}")
