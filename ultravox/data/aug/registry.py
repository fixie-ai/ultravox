import logging
from copy import deepcopy
from typing import Any, Dict, List, Optional, Type, Union

from .base import Augmentation
from .config import AugmentationArgs
from .config import AugmentationConfig


class AugRegistry:
    """
    Combined registry for augmentation types and configs.

    This class keeps track of both augmentation types and the configurations
    of those augmentations. There can be more than one configuration for a given
    augmentation type (e.g. different resampling rates for a resampling augmentation).
    """

    _aug_types: Dict[str, Type[Augmentation]] = {}
    _configs: Dict[str, AugmentationConfig] = {}

    @classmethod
    def register_type(cls, name: str):
        """Use a decorator to register augmentations."""

        def wrapper(aug_cls: Type[Augmentation]):
            cls.register_type_manual(name, aug_cls)
            return aug_cls

        return wrapper

    @classmethod
    def register_type_manual(cls, name: str, aug_cls: Type[Augmentation]):
        """Register an augmentation type manually."""
        cls._aug_types[name] = aug_cls

    @classmethod
    def register_config(cls, name: str, config: AugmentationConfig):
        """Register a named configuration."""
        cls._configs[name] = config

    @classmethod
    def get_type(cls, name: Union[str, None]) -> Type[Augmentation]:
        """Get an augmentation type from name."""
        if name is None:
            name = "null"
        if name not in cls._aug_types:
            raise KeyError(f"Augmentation type '{name}' not found in registry")
        return cls._aug_types[name]

    @classmethod
    def get_config(
        cls, name: str, override_args: Optional[Dict[str, Any]] = None
    ) -> AugmentationConfig:
        """Get an augmentation config from name and override args.

        If the config is not found, a new one is created with the given name.

        If override_args are provided, they will be used to update the config.
        """
        if name not in cls._configs:
            assert override_args, f"No override args specified for augmentation {name}"
            assert ("type" in override_args) ^ (
                "children" in override_args
            ), f"Only one of `type` or `children` must be in override args for augmentation {name}"
            if "type" in override_args:
                config = AugmentationConfig(name=name, type=override_args.pop("type"))
            elif "children" in override_args:
                config = AugmentationConfig(name=name)
            logging.info(f"Created new config for {name}: {config}")
        else:
            config = cls._configs[name]

        if override_args:
            # copy config to avoid mutating the original
            new_config = deepcopy(config)

            # convert children to config objects
            if "children" in override_args:
                override_args["children"] = [
                    (
                        cls.get_config(child.pop("name"), child)
                        if not isinstance(child, AugmentationConfig)
                        else child
                    )
                    for child in override_args["children"]
                ]

            new_config.update(override_args)
            return new_config

        return config

    @classmethod
    def create_augmentation(cls, config: AugmentationConfig) -> Augmentation:
        """Create an augmentation from a config."""
        # Check that we are not specifying both children and type
        assert not (
            config.children and config.type
        ), f"Cannot specify both children {config.children} and type {config.type}"

        # Create children if specified
        if config.children:
            children = [cls.create_augmentation(child) for child in config.children]
            return cls.get_type(config.type)(args=config.args, children=children)
        else:
            return cls.get_type(config.type)(args=config.args, **config.params)

    @classmethod
    def create_parent_augmentation(
        cls, augmentations: List[Augmentation]
    ) -> Augmentation:
        return cls.get_type("null")(args=AugmentationArgs(), children=augmentations)


AugRegistry.register_type_manual("null", Augmentation)
AugRegistry.register_config(
    "null",
    AugmentationConfig(
        name="null",
        type=None,
        params={},
        description="Shell augmentation",
    ),
)
