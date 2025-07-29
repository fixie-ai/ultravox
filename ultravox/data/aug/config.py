import dataclasses
from typing import Any, Dict, List, Optional

from simple_parsing import helpers

from ultravox.data.data_sample import SAMPLE_RATE


@dataclasses.dataclass
class AugmentationArgs:
    p: float = 1.0
    sample_rate: int = SAMPLE_RATE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p": self.p,
            "sample_rate": self.sample_rate,
        }

    def param_names(self) -> List[str]:
        return [f.name for f in dataclasses.fields(self)]


@dataclasses.dataclass
class AugmentationConfig(helpers.Serializable):
    name: str
    type: Optional[str] = None
    params: Dict[str, Any] = dataclasses.field(default_factory=dict)
    description: Optional[str] = None
    args: AugmentationArgs = dataclasses.field(default_factory=AugmentationArgs)
    children: Optional[List["AugmentationConfig"]] = None

    def update_params(self, data: Dict[str, Any]) -> None:
        self.params.update(data)

    def update(self, data: Dict[str, Any]) -> "AugmentationConfig":
        assert "type" not in data, "type cannot be updated"
        assert "name" not in data, "name cannot be updated"

        for key in self.args.param_names():
            if key in data:
                setattr(self.args, key, data.pop(key))

        if "description" in data:
            self.description = data.pop("description")

        if "children" in data:
            assert all(
                isinstance(child, AugmentationConfig) for child in data["children"]
            ), "Must update children with AugmentationConfig objects"
            self.children = data.pop("children")

        self.update_params(data)
        return self
