from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional

from MoEfication.moefication import MoEArgs

@dataclass
class MoEArguments(MoEArgs):
    r"""
    Arguments pertaining to moefication.
    """
    task_name: str = field(
        default="",
        metadata={"help": "the task name used to train the gate and lora."},
    )
    moeficate: bool = field(
        default=False,
        metadata={
            "help": "whether to moeficate the model."
        },
    )
    training_gate_layers: Optional[str] = field(
        default=None,
        metadata={
            "help": """Index of layer to train gate module. \
                    Use commas to separate multiple layers. \
                    Example: "1,2,3"
                    """
        },
    )

    def __post_init__(self, task_str):
        MoEArgs.__post_init__(self)
        self.training_gate_layers = [int(i) for i in self.training_gate_layers.split(',')] if self.training_gate_layers else None
        
        child_data = vars(self)
        moe_data = {k: child_data[k] for k in MoEArgs.__dataclass_fields__.keys() if k in child_data}
        self.moeargs = MoEArgs(**moe_data)

    def __str__(self) -> str:
        s = MoEArgs.__str__(self)
            
        return s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
