from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional
import sys
sys.path.append('/usr/workdir/MoEfication/moefication')
from moefication import MoEArgs
from ensemble import MixArgs

@dataclass
class MoEArguments(MoEArgs, MixArgs):
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
        MixArgs.__post_init__(self, task_str)
        self.training_gate_layers = [int(i) for i in self.training_gate_layers.split(',')] if self.training_gate_layers else None
        
        child_data = vars(self)
        moe_data = {k: child_data[k] for k in MoEArgs.__dataclass_fields__.keys() if k in child_data}
        self.moeargs = MoEArgs(**moe_data)
        mix_data = {k: child_data[k] for k in MixArgs.__dataclass_fields__.keys() if k in child_data}
        self.mixargs = MixArgs(**mix_data)

    def __str__(self) -> str:
        if self.mixargs.mix_gate_mode:
            s = [
                f'{self.task_str}(topk={self.topk})/{MoEArgs.__str__(self)}\''
                f'/xgate={self.mix_gate_mode}(xstatic={self.mix_static},xgamma={self.mix_gamma})'
                ]
            if self.forbidden_lora:
                s += '/forbidden_lora'
        else:
            s = MoEArgs.__str__(self)
            
        return s

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
