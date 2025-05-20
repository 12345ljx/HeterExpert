from typing import TYPE_CHECKING

if __name__ != '__main__':
    from ..extras.logging import get_logger

    if TYPE_CHECKING:
        from transformers.modeling_utils import PreTrainedModel

        from ..hparams import FinetuningArguments, ModelArguments, MoEArguments
        
    logger = get_logger(__name__)

import sys
sys.path.append('/usr/workdir/MoEfication/moefication')
from moefication import moeficate

def init_moe(model: "PreTrainedModel", moe_args: "MoEArguments", finetuning_args: "FinetuningArguments"
) -> "PreTrainedModel":
    model = moeficate(model, moe_args.moeargs)
    return model
    