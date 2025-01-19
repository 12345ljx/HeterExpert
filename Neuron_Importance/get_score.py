import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["WANDB_DISABLED"] = 'true'
from enum import Enum
from pathlib import Path
import torch
from datasets import load_from_disk
from transformers import (
    set_seed, 
    AutoTokenizer, 
    AutoConfig,
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments, 
    LogitsProcessorList, 
    InfNanRemoveLogitsProcessor
)

import sys
sys.path.append('/usr/workdir/HeterExpert')
from models.importance_llama import LlamaForCausalLM
from cluster_plot import read_examples
sys.path.append('/usr/workdir/LLaMA-Factory/src/llamafactory/hparams')
from generating_args import GeneratingArguments

from data_preprocess import DataProcessor, IGNORE_INDEX
from importance_trainer import CustomSeq2SeqTrainer

class DomainType(Enum):
    Cluster = 1
    TaskType = 2
    TaskSingle = 3

def load_dataset(task_name: str, domain_type: DomainType, seed):
    if domain_type == DomainType.Cluster:
        assert 'domain' in task_name
        sample_count = 5000
        dataset = load_from_disk(f'/usr/workdir/HeterExpert/domains/cluster/{task_name}')
    elif domain_type == DomainType.TaskType:
        assert 'domain' in task_name
        sample_count = 5000
        dataset = load_from_disk(f'/usr/workdir/HeterExpert/domains/task_type/{task_name}')
    elif domain_type == DomainType.TaskSingle:
        sample_count = 1000
        task_path = Path(f'/usr/workdir/HeterExpert/data/{task_name}_template')
        if not task_path.is_dir():
            raise FileNotFoundError(f'{task_path} not found')
        dataset = read_examples(task_path)

    return dataset.shuffle(seed=seed).select(range(sample_count))

def get_output_dir(task_name: str, domain_type: DomainType):
    if domain_type == DomainType.Cluster:
        output_dir = f'/usr/workdir/HeterExpert/Neuron_Importance/score/cluster/{task_name}'
    elif domain_type == DomainType.TaskType:
        output_dir = f'/usr/workdir/HeterExpert/Neuron_Importance/score/task_type/{task_name}'
    elif domain_type == DomainType.TaskSingle:
        output_dir = f'/usr/workdir/HeterExpert/Neuron_Importance/score/task_single/{task_name}'
        
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f'{output_dir} already exists')
    
    return output_dir

def main(task_name: str, domain_type: DomainType):
    print(f"task_name: {task_name}, domain_type: {domain_type.name}")
    random_seed = 42
    set_seed(random_seed)
    dataset = load_dataset(task_name, domain_type, random_seed)
    
    model_path = '/usr/workdir/models/llama3.2-1b'
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = DataProcessor(dataset, tokenizer, max_len=3000).dataset
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX,
    )
    
    model.cuda()
    if config.model_type == "llama":
        layer_range = range(config.num_hidden_layers)
        for param in model.parameters():
            param.requires_grad = False
        for observe_layer in layer_range:
            for param in model.model.layers[observe_layer].mlp.parameters():
                param.requires_grad = True
    else:
        raise NotImplementedError
    
    # for name, param in model.named_parameters():
    #     print(
    #         "name: {}, dtype: {}, size: {}, device: {}, trainable: {}".format(
    #             name, param.dtype, param.size(), param.device, param.requires_grad
    #         )
    #     )
    
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=get_output_dir(task_name, domain_type),
        overwrite_output_dir=True,
        do_train=False, 
        do_eval=True, 
        do_predict=False,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,
        num_train_epochs=1.0,
        logging_steps=10,
        save_strategy='steps',
        save_steps=20,
        eval_strategy='steps', 
        eval_steps=10,
        save_total_limit=3,
    )
    
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=None,
        eval_dataset=dataset,
        data_collator=data_collator,
        compute_metrics=None,
    )
    
    generating_args = GeneratingArguments()
    # Keyword arguments for `model.generate`
    gen_kwargs = generating_args.to_dict()
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    gen_kwargs["logits_processor"] = logits_processor
    
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval", **gen_kwargs)
        trainer.log_metrics("eval", metrics)
        # trainer.save_metrics("eval", metrics)
    

if __name__ == '__main__':
    task_name, domain_type = sys.argv[1], DomainType(int(sys.argv[2]))
    # task_name, domain_type = 'cb', DomainType.TaskSingle
    main(task_name, domain_type)