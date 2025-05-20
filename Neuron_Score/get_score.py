import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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
    Seq2SeqTrainingArguments
)

import sys
from models.importance_llama import LlamaForCausalLM
from models.importance_qwen2 import Qwen2ForCausalLM

from data_preprocess import DataProcessor, IGNORE_INDEX
from importance_trainer import CustomSeq2SeqTrainer

class DomainType(Enum):
    Cluster = 1
    TaskType = 2

def load_dataset(task_name: str, domain_type: DomainType, seed):
    if domain_type == DomainType.Cluster:
        assert 'domain' in task_name
        sample_count = 5000
        dataset = load_from_disk(f'./Cluster/domains/cluster/{task_name}')
    elif domain_type == DomainType.TaskType:
        assert 'domain' in task_name
        sample_count = 5000
        dataset = load_from_disk(f'./Cluster/domains/task_type/{task_name}')

    return dataset.shuffle(seed=seed).select(range(sample_count))

def get_output_dir(model_name: str, task_name: str, domain_type: DomainType):
    if domain_type == DomainType.Cluster:
        output_dir = f'./Neuron_Score/score/cluster/{model_name}/{task_name}'
    elif domain_type == DomainType.TaskType:
        output_dir = f'./Neuron_Score/score/task_type/{model_name}/{task_name}'
        
    if os.path.exists(output_dir) and os.listdir(output_dir):
        raise FileExistsError(f'{output_dir} already exists')
    
    return output_dir

def main(model_name: str, task_name: str, domain_type: DomainType):
    print(f"model_name: {model_name}, task_name: {task_name}, domain_type: {domain_type.name}")
    random_seed = 42
    set_seed(random_seed)
    
    model_path = f'/usr/workdir/models/{model_name}'
    if 'llama' in model_name:
        model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        if 'llama3' in model_name:
            template = 'llama3'
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError(f'Important model {model_name} not supported')
    
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(task_name, domain_type, random_seed)
    dataset = DataProcessor(dataset, tokenizer, template_name=template, max_len=1024).dataset
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if tokenizer.padding_side == "right" else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX,
    )
    
    model.cuda()
    if config.model_type in ["llama", "qwen2"]:
        layer_range = range(config.num_hidden_layers)
        for param in model.parameters():
            param.requires_grad = False
        for observe_layer in layer_range:
            for param in model.model.layers[observe_layer].mlp.parameters():
                param.requires_grad = True
    else:
        raise NotImplementedError
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=get_output_dir(model_name, task_name, domain_type),
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
    
    
    if training_args.do_eval:
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
    

if __name__ == '__main__':
    model_name, task_name, domain_type = sys.argv[1], sys.argv[2], DomainType(int(sys.argv[3]))
    # model_name, task_name, domain_type = 'llama3.2-1b', 'domain0', DomainType.TaskType
    main(model_name, task_name, domain_type)