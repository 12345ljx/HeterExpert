import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
os.environ["WANDB_DISABLED"] = 'true'
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
sys.path.append('/usr/workdir/HeterExpert/models')
from importance_llama import LlamaForCausalLM
sys.path.append('/usr/workdir/LLaMA-Factory/src/llmtuner/hparams')
from generating_args import GeneratingArguments

from data_preprocess import DataProcessor, IGNORE_INDEX
from importance_trainer import CustomSeq2SeqTrainer


def main(domain=None):
    if domain is None:
        domain = 'domain7'
    print(f"Domain: {domain}")
    
    sample_count = 5000
    random_seed = 42

    set_seed(random_seed)
    model_path = '/usr/workdir/models/llama-3.2-1B'
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_from_disk(f'/usr/workdir/HeterExpert/domains/{domain}').shuffle(seed=random_seed).select(range(sample_count))
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
        output_dir=f'/usr/workdir/HeterExpert/Neuron_Importance/score5000/{domain}', 
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
    domain = sys.argv[1] if len(sys.argv) > 1 else None
    main(domain)