from functools import partial
from typing import Sequence, Tuple, Any, Union
from transformers import PreTrainedTokenizer
from datasets import Dataset

from llamafactory.data.template import TEMPLATES, _add_or_replace_eos_token
from llamafactory.data.processors.processor_utils import infer_seqlen

IGNORE_INDEX = -100

def get_template(tokenizer, template_name):
    template = TEMPLATES.get(template_name, None)
    if template is None:
        raise ValueError(f"Template {template_name} does not exist.")

    stop_words = template.stop_words
    if template.replace_eos:
        if not stop_words:
            raise ValueError("Stop words are required to replace the EOS token.")
        
        eos_token = stop_words[0]
        is_added = tokenizer.eos_token_id is None
        num_added_tokens = tokenizer.add_special_tokens({"eos_token": eos_token})

        if is_added:
            print(f"Add eos token: {tokenizer.eos_token}")
        else:
            print(f"Replace eos token: {tokenizer.eos_token}")

        if num_added_tokens > 0:
            print("New tokens have been added, make sure `resize_vocab` is True.")
        stop_words = stop_words[1:]
    
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Add pad token: {tokenizer.pad_token}")
    
    return template

class DataProcessor:
    def __init__(self, dataset: Dataset, tokenizer, template_name, max_len=1024):
        dataset = dataset.add_column("sequence_length", [len(x + y) for x, y in zip(dataset['instruction'], dataset['output'])])
        dataset = dataset.sort("sequence_length", reverse=True)
        self.template = get_template(tokenizer, template_name)
        
        column_names = list(next(iter(dataset)).keys())
        preprocess_func = partial(self.preprocess_supervised_dataset, tokenizer=tokenizer, cutoff_len=max_len)
        self._dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names)
        
        print('input', dataset[0]['instruction'])
        print('output', dataset[0]['output'])
        print('input_ids', self._dataset[0]['input_ids'])
        print('attention_mask', self._dataset[0]['attention_mask'])
        print('labels', self._dataset[0]['labels'])
    
    @property
    def dataset(self):
        return self._dataset
        
    def _convert_elements_to_ids(
            self, tokenizer: "PreTrainedTokenizer", elements: list[Union[str, dict[str, str]]]
        ) -> list[int]:
            r"""
            Converts elements to token ids.
            """
            token_ids = []
            for elem in elements:
                if isinstance(elem, str):
                    if len(elem) != 0:
                        token_ids += tokenizer.encode(elem, add_special_tokens=False)
                elif isinstance(elem, dict):
                    token_ids += [tokenizer.convert_tokens_to_ids(elem.get("token"))]
                elif isinstance(elem, set):
                    if "bos_token" in elem and tokenizer.bos_token_id is not None:
                        token_ids += [tokenizer.bos_token_id]
                    elif "eos_token" in elem and tokenizer.eos_token_id is not None:
                        token_ids += [tokenizer.eos_token_id]
                else:
                    raise ValueError("Input must be string, set[str] or dict[str, str], got {}".format(type(elem)))

            return token_ids

    def _make_pairs(
            self,
            encoded_messages: Sequence[list[int]],
            cutoff_len: int
        ) -> Sequence[Tuple[list[int], list[int]]]:
            max_source_len, max_target_len = infer_seqlen(
                source_len=len(encoded_messages[0]),
                target_len=len(encoded_messages[1]),
                cutoff_len=cutoff_len
            )
            source_ids = encoded_messages[0][:max_source_len]
            target_ids = encoded_messages[1][:max_target_len]

            return (source_ids, target_ids)
        
    def preprocess_supervised_dataset(
        self,
        examples: dict[str, list[Any]],
        tokenizer: "PreTrainedTokenizer",
        cutoff_len: int,
    ) -> dict[str, list[list[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(examples["instruction"])):
            messages = [examples["instruction"][i], examples["output"][i]]
            
            input_ids, labels = [], []
            encoded_messages = []
            for i, message in enumerate(messages):
                elements = []
                if i == 0:
                    elements += self.template.format_prefix.apply()
                if i == 0:
                    elements += self.template.format_user.apply(content=message, idx=str(i // 2))
                elif i == 1:
                    elements += self.template.format_assistant.apply(content=message)
                
                encoded_messages.append(self._convert_elements_to_ids(tokenizer, elements))
                    
            source_ids, target_ids = self._make_pairs(encoded_messages, cutoff_len)
            source_label = [IGNORE_INDEX] * len(source_ids)
            input_ids += source_ids + target_ids
            labels += source_label + target_ids
            
            model_inputs["input_ids"].append(input_ids)  
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels) 
        return model_inputs