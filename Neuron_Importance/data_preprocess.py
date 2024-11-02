from functools import partial
from typing import Sequence, Tuple, Any, Union
from transformers import PreTrainedTokenizer
from datasets import Dataset

IGNORE_INDEX = -100

def infer_max_len(source_len: int, target_len: int, max_len: int, reserved_label_len: int) -> Tuple[int, int]:
    max_target_len = int(max_len * (target_len / (source_len + target_len)))
    max_target_len = max(max_target_len, reserved_label_len)
    max_source_len = max_len - max_target_len
    return max_source_len, max_target_len

class DataProcessor:
    def __init__(self, dataset: Dataset, tokenizer, max_len=1024):
        dataset = dataset.add_column("sequence_length", [len(x + y) for x, y in zip(dataset['instruction'], dataset['output'])])
        dataset = dataset.sort("sequence_length", reverse=True)
        
        column_names = list(next(iter(dataset)).keys())
        preprocess_func = partial(self.preprocess_supervised_dataset, tokenizer=tokenizer, cutoff_len=max_len)
        self._dataset = dataset.map(preprocess_func, batched=True, remove_columns=column_names)
    
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
            cutoff_len: int,
            reserved_label_len: int,
        ) -> Sequence[Tuple[list[int], list[int]]]:
            max_source_len, max_target_len = infer_max_len(
                source_len=len(encoded_messages[0]),
                target_len=len(encoded_messages[1]),
                max_len=cutoff_len,
                reserved_label_len=reserved_label_len,
            )
            source_ids = encoded_messages[0][:max_source_len]
            target_ids = encoded_messages[1][:max_target_len]

            return (source_ids, target_ids)
        
    def preprocess_supervised_dataset(
        self,
        examples: dict[str, list[Any]],
        tokenizer: "PreTrainedTokenizer",
        cutoff_len: int,
        reserved_label_len: int = 1,
    ) -> dict[str, list[list[int]]]:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        for i in range(len(examples["instruction"])):
            messages = [examples["instruction"][i], examples["output"][i]]
            input_ids, labels = [], []
            encoded_messages = []
            for message in messages:
                encoded_messages.append(self._convert_elements_to_ids(tokenizer, message))
            source_ids, target_ids = self._make_pairs(encoded_messages, cutoff_len, reserved_label_len)
            source_mask = [IGNORE_INDEX] * len(source_ids)
            input_ids += source_ids + target_ids
            labels += source_mask + target_ids
            model_inputs["input_ids"].append(input_ids)  
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels) 
        return model_inputs