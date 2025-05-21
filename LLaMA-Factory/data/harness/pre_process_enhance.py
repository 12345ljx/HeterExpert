from functools import partial
import os
from datasets import load_from_disk
from datasets.io.json import JsonDatasetWriter
import random
import json

from patterns import PATTERNS, PATTERNS_OPTIONS, PATTERNS_NO_OPTIONS
from pre_process import load_train_dataset
from pre_process_template import format_options


def update_example(example, *, task_name, template, use_char_options_format=False):
    if task_name in ("arc_easy", "arc_challenge", "arc_mix"):
        options = example["choices"]["text"]
        letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        answerIdx = letter_to_index.get(example["answerKey"], example["answerKey"])
        content = {
            "question": example['question'],
            "answer": options[answerIdx],
            "options_": format_options(options, use_char_options_format),
        }
    elif task_name == "copa":
        connector = {
            "cause": " because",
            "effect": " therefore",
        }
        example['premise'] = example['premise'].strip()
        if example['premise'].endswith('.'):
            glm_premise = example['premise'][:-1] + connector[example['question']]
        answer = example["choice1"] if example["label"] == 0 else example["choice2"]
        options = [example["choice1"], example["choice2"]]
        content = {
            "premise": example['premise'],
            "glm_premise": glm_premise,
            "question": example['question'],
            "answer": answer,
            "options_": format_options(options, use_char_options_format),
        }
    elif task_name == 'cb':
        options = ['True', 'False', 'Neither']
        content = {
            'premise': example['premise'],
            'hypothesis': example['hypothesis'],
            'answer': options[example['label']],
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "wsc":
        options = ['no', 'yes']
        content = {
            'context': example['text'],
            'text1': example['target']['span1_text'],
            'text2': example['target']['span2_text'],
            'answer': 'yes' if example['label'] else 'no',
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "openbookqa":
        letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        idx = letter_to_index[example["answerKey"].strip()]
        options = example['choices']['text']
        content = {
            "question": example['question_stem'],
            "fact": example['fact1'],
            "options_": format_options(options, use_char_options_format),
            "answer": options[idx],
        }
    else:
        raise NotImplementedError

        
    example["instruction"], example["output"] = [doc.format(**content) for doc in template]
    
    attrs_remove = [attr for attr in list(example.keys()) if attr not in ["instruction", "output"]]
    for attr in attrs_remove:
        del example[attr]
    
    # print("instruction:\n", example["instruction"], sep='')
    # print("output:\n", example["output"], sep='')
    
    return example

def main():
    random.seed(42)

    # ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa", "copa"]
    task_name = "arc_mix"
    dataset_train = load_train_dataset(task_name)

    patterns = PATTERNS_NO_OPTIONS
    templates = patterns['arc'] if task_name in ("arc_easy", "arc_challenge", "arc_mix") else patterns[task_name]

    for idx, template in enumerate(templates):
        dataset_train_tmp = dataset_train.map(partial(update_example, task_name=task_name, template=template, use_char_options_format=True))
        dataset_train_tmp = dataset_train_tmp.filter(lambda x: len(x['instruction'] + x['output']) < 3000)
        output_data_path = f"./LLaMA-Factory/data/harness/data_enhance/{task_name}_template_{idx}.json"
        JsonDatasetWriter(dataset_train_tmp, output_data_path).write()

    output_data_paths = [f"./LLaMA-Factory/data/harness/data_enhance/{task_name}_template_{idx}.json" for idx in range(len(templates))]
    all_data = []
    for file_path in output_data_paths:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            all_data.extend(lines)

    random.shuffle(all_data)

    output_data_path = f"./LLaMA-Factory/data/harness/{task_name}_template_enhance"
    if os.path.exists(output_data_path):
        raise ValueError("the data has been processed")
    os.makedirs(output_data_path, exist_ok=True)
    with open(os.path.join(output_data_path, "train.json"), 'w', encoding='utf-8') as file:
        file.writelines(all_data)
        
if __name__ == "__main__":
    main()