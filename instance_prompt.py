import os
import random
from functools import partial
from datasets import load_from_disk, load_dataset
from datasets.io.json import JsonDatasetWriter

import sys
sys.path.append('/usr/workdir/LLaMA-Factory/data/harness')
from pre_process import load_train_dataset
from pre_process_template import update_example
from patterns import PATTERNS_OPTIONS

def main():
    random.seed(42)
    
    task_name = "triviaqa"
    dataset_train = load_train_dataset(task_name)

    dataset_train = dataset_train.map(partial(update_example, task_name=task_name, patterns=PATTERNS_OPTIONS, use_char_options_format=True))
    l1 = len(dataset_train)
    dataset_train = dataset_train.filter(lambda x: len(x['instruction'] + x['output']) < 3000)
    l2 = len(dataset_train)
    print('{}% of the data was filtered'.format((l1 - l2) / l1 * 100))
    
    output_data_path = f"/usr/workdir/HeterExpert/data/{task_name}_template/train.json"
    if os.path.exists(output_data_path):
        raise ValueError("the data has been processed")
    JsonDatasetWriter(dataset_train, output_data_path).write()

if __name__ == "__main__":
    main()