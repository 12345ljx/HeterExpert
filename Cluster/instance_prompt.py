import os
import random
from functools import partial
from datasets.io.json import JsonDatasetWriter

from Cluster.patterns import PATTERNS_OPTIONS
from Cluster.prompts_tools import load_train_dataset, update_example

def main():
    random.seed(42)
    for task_name in ['anli', 'arc_challenge', 'arc_easy', 'boolq', 'cb', 'cola', 'copa', 'gsm8k', 'hellaswag', 'logiqa', 'mnli', 'mrpc', 'multirc', 'openbookqa', 
                      'piqa', 'qnli', 'qqp', 'record', 'rte', 'sciq', 'siqa', 'sst2', 'triviaqa', 'wic', 'winogrande', 'wsc']:
        dataset_train = load_train_dataset(task_name)
        dataset_train = dataset_train.map(partial(update_example, task_name=task_name, patterns=PATTERNS_OPTIONS, use_char_options_format=True))
        output_data_path = f"./Cluster/instruct_data/{task_name}_template/train.json"
        if os.path.exists(output_data_path):
            raise ValueError("the data has been processed")
        JsonDatasetWriter(dataset_train, output_data_path).write()

if __name__ == "__main__":
    main()