import random
from enum import Enum
from pathlib import Path
import numpy as np
from tqdm import tqdm
from gen_domains_cluster import merge_domain_data, read_examples

class Type(Enum):
    Textual_Entailment = 1
    Knowledge_Question_Answering = 2
    Acceptability = 3
    Commonsense_Reasoning = 4
    Paraphrase = 5
    Reading_Comprehension = 6
    Coreference_Resolution = 7
    Sentiment = 8
    

Task_Type = {
    'anli': Type.Textual_Entailment,
    'arc_challenge': Type.Knowledge_Question_Answering,
    'arc_easy': Type.Knowledge_Question_Answering,
    'boolq': Type.Knowledge_Question_Answering,
    'cb': Type.Textual_Entailment,
    'cola': Type.Acceptability,
    'copa': Type.Commonsense_Reasoning,
    'gsm8k': Type.Knowledge_Question_Answering,
    'hellaswag': Type.Commonsense_Reasoning,
    'logiqa': Type.Reading_Comprehension,
    'mnli': Type.Textual_Entailment,
    'mrpc': Type.Paraphrase,
    'multirc': Type.Reading_Comprehension,
    'openbookqa': Type.Reading_Comprehension,
    'piqa': Type.Commonsense_Reasoning,
    'qnli': Type.Textual_Entailment,
    'qqp': Type.Paraphrase,
    'record': Type.Commonsense_Reasoning,
    'rte': Type.Textual_Entailment,
    'sciq': Type.Knowledge_Question_Answering,
    'siqa': Type.Commonsense_Reasoning,
    'sst2': Type.Sentiment,
    'triviaqa': Type.Knowledge_Question_Answering,
    'wic': Type.Paraphrase,
    'winogrande': Type.Coreference_Resolution,
    'wsc': Type.Coreference_Resolution,
}

def main():
    random_seed = 42
    random.seed(random_seed)
    
    domains_num = 8
    sample_count = 5000
    domains_data = [[] for _ in range(domains_num)]
    domains_compose = np.zeros((domains_num, len(Task_Type)), dtype=np.int32)
    
    directory = Path('./Cluster/instruct_data')
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    for task_path in tqdm(subdirs):
        task_name = task_path.name.replace('_template', '')
        if task_name == 'mix_data': continue
        dataset = read_examples(task_path)
        sample_count_task = min(sample_count, len(dataset))
        dataset = dataset.shuffle(seed=random_seed).select(range(sample_count_task))
        
        data_for_domain = dataset.add_column("task", [task_name] * dataset.num_rows)
        domain_idx = Task_Type[task_name].value - 1
        domains_data[domain_idx].append(data_for_domain)
        domains_compose[domain_idx][list(Task_Type).index(task_name)] += data_for_domain.num_rows
        
    domains_data_merge = merge_domain_data(domains_data)
    print('Domain Examples Number:', [domain.num_rows for domain in domains_data_merge])
    
    for idx, domain in enumerate(domains_data_merge):
        domain_path = f'./Cluster/domains/task_type/domain{idx}'
        domain.save_to_disk(domain_path)
    np.save('./Cluster/domains/task_type/domains_compose.npy', domains_compose)
    
if __name__ == '__main__':
    main()