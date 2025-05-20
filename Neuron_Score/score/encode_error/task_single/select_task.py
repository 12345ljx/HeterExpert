import random
from pathlib import Path

def get_task_list():
    directory = Path('/usr/workdir/HeterExpert/data')
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    task_list = [task_path.name.replace('_template', '') for task_path in subdirs]
    return sorted(task_list)
    
def main():
    random.seed(42)
    task_list = get_task_list()
    task_select = random.sample(task_list, 8)
    print(task_select)  # ['siqa', 'boolq', 'anli', 'hellaswag', 'gsm8k', 'sst2', 'cb', 'winogrande']

if __name__ == '__main__':
    main()