from functools import partial
from datasets import load_from_disk, load_dataset, concatenate_datasets
from datasets.io.json import JsonDatasetWriter
from lm_eval.tasks.hellaswag.utils import preprocess as HellaSwag_preprocess


def get_data_path(task_name):
    if task_name == "arc_easy":
        raw_data_path = "./data/arc/ARC-Easy"
    elif task_name == "arc_challenge":
        raw_data_path = "./data/arc/ARC-Challenge"
    elif task_name in ["boolq", "hellaswag", "openbookqa", "winogrande", "piqa", "race", "anli", "siqa", "gsm8k", "logiqa", "sciq", "CodeAlpaca20K"
                       , "alpaca_cleaned"]:
        raw_data_path = f"./data/{task_name}"
    elif task_name in ["rte", "mrpc", "cola", "sst2", "qnli", "qqp", "mnli"]:
        raw_data_path = f"./data/glue/{task_name}"
    elif task_name in ["copa", "multirc", "record", "cb", "wic", "wsc"]:
        raw_data_path = f"./data/superglue/{task_name}"
    elif task_name == "triviaqa":
        raw_data_path = "./data/trivia_qa"
    elif task_name in ["bigbench_abstract_narrative_understanding", "bigbench_goal_step_wikihow", "bigbench_social_iqa", "bigbench_timedial"]:
        raw_data_path = f"./data/bigbench/{task_name.replace('bigbench_', '')}"
    elif task_name in ["mmlu_high_school_government_and_politics", "mmlu_high_school_world_history", "mmlu_professional_law",
                       "mmlu_us_foreign_policy", "mmlu_business_ethics", "mmlu_high_school_computer_science", "mmlu_auxiliary_train"]:
        raw_data_path = f"./data/mmlu/{task_name.replace('mmlu_', '')}"
    else:
        raise NotImplementedError
    
    return raw_data_path

def load_train_dataset(task_name):
    if task_name.startswith('mmlu_'):
        dataset_train = load_from_disk(get_data_path(task_name))
    elif task_name in ('wic', 'wsc'):
        dataset = load_dataset(get_data_path(task_name), data_files={'train': 'train.jsonl', 'validation': 'val.jsonl'})
        dataset_train = dataset["train"]
    elif task_name == 'anli':
        dataset_train = load_from_disk(get_data_path(task_name))["train_r1"]
    elif task_name == 'arc_mix':
        arc_easy_train = load_from_disk(get_data_path('arc_easy'))["train"]
        arc_challenge_train = load_from_disk(get_data_path('arc_challenge'))["train"]
        dataset_train = concatenate_datasets([arc_easy_train, arc_challenge_train])
    else:
        dataset_train = load_from_disk(get_data_path(task_name))["train"]
    return dataset_train

def process_function(task_name, examples):
    if task_name == "rte":
        examples["instruction"] = examples["sentence1"]
        examples["input"] = [f"Question: {doc} True or False?\nAnswer:" for doc in examples["sentence2"]]
        # result["output"] = ["True" if examples["label"] == 0 else "False"]
        examples["output"] = [" True" if label == 0 else " False" for label in examples["label"]]
        del examples["sentence1"]; del examples["sentence2"]; del examples["label"]
    elif task_name in ["arc_easy", "arc_challenge"]:
        examples["instruction"] = ["Question: " + doc + "\nAnswer:" for doc in examples["question"]]
        letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        examples["answerIdx"] =[letter_to_index.get(i, i) for i in examples["answerKey"]]
        examples["output"] = [' '+choices['text'][idx] for choices, idx in zip(examples['choices'], examples["answerIdx"])]
        del examples["question"]; del examples["choices"]; del examples["answerKey"]; del examples["answerIdx"]
    elif task_name == 'boolq':
        examples["input"] = [f"Question: {doc}?\nAnswer:" for doc in examples["question"]]
        examples["output"] = [" yes" if label else " no" for label in examples["answer"]]
        del examples["question"]; del examples["answer"]
    elif task_name == "hellaswag":
        examples['ctx'] = [ctx_a + " " + ctx_b.capitalize() for ctx_a, ctx_b in zip(examples["ctx_a"], examples["ctx_b"])]
        examples["input"] = [HellaSwag_preprocess(activity_label + ": " + ctx) for activity_label, ctx in zip(examples["activity_label"], examples['ctx'])]
        examples["output"] = [' '+HellaSwag_preprocess(endings[int(label)]) for endings, label in zip(examples["endings"], examples["label"])]
        del examples["activity_label"]; del examples["ctx_a"]; del examples["ctx_b"]; del examples["endings"]; del examples["label"]; del examples["ctx"]
        del examples["source_id"]; del examples["split"]; del examples["split_type"]
    elif task_name == "openbookqa":
        letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        examples["output"] = [' '+choices['text'][letter_to_index[label.strip()]] for choices, label in zip(examples['choices'], examples["answerKey"])]
        del examples['choices']; del examples['answerKey']
    elif task_name == "winogrande":
        examples["option"] = [option1 if answer == "1" else option2 for option1, option2, answer in zip(examples["option1"], examples["option2"], examples["answer"])]
        def partial_context(sentence, option):
            # Substitute the pronoun in the sentence with the specified option and ignore everything after.
            pronoun_loc = sentence.index("_")
            return sentence[:pronoun_loc] + option
        def partial_target(sentence):
            # The target is everything after the document specified pronoun.
            pronoun_loc = sentence.index("_") + 1
            return " " + sentence[pronoun_loc:].strip()
        examples["input"] = [partial_context(sentence, option) for sentence, option in zip(examples["sentence"], examples["option"])]
        examples["output"] = [partial_target(sentence) for sentence in examples["sentence"]]
        del examples["sentence"]; del examples["option"]; del examples["answer"]; del examples["option1"]; del examples["option2"]
    elif task_name == "piqa":
        examples["input"] = [f"Question: {doc}\nAnswer:" for doc in examples["goal"]]
        examples["sols"] = [[sol1, sol2] for sol1, sol2 in zip(examples["sol1"], examples["sol2"])]
        examples["output"] = [' ' + sols[label] for sols, label in zip(examples["sols"], examples["label"])]
        del examples["goal"]; del examples["sol1"]; del examples["sol2"]; del examples["label"]; del examples["sols"]
    else: raise NotImplementedError
    return examples


def main():
    # ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa"]
    task_name = "piqa"
    dataset_train = load_from_disk(get_data_path(task_name))["train"]
    dataset_train = dataset_train.map(partial(process_function, task_name), batched=True)

    # print(dataset_train)
    # dataset_train = dataset_train.filter(lambda x: len(x['output']) < 500)
    # print(dataset_train)
    output_data_path = f"./LLaMA-Factory/data/harness/{task_name}/train.json"
    JsonDatasetWriter(dataset_train, output_data_path).write()

if __name__ == '__main__':
    main()