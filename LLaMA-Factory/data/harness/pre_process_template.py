import os
from functools import partial
from datasets.io.json import JsonDatasetWriter
from lm_eval.tasks.hellaswag.utils import preprocess as HellaSwag_preprocess
import random

from patterns import PATTERNS, PATTERNS_OPTIONS, PATTERNS_NO_OPTIONS
from pre_process import load_train_dataset


def format_options(options, use_char_options_format: bool = False):
    """Formats options."""
    if use_char_options_format:
        options_prefix = "OPTIONS:"
        separator = ""
        char_options = [f"\n({chr(x)}) " for x in range(ord("A"), ord("Z") + 1)]
        options = [char_options[i] + opt for i, opt in enumerate(options)]
    else:
        options_prefix = "OPTIONS:\n- "
        separator = "\n- "
    return options_prefix + separator.join(options)


def update_example(example, *, task_name, patterns=PATTERNS_NO_OPTIONS, use_char_options_format=False):
    if task_name in ("arc_easy", "arc_challenge", "arc_mix"):
        options = example["choices"]["text"]
        letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, '1': 0, '2': 1, '3': 2, '4': 3, '5': 4}
        answerIdx = letter_to_index.get(example["answerKey"], example["answerKey"])
        content = {
            "question": example['question'],
            "answer": options[answerIdx],
            "options_": format_options(options, use_char_options_format),
        }
    elif task_name == "rte":
        options = ["True", "False"]
        content = {
            "premise": example['sentence1'],
            "hypothesis": example['sentence2'],
            "options_": format_options(options, use_char_options_format),
            "answer": options[example['label']],
        }
    elif task_name == "boolq":
        options = ["no", "yes"]
        content = {
            "text": example["passage"],
            "question": example["question"],
            "options_": format_options(options, use_char_options_format),
            "answer": "yes" if example["answer"] else "no",
        }
    elif task_name == "hellaswag":
        ctx = example["ctx_a"] + " " + example["ctx_b"].capitalize()
        options = [HellaSwag_preprocess(i) for i in example["endings"]]
        content = {
            "context": HellaSwag_preprocess(example["activity_label"] + ": " + ctx),
            "options_": format_options(options, use_char_options_format),
            "answer": options[int(example["label"])],
        }
    elif task_name == "winogrande":
        sentence = example["sentence"]
        context = sentence[:sentence.index("_")]
        next_sentence = sentence[sentence.index("_") + 1:]
        options = [example["option1"], example["option2"]]
        options = [option + next_sentence for option in options]
        content = {
            "context": context,
            "options_": format_options(options, use_char_options_format),
            "answer": options[int(example["answer"]) - 1],
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
    elif task_name == "piqa":
        options = [example['sol1'], example['sol2']]
        content = {
            "goal": example['goal'],
            "options_": format_options(options, use_char_options_format),
            "answer": options[example['label']],
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
    elif task_name == "multirc":
        options = ['no', 'yes']
        content = {
            'paragraph': example['paragraph'],
            'question': example['question'],
            'response': example['answer'],
            'answer': "yes" if example['label'] else "no",
            "options_": format_options(options, use_char_options_format),
            'glm_answer': "True" if example['label'] else "False",
        }
    elif task_name == "record":
        initial_text, *highlights = example["passage"].strip().split("\n@highlight\n")
        passage = initial_text + "\n\n"
        for highlight in highlights:
            passage += f"  - {highlight}.\n"
        query_left, query_right = example['query'].split("@placeholder")
        answer = example["answers"][0]
        options = [option + query_right for option in example['entities']]
        right_answer = answer + query_right
        whole_answer = example['query'].replace("@placeholder", answer)
        content = {
            'answer': right_answer,
            'whole_answer': whole_answer,
            'passage': passage,
            'query': query_left,
            'options_str': format_options(options, use_char_options_format=False),  # Some questions have too many options
        }
    elif task_name == "triviaqa":
        content = {
            'question': example['question'],
            'answer': example['answer']['value'],
        }
    elif task_name == "race":
        letter_to_index = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        content = {
            'article': example['article'],
            'question': example['question'],
            'answer': example['options'][letter_to_index[example['answer']]],
        }
    elif task_name == "mrpc":
        options = ['no', 'yes']
        content = {
            'sentence1': example['sentence1'],
            'sentence2': example['sentence2'],
            'answer': 'yes' if example['label'] else 'no',
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "qqp":
        options = ['no', 'yes']
        content = {
            'question1': example['question1'].replace('""', '\''),
            'question2': example['question2'].replace('""', '\''),
            'answer': 'yes' if example['label'] else 'no',
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "cola":
        options = ['no', 'yes']
        content = {
            'sentence': example['sentence'],
            'answer': 'yes' if example['label'] else 'no',
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "sst2":
        options = ['negative', 'positive']
        content = {
            'sentence': example['sentence'],
            'answer': 'positive' if example['label'] else 'negative',
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "mnli":
        options = ["True", "Neither", "False"]
        content = {
            'premise': example['premise'],
            'hypothesis': example['hypothesis'],
            'answer': options[example['label']],
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "qnli":
        options = ['yes', 'no']
        content = {
            'sentence': example['sentence'],
            'question': example['question'],
            'answer': 'no' if example['label'] else 'yes',
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "cb":
        options = ['True', 'False', 'Neither']
        content = {
            'premise': example['premise'],
            'hypothesis': example['hypothesis'],
            'answer': options[example['label']],
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "wic":
        options = ['no', 'yes']
        content = {
            'sentence1': example['sentence1'],
            'sentence2': example['sentence2'],
            'word': example['word'],
            'answer': 'yes' if example['label'] else 'no',
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
    elif task_name == "anli":
        options = ['True', 'Neither', 'False']
        content = {
            'context': example['premise'],
            'hypothesis': example['hypothesis'],
            'answer': options[example['label']],
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "siqa":
        options = [example['answerA'], example['answerB'], example['answerC']]
        content = {
            'context': example['context'],
            'question': example['question'],
            'answer': options[int(example['label']) - 1],
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "gsm8k":
        content = {
            'question': example['question'],
            'answer': example['answer'],
        }
    elif task_name == "logiqa":
        options = example['options']
        content = {
            'context': example['context'],
            'question': example['query'],
            'answer': options[example['correct_option']],
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name == "sciq":
        answer = example['correct_answer']
        options = [example['distractor1'], example['distractor2'], example['distractor3'], answer]
        random.shuffle(options)
        content = {
            'support': example['support'].lstrip(),
            'question': example['question'],
            'answer': answer,
            'options_': format_options(options, use_char_options_format),
        }
    elif task_name in ["bigbench_abstract_narrative_understanding", "bigbench_goal_step_wikihow", "bigbench_social_iqa", "bigbench_timedial"]:
        content = {
            'inputs': example['inputs'],
            'answer': example['targets'][0],
        }
    elif task_name in ["mmlu_professional_law", "mmlu_high_school_world_history", "mmlu_high_school_government_and_politics",
                       "mmlu_us_foreign_policy", "mmlu_business_ethics", "mmlu_high_school_computer_science", "mmlu_auxiliary_train"]:
        content = {
            'question': example['question'],
            'answer': example['choices'][example['answer']],
        }
    elif task_name in ["CodeAlpaca20K", "alpaca-cleaned"]:
        PROMPT_DICT = {
            "prompt_input": (
                "Below is an instruction that describes a task, paired with an input that provides further context. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
            ),
            "prompt_no_input": (
                "Below is an instruction that describes a task. "
                "Write a response that appropriately completes the request.\n\n"
                "### Instruction:\n{instruction}\n\n### Response:"
            ),
        }
        if example['input'] != '':
            example['instruction'] = PROMPT_DICT["prompt_input"].format(instruction=example['instruction'], input=example['input'])
        else:
            example['instruction'] = PROMPT_DICT["prompt_no_input"].format(instruction=example['instruction'])
            
        content = {
            'question': example['instruction'],
            'answer': example['output'],
        }
    else:
        raise NotImplementedError
    
    if task_name in ("arc_easy", "arc_challenge", "arc_mix"):
        template = random.choice(patterns['arc'])
    else:
        template = random.choice(patterns[task_name])
        
    example["instruction"], example["output"] = [doc.format(**content) for doc in template]
    
    attrs_remove = [attr for attr in list(example.keys()) if attr not in ["instruction", "output"]]
    for attr in attrs_remove:
        del example[attr]
    
    # print("instruction:\n", example["instruction"], sep='')
    # print("output:\n", example["output"], sep='')
    
    return example



def main():
    random.seed(42)
    
    # ["boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa", "piqa", "copa", "record", "triviaqa", "race",
    # "mrpc", "cola", "sst2", "qnli"]
    task_name = "arc_mix"
    dataset_train = load_train_dataset(task_name)

    dataset_train = dataset_train.map(partial(update_example, task_name=task_name, patterns=PATTERNS_NO_OPTIONS, use_char_options_format=False))
    l1 = len(dataset_train)
    dataset_train = dataset_train.filter(lambda x: len(x['instruction'] + x['output']) < 3000)
    l2 = len(dataset_train)
    print('{}% of the data was filtered'.format((l1 - l2) / l1 * 100))
    
    output_data_path = f"./LLaMA-Factory/data/harness/{task_name}_template/train.json"
    if os.path.exists(output_data_path):
        raise ValueError("the data has been processed")
    JsonDatasetWriter(dataset_train, output_data_path).write()

if __name__ == "__main__":
    main()