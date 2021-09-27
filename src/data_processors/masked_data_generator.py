from datasets import load_dataset, load_metric
import ast
import copy
import datetime
import pickle
from tqdm import tqdm

from attribution_example import AttrToken
from attribution_example import AttrExample
import spacy
nlp = spacy.load("en_core_web_sm")

import time
from functools import wraps


def my_timer(my_func):

    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()
        print('"{}" took {:.3f} s to execute\n'.format(my_func.__name__, (tend - tstart)))
        return output

    return timed


def get_task_keys(task_name):
    task_keys = {"cola" : {'text_a': 'sentence', 'label': 'label', 'guid': 'idx'},
                 "sst2" : {'text_a': 'sentence', 'label': 'label', 'guid': 'idx'},
                 "mrpc" : {'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'guid': 'idx'},
                 "qqp" : {'text_a': 'question1', 'text_b': 'question2', 'label': 'label', 'guid': 'idx'},
                 "stsb" : {'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'guid': 'idx'},
                 "mnli_mismatched" : {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'idx'},
                 "mnli_matched" : {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'idx'},
                 "qnli" : {'text_a': 'question', 'text_b': 'sentence', 'label': 'label', 'guid': 'idx'},
                 "rte_g" : {'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'guid': 'idx'},
                 "wnli" : {'text_a': 'sentence1', 'text_b': 'sentence2', 'label': 'label', 'guid': 'idx'},

                 "anli": {'text_a': 'text_a', 'text_b': 'text_b', 'label': 'label', 'guid': 'guid'},
                 "r1": {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'uid'},
                 "r2": {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'uid'},
                 "r3": {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'uid'},

                 "sst": {'text_a': 'sentence', 'label': 'label', 'guid': 'uid'},
                 "esnli": {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'uid'},

                 # The below tasks (except for SNLI) are handled directly in their corresponding projects:
                 # jiant for SuperGLUE and transformer for SQuADv1-2
                 # SNLI will be handled with the above tasks.

                 "boolq" : ['question', 'passage', 'idx', 'label'],
                 "cb" : ['premise', 'hypothesis', 'idx', 'label'],
                 "copa" : ['premise', 'choice1', 'choice2', 'question', 'idx', 'label'],
                 "multirc" : ('paragraph', 'question', 'answer', 'idx', 'label'),
                 "record" : ['passage', 'query', 'entities', 'answers', 'idx'],
                 "rte_sg": ['premise', 'hypothesis', 'idx', 'label'],
                 "wic" : ['word', 'sentence1', 'sentence2', 'start1', 'start2', 'end1', 'end2', 'idx', 'label'],
                 "wsc" : ['text', 'span1_index', 'span2_index', 'span1_text', 'span2_text', 'idx', 'label'],

                 "snli" : {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label'},
                 "squad" : ['id', 'title', 'context', 'question', 'answers'],
                 "squad_v2" : ['id', 'title', 'context', 'question', 'answers']
                 }

    return task_keys[task_name]


def generate_masked_examples(example, task_name, labels, mask): # Type == "occlusion" or "ris"
    glue = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli']
    tasks_support = glue + ['snli'] + ['anli', 'r1', 'r2', 'r3']

    if task_name != "multirc":
        task_keys = get_task_keys((task_name + "_g") if task_name == "rte" else task_name)
        text_a = example[task_keys["text_a"]]
        text_b = example[task_keys["text_b"]] if "text_b" in task_keys else ""
        label = labels[int(example[task_keys["label"]])] if len(labels) > 0 else str(example[task_keys["label"]])
        guid = example[task_keys["guid"]] if "guid" in task_keys and task_keys["guid"] in example else ""

        ori_attr_example = AttrExample(text_a=text_a, text_b=text_b, label=label, guid=guid)
    else:
        paragraph, question, answer, idx, label = get_task_keys(task_name)
        text_a = example[paragraph] + " " + example[question]
        text_b = example[answer]
        label = str(example[label])
        guid = str(example[idx])

        # --------------------------------------------------------
        # Handling parsed_text_a here for boosting the progress.
        # --------------------------------------------------------
        paragraph_cache_key = str(example[idx][paragraph])
        question_cache_key = str(example[idx][question])
        parsed_paragraph, parsed_question = [], []

        if paragraph_cache_key in paragraph_cache_for_multirc:
            parsed_paragraph = copy.deepcopy(paragraph_cache_for_multirc[paragraph_cache_key]) # List of AttrTokens parsed for a paragraph
        else:
            doc = nlp(example[paragraph]) # Parse paragraph first for caching
            for token in doc:
                attr_token = AttrToken(token=token.text, score=0)
                parsed_paragraph.append(attr_token)
            paragraph_cache_for_multirc[paragraph_cache_key] = copy.deepcopy(parsed_paragraph)

        if question_cache_key in question_cache_for_multirc:
            parsed_question = copy.deepcopy(question_cache_for_multirc[question_cache_key]) # List of AttrTokens parsed for a paragraph
        else:
            doc = nlp(example[question]) # Parse question first for caching
            for token in doc:
                attr_token = AttrToken(token=token.text, score=0)
                parsed_question.append(attr_token)
            question_cache_for_multirc[question_cache_key] = copy.deepcopy(parsed_question)

        parsed_text_a = parsed_paragraph + parsed_question
        # --------------------------------------------------------

        ori_attr_example = AttrExample(text_a=text_a, text_b=text_b, label=label, guid=guid, parsed_text_a=parsed_text_a)

    generated_list = []
    for idx in range(ori_attr_example.get_length()):
        masked_attr_example = copy.deepcopy(ori_attr_example)
        masked_attr_example.set_masked_token(idx, mask=mask)
        generated_list.append(masked_attr_example)

    return {"ori_example": ori_attr_example,
            "masked_examples": generated_list,
            "total_len": len(generated_list) + 1}


@my_timer
def general_masked_datasets(output_dir, mode="train", multiple_tasks=False, split=1):
    glue = ['cola', 'sst2', 'mrpc', 'qqp', 'stsb', 'mnli_mismatched', 'mnli_matched', 'qnli', 'rte', 'wnli']
    superglue = ['boolq', 'cb', 'copa', 'multirc', 'record', 'rte', 'wic', 'wsc']
    stanford = ['snli', 'squad', 'squad_v2']
    anli = ['anli', "r1", "r2", "r3"]
    others = ['sst', 'esnli']

    # Use for saving masked examples of multiple tasks to file
    data_sets = {}

    for idx, task_name in enumerate(glue + superglue + stanford + anli + others):
        # For MiniQ1 ONLY
        # if task_name not in ["cola", "rte", "mrpc", "sst2", "qqp", "qnli", "stsb"]:
        if task_name not in ['multirc']:
            continue

        data_set, labels = [], []

        group = "glue"
        if idx >= len(glue):
            group = "super_glue"
        if idx >= len(glue) + len(superglue):
            group = "stanford"
        if idx >= len(glue) + len(superglue) + len(stanford):
            group = "anli"
        if idx >= len(glue) + len(superglue) + len(stanford) + len(anli):
            group = "others"

        if task_name in anli:
            new_examples = []
            input_dir = "datasets/smfa_nli/anli/"
            file_name = "dev" #"test"

            if task_name == "anli":
                file_path = input_dir + "full_unfiltered/" + file_name + ".txt"
            elif task_name == "r1":
                file_path = input_dir + "r1/" + file_name + ".jsonl"
            elif task_name == "r2":
                file_path = input_dir + "r2/" + file_name + ".jsonl"
            elif task_name == "r3":
                file_path = input_dir + "r3/" + file_name + ".jsonl"

            input_file = open(file_path, "r")
            for line in input_file:
                example = ast.literal_eval(line.strip())
                new_examples.append(example)
            input_file.close()
        else:
            # if task_name not in anli:
            #     continue

            print("********** " + task_name + " **********")
            dataset = load_dataset(group, task_name) if group in ["glue", "super_glue"] else load_dataset(task_name)
            if mode == "train":
                dataset = dataset['train']
            else:
                dataset = dataset['validation'] if 'validation' in dataset else dataset['val']
            examples = dataset[:]

            # ThangPM's NOTE 07-30-20
            # Only these tasks need to convert labels from integers to text
            if task_name in ["mnli_matched", "mnli_mismatched", "qnli", "rte", "snli"]:
                labels = dataset.features['label'].names

            if task_name == "squad_v2":
                print("Number of negative examples of SQuADv2: " + str(sum([len(example['text']) == 0 for example in examples['answers']])))

            # ThangPM's NOTE 07-28-20
            # Convert a dictionary (examples) with lists to a list of
            # dictionaries (new_examples) corresponding to each example.
            new_examples = []
            all_values = [examples[key] for key in examples.keys()]
            for idx in range(len(all_values[0])):
                input_dict = {}
                for idx2, key in enumerate(examples.keys()):
                    input_dict[key] = all_values[idx2][idx]
                new_examples.append(input_dict)

        batch_size = int(len(new_examples) / split)
        for i in tqdm(range(split + 1)):  # +1 for handling the remaining examples

            if i != 1:
                continue

            start, end = i*batch_size, (i+1)*batch_size
            sub_examples = new_examples[start:end] if i < split else new_examples[start:]

            if len(sub_examples) > 0:
                for idx2, example in tqdm(enumerate(sub_examples)):

                    # FOR SQUADv2 ONLY
                    # Retain only adversarial examples as discussed with Dr. Nguyen
                    if task_name == "squad_v2" and len(example["answers"]["text"]) > 0:
                        continue

                    example_dict = generate_masked_examples(example, task_name, labels, mask="[MASK]")
                    data_set.append(example_dict)

                key = task_name if split == 1 else task_name + "_split{}".format(i)
                data_sets[key] = data_set

                if not multiple_tasks:
                    with open(output_dir + "masked_examples_" + mode + "_" + key + ".pickle", "wb") as file_path:
                        pickle.dump(data_sets, file_path)
                        data_set.clear()

    # Write input examples to files for GLUE + SNLI
    if multiple_tasks:
        with open(output_dir + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + "masked_examples.pickle", "wb") as file_path:
            pickle.dump(data_sets, file_path)


if __name__ == '__main__':
    output_dir = ""
    paragraph_cache_for_multirc = {}
    question_cache_for_multirc = {}

    # general_masked_datasets(output_dir=output_dir, mode="train", multiple_tasks=False, split=1)
    general_masked_datasets(output_dir=output_dir, mode="dev", multiple_tasks=False, split=3)

