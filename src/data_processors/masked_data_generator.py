from datasets import load_dataset
import copy
import datetime
import pickle
from tqdm import tqdm

from data_objects.attribution_example import AttrToken
from data_objects.attribution_example import AttrExample

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
    task_keys = {"sst2" : {'text_a': 'sentence', 'label': 'label', 'guid': 'idx'},
                 "sst": {'text_a': 'sentence', 'label': 'label', 'guid': 'uid'},
                 "esnli": {'text_a': 'premise', 'text_b': 'hypothesis', 'label': 'label', 'guid': 'uid'},
                 "multirc" : ('paragraph', 'question', 'answer', 'idx', 'label'),
                 }

    return task_keys[task_name]


def generate_masked_examples(example, task_name, labels, mask):

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
            # List of AttrTokens parsed for a paragraph
            parsed_paragraph = copy.deepcopy(paragraph_cache_for_multirc[paragraph_cache_key])
        else:
            doc = nlp(example[paragraph]) # Parse paragraph first for caching
            for token in doc:
                attr_token = AttrToken(token=token.text, score=0)
                parsed_paragraph.append(attr_token)
            paragraph_cache_for_multirc[paragraph_cache_key] = copy.deepcopy(parsed_paragraph)

        if question_cache_key in question_cache_for_multirc:
            # List of AttrTokens parsed for a paragraph
            parsed_question = copy.deepcopy(question_cache_for_multirc[question_cache_key])
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
def general_masked_datasets(output_dir, mode="train", multiple_tasks=False):
    glue = ['sst2']
    superglue = ['multirc']
    others = ['sst', 'esnli']

    # Use for saving masked examples of multiple tasks to file
    data_sets = {}

    for idx, task_name in enumerate(glue + superglue + others):
        data_set, labels = [], []
        split = 3 if task_name == "multirc" else 1

        group = "glue"
        if idx >= len(glue):
            group = "super_glue"
        if idx >= len(glue) + len(superglue):
            group = "others"

        print("********** " + task_name + " **********")
        dataset = load_dataset(group, task_name) if group in ["glue", "super_glue"] else load_dataset(task_name)
        if mode == "train":
            dataset = dataset['train']
        else:
            dataset = dataset['validation'] if 'validation' in dataset else dataset['val']
        examples = dataset[:]

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
        for i in tqdm(range(split + 1)):  # split+1 for handling the remaining examples

            start, end = i*batch_size, (i+1)*batch_size
            sub_examples = new_examples[start:end] if i < split else new_examples[start:]

            if len(sub_examples) > 0:
                for idx2, example in tqdm(enumerate(sub_examples)):
                    example_dict = generate_masked_examples(example, task_name, labels, mask="[MASK]")
                    data_set.append(example_dict)

                # Thang's note 10/23/2021
                # key for SST-2 task is 'sst2' because it's used to load this dataset from Huggingface
                # but the code loading the pickle file in run_analyzer.py is using 'sst-2'
                key = task_name if split == 1 else task_name + "_split{}".format(i)
                data_sets[key] = data_set

                if not multiple_tasks:
                    if task_name == "sst2":
                        key = key.replace("sst2", "sst-2")

                    with open(output_dir + "masked_examples_" + mode + "_" + key + ".pickle", "wb") as file_path:
                        pickle.dump(data_sets, file_path)
                        data_set.clear()

    # Write input examples to files for GLUE + ESNLI and MultiRC
    # i.e., examples of multiple tasks in ONE pickle file.
    if multiple_tasks:
        with open(output_dir + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + "_" + "masked_examples.pickle", "wb") as file_path:
            pickle.dump(data_sets, file_path)


if __name__ == '__main__':
    output_dir = "../../data/pickle_files/masked_examples/"
    paragraph_cache_for_multirc = {}
    question_cache_for_multirc = {}

    general_masked_datasets(output_dir=output_dir, mode="train", multiple_tasks=False)
    general_masked_datasets(output_dir=output_dir, mode="dev", multiple_tasks=False)

