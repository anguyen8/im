from os import makedirs, mkdir, listdir
from os.path import join, isfile, exists

import pickle
import csv
import re

from datasets import load_dataset
from collections import Counter
from tqdm import tqdm
import json

import spacy
nlp = spacy.load("en_core_web_sm")


class Example(object):
    def __init__(self, id, text_a, text_b, label):
        self.id = id
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

        '''
        Example for highlights:
        self.highlights["text_a"] = [[(0, 'What'), (1, 'causes'), (2, 'a'), (3, 'change'), (4, 'in'), (5, 'motion'), (6, '?')],
                                     [(7, 'The'), (8, 'application'), (9, 'of'), (10, 'a'), (11, 'force'), (12, '.')],
                                     .......]
        '''
        self.highlights = {"text_a": [], "text_b": []}
        self.sentences = []

        self.processed_length = 0   # Count # tokens in paragraph (questions and answers are included)
        self.split = 0              # 1: train / 3: dev / 2: test


def parse_paragraph(text):

    sentences = text.strip().split("<br>")
    processed_sentences = []
    
    for sentence in sentences:
        if sentence:
            sentence = sentence.split("</b>")[1]
            processed_sentences.append(sentence.strip().replace("  ", " "))

    return processed_sentences


def get_hf_unique_paragraphs():

    hf_dataset = load_dataset("super_glue", "multirc")
    hf_dataset = hf_dataset['validation']
    hf_examples = hf_dataset[:]

    new_examples = []
    all_values = [hf_examples[key] for key in hf_examples.keys()]
    for idx in range(len(all_values[0])):
        input_dict = {}
        for idx2, key in enumerate(hf_examples.keys()):
            input_dict[key] = all_values[idx2][idx]
        new_examples.append(input_dict)

    hf_paragraphs = []
    for example in new_examples:
        paragraph = example["paragraph"].strip().replace("  ", " ")
        if paragraph not in hf_paragraphs:
            hf_paragraphs.append(paragraph)

    return hf_paragraphs


if __name__ == '__main__':

    data_path = "../datasets/multirc/"
    examples = []
    id_counter = 0

    list_a = ["word1", "word2", "word3", "word4", "word5"]
    list_b = ["word2", "word5", "word4", "word1", "word3"]

    hf_paragraphs = get_hf_unique_paragraphs()

    all_file_names = listdir(data_path)
    for file_name in all_file_names:
        if not isfile(data_path + file_name):
            continue

        input_file = open(data_path + file_name, "r")

        if file_name == "dev_83-fixedIds.json":
            dev_set = json.loads(next(input_file))["data"]

            paragraphs = [" ".join(parse_paragraph(entry["paragraph"]["text"].strip())) for entry in dev_set]
            sorted_indices = [hf_paragraphs.index(x.strip()) for x in paragraphs]
            dev_set = [dev_set[sorted_indices.index(i)] for i in range(len(sorted_indices))]

            for entry in dev_set:
                sentences = parse_paragraph(entry["paragraph"]["text"])
                tokenized_sentences = []
                counter = 0
                for sentence in sentences:
                    doc = nlp(sentence)  # Parse paragraph first for caching
                    tokenized_sentences.append([(idx + counter, token.text) for idx, token in enumerate(doc) if token.text != " "])
                    # tokenized_sentences.append([idx + counter for idx, token in enumerate(doc) if token.text != " "])
                    counter += len(tokenized_sentences[-1])

                questions = entry["paragraph"]["questions"]
                for question in questions:
                    question_text = question["question"]
                    highlights = question["sentences_used"]
                    answers = question["answers"]
                    for answer in answers:
                        answer_text = answer["text"]
                        label = 1 if answer["isAnswer"] else 0

                        example = Example(id=id_counter, text_a=" ".join(sentences + [question_text]), text_b=answer_text, label=label)
                        [example.highlights["text_a"].extend(tokenized_sentences[sent_id]) for sent_id in highlights]
                        example.sentences = sentences
                        example.split = 3  # For Dev set only

                        doc = nlp(" ".join([question_text, answer_text]))
                        example.processed_length = counter + len([token.text for token in doc if token.text != " "])
                        examples.append(example)

                        id_counter += 1

        input_file.close()

    pickle_fp = "../pickle_files/"
    if not exists(pickle_fp):
        makedirs(pickle_fp)

    with open(pickle_fp + "multirc_preprocessed.pickle", "wb") as file_path:
        pickle.dump(examples, file_path)

    print("Complete! There are {} examples.".format(len(examples)))


'''

'''