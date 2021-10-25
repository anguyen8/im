from os import makedirs, mkdir, listdir
from os.path import join, isfile, exists
import pickle
from tqdm import tqdm
from datasets import load_dataset
import string
import nltk
import difflib


class Phrase(object):
    def __init__(self, id, text, score):
        self.id = id
        self.text = text
        self.score = score


class Sentence(object):
    def __init__(self, id, text, score):
        self.id = id
        self.text = text
        self.score = score
        self.phrases = []
        self.split = 0 # 1: train / 3: dev / 2: test
        
    def add_phrase(self, id, text, score):
        phrase = Phrase(id, text, score)
        self.phrases.append(phrase)


def verify_sst2_label():
    datasets = load_dataset("glue", "sst2")
    sst2_dataset = [datasets["train"], datasets["validation"], datasets["test"]]

    sst2_examples = []
    sst2_labels = []
    for split in sst2_dataset:
        for sst2_example in split:
            sst2_examples.append(sst2_example["sentence"].lower())
            sst2_labels.append(sst2_example["label"])

    with open("../pickle_files/sst_preprocessed.pickle", "rb") as input_file:
        processed_dataset = pickle.load(input_file)

    positive, neutral, negative = 0, 0, 0
    for example in tqdm(processed_dataset):
        for phrase in example.phrases:
            if example.text.lower() in sst2_examples:
                if float(example.score) > 0.5:
                    positive += 1
                elif float(example.score) < 0.5:
                    negative += 1
                else:
                    neutral += 1
            elif phrase.text.lower() in sst2_examples:
                if float(example.score) > 0.5:
                    positive += 1
                elif float(example.score) < 0.5:
                    negative += 1
                else:
                    neutral += 1

    print("Positive = {}, Negative = {}, Neutral = {}".format(positive, negative, neutral))


def clarify_sst_and_sst2():
    data_path = "../datasets/sst/stanfordSentimentTreebank/"
    sentences, phrases = [], []
    all_file_names = listdir(data_path)

    # Get all sentences and phrases in SST raw dataset
    for file_name in all_file_names:
        input_file = open(data_path + file_name, "r")
        if file_name == "dictionary.txt":
            for line in input_file:
                phrase_text, phrase_id = line.strip().split("|")
                phrases.append(phrase_text.strip().lower())

    # Get all examples including sentences & phrases in SST-2
    datasets = load_dataset("glue", "sst2")
    sst2_dataset = [datasets["train"], datasets["validation"], datasets["test"]]

    # Get all sentences in SST
    datasets = load_dataset("sst")
    sst_dataset = [datasets["train"], datasets["validation"], datasets["test"]]

    sst2_examples = []
    for split in sst2_dataset:
        for sst2_example in split:
            sst2_examples.append(sst2_example["sentence"].strip().lower())

    sst_examples = []
    for split in sst_dataset:
        for sst_example in split:
            sst_examples.append(sst_example["sentence"].strip().lower())

    sent_counter, phrase_counter, outliner = 0, 0, 0
    for sst2_example in tqdm(sst2_examples):
        if sst2_example in sst_examples:
            sent_counter += 1
        elif sst2_example in phrases:
            phrase_counter += 1
        else:
            closest_phrases = difflib.get_close_matches(sst2_example, phrases)

            flag = True
            for phrase in closest_phrases:
                distance = nltk.edit_distance(str(phrase), sst2_example)
                if distance <= 2:
                    phrase_counter += 1
                    flag = False
                    break
            if flag:
                outliner += 1

    print("The number of sentences / phrases / outliner of SST-2 = {} / {} / {}".format(sent_counter, phrase_counter, outliner))


if __name__ == '__main__':

    # For double check dataset
    # verify_sst2_label()
    # clarify_sst_and_sst2()

    data_path = "../datasets/sst/stanfordSentimentTreebank/"

    sentences, phrases = [], []
    split_dict, sentiment_dict = {}, {}

    all_file_names = listdir(data_path)
    for file_name in all_file_names:
        input_file = open(data_path + file_name, "r")
        if file_name == "datasetSentences.txt":
            next(input_file)  # Skip the first header row
            for line in input_file:
                sent_id, sent_text = line.strip().split("\t")
                sentence = Sentence(id=sent_id, text=sent_text, score=-1)
                sentences.append(sentence)

        elif file_name == "dictionary.txt":
            for line in input_file:
                phrase_text, phrase_id = line.strip().split("|")
                phrase = Phrase(id=phrase_id, text=phrase_text, score=-1)
                phrases.append(phrase)

        elif file_name == "datasetSplit.txt":
            next(input_file)  # Skip the first header row
            for line in input_file:
                sent_id, split = line.strip().split(",")
                split_dict[sent_id] = split

        elif file_name == "sentiment_labels.txt":
            next(input_file)  # Skip the first header row
            for line in input_file:
                phrase_id, sentiment_score = line.strip().split("|")
                sentiment_dict[phrase_id] = sentiment_score

        input_file.close()

    # Update split
    for sentence in sentences:
        sentence.split = split_dict[sentence.id]

    # Update phrases' sentiment value
    for phrase in phrases:
        phrase.score = sentiment_dict[phrase.id]

    # Sort list of phrases by descending in text's length
    phrases = sorted(phrases, key=lambda x: -len(x.text))

    number_of_sentences_assigned = 0
    number_of_phrases_assigned = 0

    for sentence in tqdm(sentences):
        for phrase in phrases:
            if phrase.text in sentence.text:
                # Store all the phrases that belong to the sentence (including the sentence)
                sentence.phrases.append(phrase)
                number_of_phrases_assigned += 1

                # Assign sentiment score for the sentence if they are identical
                if phrase.text == sentence.text:
                    sentence.score = phrase.score
                    number_of_sentences_assigned += 1

    pickle_fp = "../../../data/pickle_files/human_annotations/"
    if not exists(pickle_fp):
        makedirs(pickle_fp)

    with open(pickle_fp + "sst_preprocessed.pickle", "wb") as file_path:
        pickle.dump(sentences, file_path)

    print("Complete! There are {} sentences and {} phrases assigned.".format(number_of_sentences_assigned, number_of_phrases_assigned))


'''
100%|█████████████████████████████████████| 11855/11855 [23:09<00:00,  8.53it/s]
Logs: Complete! There are 11286 sentences and 1062631 phrases assigned.
Notes:
    + There are sentences which do not exist in dictionary (11855 - 11286 = 569) --> Could not find its sentiment score.
    + There are multiple cases in which a phrase belongs to multiple sentences
    

clarify_sst_and_sst2()
(attribution_eval) thang@gpu4:~/Projects/attribution_eval/src$ python data_processor_sst.py
Reusing dataset glue (/home/thang/.cache/huggingface/datasets/glue/sst2/1.0.0/dacbe3125aa31d7f70367a07a8a9e72a5a0bfeb5fc42e75c9db75b96da6053ad)
No config specified, defaulting to: sst/default
Reusing dataset sst (/home/thang/.cache/huggingface/datasets/sst/default/1.0.0/b8a7889ef01c5d3ae8c379b84cc4080f8aad3ac2bc538701cbe0ac6416fb76ff)
100%|█████████████████████████████████████████████| 70042/70042 [2:52:54<00:00,  6.75it/s]
The number of sentences / phrases / outliner of SST-2 = 9467 / 60530 / 45
'''

