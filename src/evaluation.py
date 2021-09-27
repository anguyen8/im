from tqdm import tqdm
import numpy as np
import copy
import math
import random
import pickle
from collections import Counter
import statistics
import itertools
from sklearn.metrics import auc

import spacy
nlp = spacy.load("en_core_web_sm")

from analyzers.base import skipped_indices


class Evaluation(object):
    def __init__(self, model_wrapper, dev_set, deletion_level=0):
        self.model_wrapper = model_wrapper
        self.dev_set = dev_set
        self.deletion_level = deletion_level
        self.del_examples = []
        self.chunk_size_list = []

    # AUC
    def generate_examples_for_deletion_method(self, corr_pred_only):
        self.del_examples = []
        self.chunk_size_list = []

        for item in tqdm(self.dev_set):
            example = item["ori_example"]
            self.del_examples.append(example.get_input_example())

            # ONLY consider correct predictions
            if corr_pred_only and self.model_wrapper.labels.index(example.label) != example.pred_label:
                self.chunk_size_list.append(1)
                continue

            del_example = copy.deepcopy(example)
            flags = [True] * del_example.get_length()
            deletion_level = self.deletion_level
            if self.deletion_level == 0 or self.deletion_level >= len(flags):
                deletion_level = len(flags) - 1
                deletion_level = math.ceil(deletion_level * 0.2)  # Only replace 20% tokens per example.
                if deletion_level <= 1:
                    deletion_level = 2

            for i in range(deletion_level):  # The last removed sentence should include the last word
                max_pos = np.argmax(np.array(del_example.get_attribution_scores()))
                assert flags[max_pos.item()] is True
                flags[max_pos.item()] = False

                del_input_example = del_example.get_input_example_for_deletion_method(flags)
                self.del_examples.append(del_input_example)
                del_example.get_attribution_scores()[max_pos] = -999

            # Number of input-removed examples (including original input example)
            self.chunk_size_list.append(deletion_level + 1)

    # AUC_rep
    def generate_examples_for_replacement_method(self, corr_pred_only):
        self.del_examples = []
        self.chunk_size_list = []

        for item in tqdm(self.dev_set):
            example = item["ori_example"]
            self.del_examples.append(example.get_input_example())

            # ONLY consider correct predictions
            if corr_pred_only and self.model_wrapper.labels.index(example.label) != example.pred_label:
                self.chunk_size_list.append(1)
                continue

            masked_example = copy.deepcopy(example)
            flags = [True] * masked_example.get_length()
            deletion_level = self.deletion_level
            if self.deletion_level == 0 or self.deletion_level >= len(flags):
                deletion_level = len(flags) - 1
                deletion_level = math.ceil(deletion_level * 0.2)  # Only replace 20% tokens per example.
                if deletion_level <= 1:
                    deletion_level = 2

            for i in range(deletion_level):  # The last removed sentence should include the last word
                max_pos = np.argmax(np.array(masked_example.get_attribution_scores()))
                assert flags[max_pos.item()] is True
                flags[max_pos.item()] = False

                del_input_example = masked_example.get_input_example_for_replacement_method(self.model_wrapper, max_pos.item())
                self.del_examples.append(del_input_example)
                masked_example.get_attribution_scores()[max_pos] = -999

            # Number of input-removed examples (including original input example)
            self.chunk_size_list.append(deletion_level + 1)

    def generate_examples_for_roar(self, prefix, suffix, del_rate=0.2, use_bert=False, random_baseline=False):
        self.del_examples = []

        for item in tqdm(self.dev_set):
            example = item["ori_example"]

            del_example = copy.deepcopy(example)
            flags = [True] * del_example.get_length()
            deletion_level = self.deletion_level
            if self.deletion_level == 0 or self.deletion_level >= len(flags):
                deletion_level = len(flags) - 1
                deletion_level = math.ceil(deletion_level * del_rate)  # Only replace 20% tokens per example.
                if deletion_level <= 1:
                    deletion_level = 1  # Remove at least 1 token

            if not random_baseline:
                for i in range(deletion_level):  # The last removed sentence should include the last word
                    max_pos = np.argmax(np.array(del_example.get_attribution_scores()))
                    assert flags[max_pos.item()] is True
                    flags[max_pos.item()] = False
                    del_example.get_attribution_scores()[max_pos] = -999
            else:
                random_indices = random.sample(list(range(0, len(flags), 1)), k=deletion_level)
                flags = [True if idx not in random_indices else False for idx in range(len(flags))]

            if not use_bert:
                del_input_example = del_example.get_input_example_for_deletion_method(flags)
            else:
                del_input_example = del_example.get_input_example_for_bert_based_deletion_method(self.model_wrapper, flags)

            self.del_examples.append(del_input_example)

        lite_pickle_fb = prefix + "del_examples_" + str(del_rate) + ("_use_bert" if use_bert else "") + ("_baseline" if random_baseline else "") + suffix
        with open(lite_pickle_fb, "wb") as file_path:
            pickle.dump(self.del_examples, file_path)

    def compute_auc_score(self, method="LOO", corr_pred_only=False):

        # AUC_rep
        if method == "replacement":
            self.generate_examples_for_replacement_method(corr_pred_only)
        # Vanilla AUC
        else:
            self.generate_examples_for_deletion_method(corr_pred_only)

        all_conf_scores, _ = self.model_wrapper.predict_proba_with_examples(self.del_examples)
        auc_scores, auc_scores_norm = [], []
        count = 0

        for idx in range(len(self.chunk_size_list)):
            if idx == 0:
                start = 0
                end = self.chunk_size_list[idx]
            else:
                start = sum(self.chunk_size_list[:idx])
                end = sum(self.chunk_size_list[:idx + 1])

            # SKIP INCORRECT PREDICTIONS
            if end - start == 1:
                continue

            conf_scores = all_conf_scores[start:end]
            ori_pred = np.argmax(conf_scores[0])

            # Prediction of original example when evaluation and analyzing must be the same
            # assert ori_pred == self.model_wrapper.labels.index(self.del_examples[start].label)
            # assert ori_pred == self.dev_set[idx]["ori_example"].get_pred_label()

            if ori_pred != self.dev_set[idx]["ori_example"].get_pred_label():
                count += 1

            # Compute AUC score
            auc_conf_scores = [conf_score[ori_pred] for conf_score in conf_scores[1:]]
            x = np.arange(1, len(auc_conf_scores) + 1) / len(auc_conf_scores)
            y_pred = auc_conf_scores

            if len(x) > 1:
                auc_score = auc(x, y_pred)
                auc_scores.append(auc_score)

        # auc_scores = random.sample(auc_scores, 700)
        print("Evaluation metric: " + ("AUC" if method=="LOO" else "AUC_rep"))
        print("Average AUC score = " + str(statistics.mean(auc_scores)))
        print("Number of examples: " + str(len(auc_scores)))
        print("Number of examples changing predictions: " + str(count))

        return statistics.mean(auc_scores)

    def compute_IoU_score(self, analyzer, groundtruth_sets, alpha=0.05, visualize=False):

        def intersection(lst1, lst2):
            # return list(set(lst1) & set(lst2))
            return list((Counter(lst1) & Counter(lst2)).elements())

        def union(lst1, lst2):
            # return list(set().union(lst1, lst2))
            return list((Counter(lst1) | Counter(lst2)).elements())

        def compute_scores(sets_1, sets_2):
            IoU_scores = []

            for set1, set2 in zip(sets_1, sets_2):
                if len(union(set1, set2)) == 0:
                    IoU_scores.append(1)    # Assume that this example has no important words
                    continue

                IoU_score = len(intersection(set1, set2)) / len(union(set1, set2))
                IoU_scores.append(IoU_score)

            precision_scores, recall_scores = [], []
            for set1, set2 in zip(sets_1, sets_2):
                if len(set2) == 0:
                    precision_score = 1 if len(set1) == 0 else 0
                    recall_score = 1
                elif len(set1) == 0:
                    precision_score = recall_score = 0
                else:
                    precision_score = len(intersection(set1, set2)) / len(set1)
                    recall_score = len(intersection(set1, set2)) / len(set2)

                precision_scores.append(precision_score)
                recall_scores.append(recall_score)

            avg_IoU_score = round(sum(IoU_scores) / len(IoU_scores), 4)
            avg_precision_score = round(sum(precision_scores) / len(precision_scores), 4)
            avg_recall_score = round(sum(recall_scores) / len(recall_scores), 4)

            return avg_IoU_score, avg_precision_score, avg_recall_score

        # Each example includes a set of high-attribution words for predicted label (pos/neg)
        sets_0, sets_1, sets_2 = [], [], []
        gt_dev_set = [sent for sent in groundtruth_sets if str(sent.split) == '3']   # split = 3 for dev set

        # Skip examples whose lengths after tokenization >= 512 (For MultiRC, max_length of BertMLM = 512)
        # Handle this since gt_dev_set ALWAYS has full examples per split
        task_name = self.model_wrapper.data_args.task_name if self.model_wrapper.data_args.task_name != "sst-2" else "sst"
        skipped_list = skipped_indices[task_name + "_split{}".format(self.model_wrapper.split)] if self.model_wrapper.max_split > 1 else skipped_indices[task_name]
        gt_dev_set = [example for idx, example in enumerate(gt_dev_set) if idx not in skipped_list]

        # USE ONLY FOR ANALYZING ESNLI
        # kept_list = kept_indices[task_name + "_split{}".format(self.model_wrapper.split)] if self.model_wrapper.max_split > 1 else kept_indices[task_name]
        # gt_dev_set = [example for idx, example in enumerate(gt_dev_set) if idx in kept_list]

        # Number of examples must be equal between dev_set and groundtruth
        assert(len(self.dev_set) == len(gt_dev_set))
        # print("***** Number of evaluated examples in total: {} *****".format(len(self.dev_set)))
        count_correct_preds = 0

        # For Statistics
        annotation_stats = {"analyzer": [], "human": [], "sent_len": []}

        # Generate_high_attribution_set
        for idx, (item, item2) in enumerate(zip(self.dev_set, gt_dev_set)):
        # for idx, (item, item2) in tqdm(enumerate(zip(self.dev_set, gt_dev_set))):

            example = item["ori_example"]
            pred_label = example.get_pred_label()
            all_tokens = [x.token for x in example.get_all_attr_tokens()]

            # CORRECT PREDICTIONS ONLY
            if example.label not in self.model_wrapper.labels:
                example.label = '1' if float(example.label) >= 0.5 else '0'
            if self.model_wrapper.labels.index(example.label) != example.pred_label:
                continue

            count_correct_preds += 1

            # Prepare set0 (baseline) and set1 (RIS or OccEmpty)
            # Normalize + Zero-out and binarize
            if max(example.get_attribution_scores()) > 0:
                attr_scores = example.get_attribution_scores() / max(example.get_attribution_scores())
                attr_scores = [1 if score >= alpha else 0 for score in attr_scores]
            else:
                attr_scores = [0] * len(example.get_attribution_scores())

            # Baseline: All words are important
            # attr_scores = [1] * len(attr_scores)

            # Number of attribution scores and tokens must be equal for each example
            # Attribution score must be in [0, 1]
            assert len(all_tokens) == len(attr_scores)
            assert min(attr_scores) >= 0 and max(attr_scores) <= 1

            high_attr_tokens = [idx for idx, attr_score in enumerate(attr_scores) if attr_score == 1]
            sets_1.append(high_attr_tokens)
            sets_0.append([idx for idx, attr_score in enumerate(attr_scores)])  # Baseline: All words are highlighted

            # For Statistics
            annotation_stats["analyzer"].append(len(high_attr_tokens))
            annotation_stats["sent_len"].append(len(attr_scores))

            # Prepare set2: Groundtruth
            set2 = []
            if hasattr(item2, "phrases"):   # SST
                for phrase in item2.phrases:
                    # positive = 1 or negative = 0
                    if (pred_label == 1 and float(phrase.score) >= 0.7) or (pred_label == 0 and float(phrase.score) <= 0.3):
                        doc = nlp(phrase.text)
                        phrase_tokens = [token.text for token in doc]
                        for i in range(len(all_tokens) - len(phrase_tokens) + 1):
                            # if sub tokens are found in all_tokens and its length <= 50% of sentence length
                            if (all_tokens[i:i + len(phrase_tokens)] == phrase_tokens) and (len(phrase_tokens) <= len(all_tokens) / 2):
                                set2.append([idx for token, idx in zip(phrase_tokens, range(i, i + len(phrase_tokens), 1))])
                sets_2.append(list(set(itertools.chain(*set2))))
                annotation_stats["human"].append(len(sets_2[-1]))

            elif hasattr(item2, "highlights"): # ESNLI and MultiRC
                # ThangPM: Temporarily hot fixed.
                # Handle 24 cases that have empty tokens in either hypothesis or premise of sets_1.
                # Reason: Due to redundant spaces in the Huggingface datasets used --> Need to process this next time.
                if item2.processed_length != len(all_tokens):
                    del_indices = [i for i, x in enumerate(all_tokens) if x == " "]
                    all_tokens = [value for idx, value in enumerate(all_tokens) if idx not in del_indices]
                    if item2.processed_length != len(all_tokens):
                        print(all_tokens)
                    else:
                        attr_scores = [value for idx, value in enumerate(attr_scores) if idx not in del_indices]
                        high_attr_tokens = [idx for idx, attr_score in enumerate(attr_scores) if attr_score == 1]
                        sets_1[-1] = high_attr_tokens
                        annotation_stats["analyzer"][-1] = len(high_attr_tokens)
                        annotation_stats["sent_len"][-1] = len(attr_scores)

                set2 = sorted([idx for idx, token in item2.highlights["text_a"] + item2.highlights["text_b"]])
                sets_2.append(set2)
                annotation_stats["human"].append(len(sets_2[-1]))

            # Visualize sentences with highlight words to compare RIS with OccEmpty.
            if visualize:
                self.highlight_text(analyzer, idx, tokens=all_tokens, set1=sets_1[-1], set2=sets_2[-1], label=example.label)
                IoU_score = len(intersection(sets_1[-1], sets_2[-1])) / len(union(sets_1[-1], sets_2[-1]))
                if len(sets_2[-1]) == 0:
                    precision_score = 1 if len(sets_1[-1]) == 0 else 0
                    recall_score = 1
                elif len(sets_1[-1]) == 0:
                    precision_score = recall_score = 0
                else:
                    precision_score = len(intersection(sets_1[-1], sets_2[-1])) / len(sets_1[-1])
                    recall_score = len(intersection(sets_1[-1], sets_2[-1])) / len(sets_2[-1])
                print("\nIoU = {:.2f}, Precision = {:.2f}, Recall = {:.2f}".format(IoU_score, precision_score, recall_score))

        # For Statistics
        if visualize:
            analyzer_coverage, human_coverage = 0, 0
            for analyzer_attr_len, human_ann_len, example_len in zip(annotation_stats["analyzer"], annotation_stats["human"], annotation_stats["sent_len"]):
                analyzer_coverage += analyzer_attr_len / example_len
                human_coverage += human_ann_len / example_len
            analyzer_coverage /= len(annotation_stats["sent_len"])
            human_coverage /= len(annotation_stats["sent_len"])
            print("{} = {:.2f}, Human = {:.2f}, Examples Length Average = {:.2f}".format(analyzer, analyzer_coverage, human_coverage, statistics.mean(annotation_stats["sent_len"])))

            results = []
            for x, y in zip(annotation_stats["analyzer"], annotation_stats["sent_len"]):
                results.append(x / y)
            print("{}: Min = {:.2f} Max = {:.2f}".format(analyzer, min(results), max(results)))

            results = []
            for x, y in zip(annotation_stats["human"], annotation_stats["sent_len"]):
                results.append(x / y)
            print("{}: Min = {:.2f} Max = {:.2f}".format("Human", min(results), max(results)))

        IoU_baseline, precision_baseline, recall_baseline = compute_scores(sets_1=sets_0, sets_2=sets_2)
        IoU, precision, recall = compute_scores(sets_1=sets_1, sets_2=sets_2)

        # print("***** Number of evaluated examples: {} *****".format(count_correct_preds))

        return {"scores": (IoU, precision, recall), "scores_baseline": (IoU_baseline, precision_baseline, recall_baseline)}

    def highlight_text(self, analyzer, idx, tokens, set1, set2, label):

        print("\n----- Example {} ----- Label: {}".format(idx, label))

        # token's foreground is black, token's background is orange/yellow
        format = ';'.join([str(0), str(30), str(43)])

        highlight_tokens_set1 = copy.deepcopy(tokens)
        highlight_tokens_set2 = copy.deepcopy(tokens)

        for idx, token in enumerate(tokens):
            highlight_tokens_set1[idx] = '\x1b[%sm%s\x1b[0m' % (format, token) if idx in set1 else token
            highlight_tokens_set2[idx] = '\x1b[%sm%s\x1b[0m' % (format, token) if idx in set2 else token

        print(analyzer + ": " + " ".join(highlight_tokens_set1))    # highlight important words for set1
        print("\nHuman: " + " ".join(highlight_tokens_set2))          # highlight important words for set2
