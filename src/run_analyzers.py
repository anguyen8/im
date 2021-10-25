import sys
from data_processors.human_annotations import data_processor_sst
from data_processors.human_annotations import data_processor_esnli
from data_processors.human_annotations import data_processor_multirc
sys.modules['data_processor_sst'] = data_processor_sst
sys.modules['data_processor_esnli'] = data_processor_esnli
sys.modules['data_processor_multirc'] = data_processor_multirc

import pickle
import time
from functools import wraps

import numpy as np
from matplotlib import pylab as plt
from model_wrapper import RoBERTa_Model_Wrapper
from analyzers import OcclusionAnalyzer
from analyzers import InputMarginalizationAnalyzer
from analyzers import LimeAnalyzer
from evaluation import Evaluation

# from lime.lime_text import LimeTextExplainer
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

import spacy
nlp = spacy.load("en_core_web_sm")


# Python program to print colored text and background
class Colors:
    '''Colors class:reset all colors with colors.reset; two sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable, underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold'''

    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'

    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'

    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


MASKED_EXAMPLES = "masked_examples"
FILLED_EXAMPLES = "filled_examples"
INPUT_EXAMPLES = "input_examples"
CHUNK_SIZE_LIST = "chunk_size_list"
ALL_CONF_SCORES = "all_conf_scores"

folder_name_dict = {
    "sst-2": "SST-2",
    "sst": "SST",
    "esnli": "ESNLI",
    "multirc": "MultiRC",
}

skipped_indices = {
    "sst-2": [],
    "sst": [],
    "esnli": [],
    "multirc_split0": [497, 498, 500, 506, 513, 517, 518, 519, 520, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735],
    "multirc_split1": [66, 81, 90, 102, 108, 109, 110, 111, 112, 128, 131, 132, 136, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 1079, 1080, 1082, 1086, 1088, 1089, 1092, 1093, 1094, 1095, 1096, 1097, 1103, 1104, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1121, 1122, 1123, 1124, 1125, 1126, 1127, 1144, 1145, 1146, 1147, 1148, 1149, 1150],
    "multirc_split2": [626, 629, 630, 645, 646, 647, 648, 650, 677, 678, 679, 681, 685, 1048, 1049, 1050, 1051, 1052, 1055, 1056, 1061, 1062, 1063, 1064, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1076, 1077, 1083, 1084, 1085, 1086, 1087, 1088, 1089, 1090, 1092, 1093, 1097, 1103, 1105, 1106, 1109, 1111, 1113, 1114, 1116, 1117, 1118, 1119, 1120, 1121, 1124, 1126, 1127, 1128, 1129, 1130, 1131, 1133, 1134, 1135, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1147, 1152, 1153, 1154, 1155, 1156, 1157, 1158, 1159, 1161, 1163],
}


def my_timer(my_func):

    @wraps(my_func)
    def timed(*args, **kw):
        tstart = time.time()
        output = my_func(*args, **kw)
        tend = time.time()
        print('"{}" took {:.3f} s to execute\n'.format(my_func.__name__, (tend - tstart)))
        return output

    return timed


def run_occlusion(model_wrapper, task_name, analyzer, pickle_fn, replaced_token):
    occ_analyzer = OcclusionAnalyzer(model_wrapper, task_name, analyzer=analyzer, pickle_fn=pickle_fn,
                                     processed_pickle_fn=FILLED_EXAMPLES, replaced_token=replaced_token)
    occ_analyzer.run()

    # Handle this temporarily since I haven't skipped those examples when running OccEmpty
    if task_name == "multirc" or task_name == "esnli":
        skipped_list = skipped_indices[task_name + "_split{}".format(model_wrapper.split)] if model_wrapper.max_split > 1 else skipped_indices[task_name]
        dev_set = [example for idx, example in enumerate(occ_analyzer.get_dev_set()) if idx not in skipped_list]
        occ_analyzer.set_dev_set(dev_set)

    occ_evaluation = Evaluation(model_wrapper, occ_analyzer.get_dev_set())
    eval_metric = model_wrapper.data_args.eval_metric

    # AUC and AUC_rep
    if eval_metric == "auc":
        occ_evaluation.compute_auc_score(corr_pred_only=True)                   # AUC

    elif eval_metric == "auc_bert":
        occ_evaluation.compute_auc_score("replacement", corr_pred_only=True)    # i.e., AUC_rep

    # ROAR
    elif eval_metric == "roar" or eval_metric == "roar_bert":
        del_rates = [0.1, 0.2, 0.3]
        use_bert = True if eval_metric == "roar_bert" else False

        for del_rate in del_rates:
            occ_evaluation.generate_examples_for_roar(prefix=occ_analyzer.prefix,
                                                      suffix=occ_analyzer.suffix,
                                                      del_rate=del_rate, use_bert=use_bert,
                                                      random_baseline=False)

        print("Finished generating new examples for {}. \n"
              "Please run the script `run_glue.sh` to re-train and re-evaluate the model on new examples.".format(eval_metric))

    # Human annotations/highlights
    elif model_wrapper.data_args.eval_metric == "human_highlights":
        alphas = np.arange(0.05, 1, 0.05)
        agreement_levels = [2, 3]

        for agreement_level in agreement_levels:
            groundtruth_path = "../data/pickle_files/human_annotations/{}_preprocessed{}.pickle".format(task_name, ("_" + str(agreement_level) if task_name == "esnli" else ""))
            with open(groundtruth_path, "rb") as file_path:
                groundtruth_sets = pickle.load(file_path)

                if model_wrapper.split != None:
                    chunk_size = int(len(groundtruth_sets) / model_wrapper.max_split)
                    groundtruth_sets = groundtruth_sets[model_wrapper.split*chunk_size:(model_wrapper.split+1)*chunk_size]

            scores_baseline = []
            for alpha in alphas:
                results = occ_evaluation.compute_IoU_score(analyzer, groundtruth_sets, alpha, visualize=True if len(alphas) == 1 else False)
                IoU, precision, recall = results["scores"]
                scores_baseline.append(results["scores_baseline"])

                # For easily copying to Google Sheets.
                print("\t".join([str(alpha), str(IoU), str(precision), str(recall)]))

            for (IoU_baseline, precision_baseline, recall_baseline) in tuple(set(scores_baseline)):
                # For easily copying to Google Sheets.
                print("\t".join(["Baseline:", str(IoU_baseline), str(precision_baseline), str(recall_baseline)]))

    del occ_analyzer


def run_input_marginalization(model_wrapper, task_name, analyzer, pickle_fn, threshold):
    input_margin_analyzer = InputMarginalizationAnalyzer(model_wrapper, task_name, analyzer=analyzer, pickle_fn=pickle_fn,
                                                         processed_pickle_fn=FILLED_EXAMPLES, threshold=threshold,)
    input_margin_analyzer.run()

    # Handle this temporarily since I haven't skipped those examples when running
    if task_name == "multirc" or task_name == "esnli":
        skipped_list = skipped_indices[task_name + "_split{}".format(model_wrapper.split)] if model_wrapper.max_split > 1 else skipped_indices[task_name]
        dev_set = [example for idx, example in enumerate(input_margin_analyzer.get_dev_set()) if idx not in skipped_list]
        input_margin_analyzer.set_dev_set(dev_set)

    input_margin_evaluation = Evaluation(model_wrapper, input_margin_analyzer.get_dev_set())
    eval_metric = model_wrapper.data_args.eval_metric

    # AUC and AUC_rep
    if model_wrapper.data_args.eval_metric == "auc":
        input_margin_evaluation.compute_auc_score(corr_pred_only=True)                  # AUC

    elif eval_metric == "auc_bert":
        input_margin_evaluation.compute_auc_score("replacement", corr_pred_only=True)   # i.e., AUC_rep

    # ROAR
    elif eval_metric == "roar" or eval_metric == "roar_bert":
        del_rates = [0.1, 0.2, 0.3]
        use_bert = True if eval_metric == "roar_bert" else False

        for del_rate in del_rates:
            input_margin_evaluation.generate_examples_for_roar(prefix=input_margin_analyzer.prefix,
                                                               suffix=input_margin_analyzer.suffix,
                                                               del_rate=del_rate, use_bert=use_bert,
                                                               random_baseline=False)

        print("Finished generating new examples for {}. \n"
              "Please run the script `run_glue.sh` to re-train and re-evaluate the model on new examples.".format(eval_metric))

    # Human annotations/highlights
    elif model_wrapper.data_args.eval_metric == "human_highlights":
        alphas = np.arange(0.05, 1, 0.05)
        agreement_levels = [2, 3]

        for agreement_level in agreement_levels:
            groundtruth_path = "../data/pickle_files/human_annotations/{}_preprocessed{}.pickle".format(task_name, ("_" + str(agreement_level) if task_name == "esnli" else ""))
            with open(groundtruth_path, "rb") as file_path:
                groundtruth_sets = pickle.load(file_path)

                if model_wrapper.split != None:
                    chunk_size = int(len(groundtruth_sets) / model_wrapper.max_split)
                    groundtruth_sets = groundtruth_sets[model_wrapper.split*chunk_size:(model_wrapper.split+1)*chunk_size]

            scores_baseline = []
            for alpha in alphas:
                results = input_margin_evaluation.compute_IoU_score(analyzer, groundtruth_sets, alpha, visualize=True if len(alphas) == 1 else False)
                IoU, precision, recall = results["scores"]
                scores_baseline.append(results["scores_baseline"])

                # For easily copying to Google Sheets.
                print("\t".join([str(alpha), str(IoU), str(precision), str(recall)]))

            for (IoU_baseline, precision_baseline, recall_baseline) in tuple(set(scores_baseline)):
                # For easily copying to Google Sheets.
                print("\t".join(["Baseline:", str(IoU_baseline), str(precision_baseline), str(recall_baseline)]))

    del input_margin_analyzer


def run_lime(model_wrapper, task_name, analyzer, pickle_fn, replaced_token):
    lime_analyzer = LimeAnalyzer(model_wrapper, task_name, analyzer=analyzer, pickle_fn=pickle_fn,
                                 processed_pickle_fn=FILLED_EXAMPLES, replaced_token=replaced_token)
    lime_analyzer.run()

    # Handle this temporarily since I haven't skipped those examples when running OccEmpty
    if task_name == "multirc" or task_name == "esnli":
        skipped_list = skipped_indices[task_name + "_split{}".format(model_wrapper.split)] if model_wrapper.max_split > 1 else skipped_indices[task_name]
        dev_set = [example for idx, example in enumerate(lime_analyzer.get_dev_set()) if idx in skipped_list]
        lime_analyzer.set_dev_set(dev_set)

    lime_evaluation = Evaluation(model_wrapper, lime_analyzer.get_dev_set())

    # ThangPM: Reported only Human annotations/highlights in paper
    alphas = np.arange(0.05, 1, 0.05)
    agreement_levels = [2, 3]

    for agreement_level in agreement_levels:
        groundtruth_path = "../data/pickle_files/human_annotations/{}_preprocessed{}.pickle".format(task_name, ("_" + str(agreement_level) if task_name == "esnli" else ""))
        with open(groundtruth_path, "rb") as file_path:
            groundtruth_sets = pickle.load(file_path)

            if model_wrapper.split != None:
                chunk_size = int(len(groundtruth_sets) / model_wrapper.max_split)
                groundtruth_sets = groundtruth_sets[model_wrapper.split * chunk_size:(model_wrapper.split + 1) * chunk_size]

        scores_baseline = []
        for alpha in alphas:
            results = lime_evaluation.compute_IoU_score(analyzer, groundtruth_sets, alpha, visualize=True if len(alphas) == 1 else False)
            IoU, precision, recall = results["scores"]
            scores_baseline.append(results["scores_baseline"])

            # For easily copying to Google Sheets.
            print("\t".join([str(alpha), str(IoU), str(precision), str(recall)]))

        for (IoU_baseline, precision_baseline, recall_baseline) in tuple(set(scores_baseline)):
            # For easily copying to Google Sheets.
            print("\t".join(["Baseline:", str(IoU_baseline), str(precision_baseline), str(recall_baseline)]))

    del lime_analyzer


@my_timer
def run_analyzers(threshold=10e-5):

    # Prepare a classifier for LIME to generate explanations
    model_wrapper = RoBERTa_Model_Wrapper()
    task_name = model_wrapper.data_args.task_name

    # Since SST-2 model is used for SST task, we need to convert it to SST-2 for loading model first and then
    # changing it back to SST task below if the flag is on.
    if task_name == "sst-2" and model_wrapper.data_args.sst_flag:
        task_name = "sst"

    analyzer = model_wrapper.data_args.analyzer
    model_wrapper.max_split = 1 if task_name != "multirc" else 3

    for i in range(model_wrapper.max_split):

        mode = "train" if model_wrapper.data_args.checkpoint.find("train") != -1 else "dev"
        pickle_fn = "../data/pickle_files/masked_examples/masked_examples_{}_{}_split{}.pickle".format(mode, task_name, i)
        model_wrapper.split = i
        if model_wrapper.max_split == 1:
            pickle_fn = "../data/pickle_files/masked_examples/masked_examples_{}_{}.pickle".format(mode, task_name)
            model_wrapper.split = None

        if analyzer == "InputMargin":
            run_input_marginalization(model_wrapper, task_name, analyzer=analyzer, pickle_fn=pickle_fn, threshold=threshold)
        elif analyzer == "OccEmpty":
            run_occlusion(model_wrapper, task_name, analyzer=analyzer, pickle_fn=pickle_fn, replaced_token="")
        elif analyzer == "OccZero":
            run_occlusion(model_wrapper, task_name, analyzer=analyzer, pickle_fn=pickle_fn, replaced_token="[PAD]")
        elif analyzer == "OccUnk":
            run_occlusion(model_wrapper, task_name, analyzer=analyzer, pickle_fn=pickle_fn, replaced_token="[UNK]")

        elif analyzer == "LIME" or analyzer == "LIME-BERT":
            replace_token = "" if analyzer == "LIME" else model_wrapper.mask_token
            run_lime(model_wrapper, task_name, analyzer=analyzer, pickle_fn=pickle_fn, replaced_token=replace_token)


if __name__ == '__main__':
    run_analyzers()

