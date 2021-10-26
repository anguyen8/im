import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data.dataset import Dataset

from ...tokenization_roberta import RobertaTokenizer, RobertaTokenizerFast
from ...tokenization_utils import PreTrainedTokenizer
from ...tokenization_xlm_roberta import XLMRobertaTokenizer
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures

logger = logging.getLogger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    # ThangPM
    model_base: str = field(default="", metadata={"help": "Model base (e.g., bert-base-uncased, roberta-base)"})
    masked_lm: str = field(default="", metadata={"help": "Masked language models for running analyzers (e.g., bert-base-uncased, roberta-base)"})
    analyzer: str = field(default="", metadata={"help": "Analyzers (e.g., IM, LOOEmpty, LOOUnk, LOOZero)"})
    sst_flag: bool = field(default=False, metadata={"help": "Specify SST task when necessary since task_name is set to SST-2"})
    checkpoint: str = field(default="", metadata={"help": "Specify a checkpoint i.e. output folder for attribution methods"})
    eval_metric: str = field(default="auc", metadata={"help": "Specify a metric used to evaluate IM and LOO_empty ['auc', 'auc_bert', 'roar', 'roar_bert', 'human_highlights']"})
    overwrite_results: bool = field(default=False, metadata={"help": "Overwrite cache and re-compute attribution scores"})

    roar: bool = field(default=False, metadata={"help": "Use this flag for ROAR"})
    attribution_train_dir: str = field(default="", metadata={"help": "Provide a path to modified examples for re-training with ROAR"})
    attribution_eval_dir: str = field(default="", metadata={"help": "Provide a path to modified examples for evaluation with ROAR"})
    roar_settings: str = field(default="", metadata={"help": "Specify settings (e.g., attribution method, deletion rate) for ROAR"})
    roar_type: str = field(default="vanilla", metadata={"help": "Specify ROAR type in ['vanilla', 'vanilla_baseline', 'bert', 'bert_baseline']"})

    sanity_check: bool = field(default=False, metadata={"help": "Run sanity check"})
    reproduce_im: bool = field(default=False, metadata={"help": "Reproduce SST-2 results under AUC and AUC_rep metrics for 700 randomly-selected examples from correctly-predicted ones"})

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    args: GlueDataTrainingArguments
    output_mode: str
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizer,
        limit_length: Optional[int]=None,
        mode: Union[str, Split]=Split.train,
        cache_dir: Optional[str]=None,
        examples=None,
    ):
        self.args = args
        self.processor = glue_processors[args.task_name]()
        self.output_mode = glue_output_modes[args.task_name]
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(
            cache_dir if cache_dir is not None else args.data_dir,
            "cached_{}_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(args.max_seq_length), args.task_name),
        )

        label_list = self.processor.get_labels()
        if args.task_name in ["mnli", "mnli-mm"] and tokenizer.__class__ in (
            RobertaTokenizer,
            RobertaTokenizerFast,
            XLMRobertaTokenizer,
        ):
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        lock_path = cached_features_file + ".lock"
        with FileLock(lock_path):

            if os.path.exists(cached_features_file) and not args.overwrite_cache:
                start = time.time()
                self.features = torch.load(cached_features_file)
                logger.info(f"Loading features from cached file {cached_features_file} [took %.3f s]", time.time() - start)
            else:
                logger.info(f"Creating features from dataset file at {args.data_dir}")

                if examples is None:
                    if mode == Split.dev:
                        self.examples = self.processor.get_dev_examples(args.data_dir)
                        self.attribution_dir = args.attribution_eval_dir
                    elif mode == Split.test:
                        self.examples = self.processor.get_test_examples(args.data_dir)
                    else:
                        self.examples = self.processor.get_train_examples(args.data_dir)
                        self.attribution_dir = args.attribution_train_dir
                else:
                    self.examples = examples

                if limit_length is not None:
                    self.examples = self.examples[:limit_length]

                # ------------------------------------------------------------------------------------------
                # ThangPM: For ROAR
                # Load examples for re-training and evaluating models
                # ------------------------------------------------------------------------------------------
                if args.roar:
                    import pickle
                    attr_method, roar_rate = args.roar_settings.split("/")
                    lite_pickle_fb = self.attribution_dir + attr_method + "/" + \
                                     "del_examples_" + roar_rate + "_" + args.roar_type + ".pickle"

                    print("********** LOADING PICKLE FILE: " + lite_pickle_fb)
                    with open(lite_pickle_fb, "rb") as file_path:
                        self.examples = pickle.load(file_path)
                # ------------------------------------------------------------------------------------------

                self.features = glue_convert_examples_to_features(
                    self.examples,
                    tokenizer,
                    max_length=args.max_seq_length,
                    label_list=label_list,
                    output_mode=self.output_mode,
                )
                start = time.time()

                # ^ This seems to take a lot of time so I want to investigate why and how we can improve.
                logger.info("Saving features into cached file %s [took %.3f s]", cached_features_file, time.time() - start)

            # To print out the distribution of dataset regarding gold labels
            distribution_dict = {}
            for feature in self.features:
                if feature.label not in distribution_dict:
                    distribution_dict[feature.label] = 0

                distribution_dict[feature.label] += 1

            print(distribution_dict)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


