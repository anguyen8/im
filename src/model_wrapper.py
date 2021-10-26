import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np
import torch
from scipy.special import softmax

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, EvalPrediction, GlueDataset
from transformers import BertForMaskedLM, RobertaForMaskedLM
from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import (
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_compute_metrics,
    glue_output_modes,
    glue_processors,
    glue_tasks_num_labels,
    set_seed,
    InputExample,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )


class ModelWrapper(object):

    def __init__(self):

        # perform evaluation on single GPU
        self.model_args, self.data_args, self.training_args, self.logger = self.init_args()

        try:
            # ThangPM: ONLY FOR PYCHARM
            if self.data_args.task_name.upper() == "${TASK_NAME}":
                task_name = os.environ.get("TASK_NAME")
                self.data_args.task_name = task_name.lower()
                self.data_args.data_dir = self.data_args.data_dir.replace("${TASK_NAME}", task_name)
                self.model_args.model_name_or_path = self.model_args.model_name_or_path.replace("${TASK_NAME}", task_name)

            self.num_labels = glue_tasks_num_labels[self.data_args.task_name]
            self.output_mode = glue_output_modes[self.data_args.task_name]
            self.labels = glue_processors[self.data_args.task_name]().get_labels()

            # Mostly for large datasets (e.g., MultiRC)
            self.split = None
            self.max_split = None
        except KeyError:
            raise ValueError("Task not found: %s" % (self.data_args.task_name))

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.

        self.config = AutoConfig.from_pretrained(
            self.model_args.config_name if self.model_args.config_name else self.model_args.model_name_or_path,
            num_labels=self.num_labels,
            finetuning_task=self.data_args.task_name,
            cache_dir=self.model_args.cache_dir,
        )
        self.config.fix_tfm = False
        self.config.output_attentions = True

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_args.tokenizer_name if self.model_args.tokenizer_name else self.model_args.model_name_or_path,
            cache_dir=self.model_args.cache_dir,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_args.model_name_or_path,
            from_tf=bool(".ckpt" in self.model_args.model_name_or_path),
            config=self.config,
            cache_dir=self.model_args.cache_dir,
        )

        # ThangPM: Random weights of the classifier ONLY for SANITY CHECK
        if self.data_args.sanity_check:
            torch.manual_seed(self.training_args.seed)
            torch.nn.init.xavier_uniform_(self.model.classifier.weight)

        self.train_dataset, self.eval_dataset, self.test_dataset = None, None, None

        # Initialize our Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            compute_metrics=self.build_compute_metrics_fn(self.data_args.task_name),
        )

        if self.data_args.masked_lm and self.data_args.masked_lm != "None":
            if self.data_args.masked_lm == "roberta-base":
                self.mask_token = "<mask>"
                self.masked_model = RobertaForMaskedLM.from_pretrained(self.data_args.masked_lm)
            else:
                self.mask_token = "[MASK]"
                self.masked_model = BertForMaskedLM.from_pretrained(self.data_args.masked_lm)

            self.masked_model = self.masked_model.to(self.training_args.device)
            self.masked_tokenizer = AutoTokenizer.from_pretrained(self.data_args.masked_lm)

            self.masked_trainer = Trainer(
                model=self.masked_model,
                args=self.training_args,
            )

        # ONLY USE FOR LIME
        self.index = -1
        self.text_a = ""
        self.text_b = ""

    def init_args(self):
        # See all possible arguments in src/transformers/training_args.py
        # or by passing the --help flag to this script.
        # We now keep distinct sets of args, for a cleaner separation of concerns.

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

        if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
            # If we pass only one argument to the script and it's the path to a json file,
            # let's parse it to get our arguments.
            model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        else:
            model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
        )
        logger.warning(
            "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            training_args.local_rank,
            training_args.device,
            training_args.n_gpu,
            bool(training_args.local_rank != -1),
            training_args.fp16,
        )
        logger.info("Training/evaluation parameters %s", training_args)

        # Set seed
        set_seed(training_args.seed)

        return model_args, data_args, training_args, logger

    def load_and_cache_examples(self):

        # Get datasets
        train_dataset = (
            GlueDataset(self.data_args, tokenizer=self.tokenizer, cache_dir=self.model_args.cache_dir)
            if self.training_args.do_train
            else None
        )
        eval_dataset = (
            GlueDataset(self.data_args, tokenizer=self.tokenizer, mode="dev", cache_dir=self.model_args.cache_dir)
            if self.training_args.do_eval
            else None
        )
        test_dataset = (
            GlueDataset(self.data_args, tokenizer=self.tokenizer, mode="test", cache_dir=self.model_args.cache_dir)
            if self.training_args.do_predict
            else None
        )

        return train_dataset, eval_dataset, test_dataset

    def build_compute_metrics_fn(self, task_name: str) -> Callable[[EvalPrediction], Dict]:
        def compute_metrics_fn(p: EvalPrediction):
            if self.output_mode == "classification":
                preds = np.argmax(p.predictions, axis=1)
            elif self.output_mode == "regression":
                preds = np.squeeze(p.predictions)
            return glue_compute_metrics(task_name, preds, p.label_ids)

        return compute_metrics_fn

    def prepare_dataset_for_prediction(self, list_sentences, use_mlm=False):

        # Convert sentences to a list of InputExamples
        examples = []
        assert self.text_a == "" or self.text_b == ""

        for idx, sentence in enumerate(list_sentences):
            text_a = self.text_a if self.text_a else sentence
            text_b = self.text_b if self.text_b else sentence

            # SST/SST-2 WITH LIME AND LIME-BERT
            if self.data_args.task_name.lower() == "sst-2":
                text_b = ""

            example = InputExample(
                guid="predict-" + str(idx),
                text_a=text_a,
                text_b=text_b,
                label=None,
            )
            examples.append(example)

        # Save value for restoring after getting pred_dataset
        max_seq_length = self.data_args.max_seq_length
        if use_mlm:
            self.data_args.max_seq_length = 512

        predict_dataset = (
            GlueDataset(self.data_args, tokenizer=self.tokenizer, cache_dir=self.model_args.cache_dir, examples=examples)
        )

        self.data_args.max_seq_length = max_seq_length

        return predict_dataset

    '''
        This function is used for all analyzers
        Input: List of sentences ['stringA', 'stringB']
        Output: Probabilities of all sentences with shape (n, k) with n is number of sentences, k is number of labels
    '''
    def predict_proba(self, list_sentences):

        predict_dataset = self.prepare_dataset_for_prediction(list_sentences)
        prediction_output = self.trainer.predict(test_dataset=predict_dataset)
        total_logits = softmax(prediction_output.predictions, axis=1)
        return total_logits

    def predict_proba_with_examples(self, examples):

        # If the task is "SST" then convert all soft labels to either '0' or '1' based on the threshold 0.5
        if self.data_args.task_name.lower() == "sst-2" and self.data_args.sst_flag:
            for example in examples:
                example.label = '0' if example.label and float(example.label) < 0.5 else '1'

        predict_dataset = (
            GlueDataset(self.data_args, tokenizer=self.tokenizer, cache_dir=self.model_args.cache_dir, examples=examples)
        )

        prediction_output = self.trainer.predict(test_dataset=predict_dataset)
        total_logits = softmax(prediction_output.predictions, axis=1)

        return total_logits

    def predict_as_masked_lm(self, examples, top_N, threshold=10e-5, max_length=128):
        mlm_outputs, return_outputs = [], []
        mask_id = self.masked_tokenizer.convert_tokens_to_ids(self.mask_token)

        # ========================================================================================================
        input_pairs = [(example.text_a, example.text_b) for example in examples]

        # max_length = 512 only for MultiRC
        encoded_inputs = self.masked_tokenizer.batch_encode_plus(batch_text_or_text_pairs=input_pairs,
                                                                 max_length=max_length,
                                                                 pad_to_max_length=True,
                                                                 return_tensors="pt")
        encoded_inputs.data["input_ids"] = encoded_inputs.data["input_ids"].to(self.training_args.device)
        encoded_inputs.data["attention_mask"] = encoded_inputs.data["attention_mask"].to(self.training_args.device)

        if "token_type_ids" in encoded_inputs:
            encoded_inputs.data["token_type_ids"] = encoded_inputs.data["token_type_ids"].to(self.training_args.device)

        with torch.no_grad():
            mlm_outputs = self.masked_model(**encoded_inputs)[0]
        # ========================================================================================================

        for idx, encoded_input in enumerate(encoded_inputs.data['input_ids']):
            mask_index = list(encoded_input).index(mask_id)

            # Use fixed top_N instead of threshold
            if mask_index != -1:
                masked_probs = softmax(mlm_outputs[idx][mask_index].cpu().data.numpy())
                top_token_ids = np.argsort(masked_probs)[::-1][:top_N]
                masked_tokens = self.masked_tokenizer.convert_ids_to_tokens(top_token_ids)
                masked_probs = sorted(masked_probs, reverse=True)[:top_N]
                return_outputs.append((masked_tokens, masked_probs))

        return return_outputs

    def predict_multi_blanks_as_masked_lm(self, masked_sents):
        filled_sents = []
        mask_id = self.masked_tokenizer.convert_tokens_to_ids(self.mask_token)

        predict_dataset = self.prepare_dataset_for_prediction(masked_sents)
        last_hidden_states = self.masked_trainer.predict_masked_lm(test_dataset=predict_dataset).predictions

        for idx, preds in enumerate(last_hidden_states):

            masked_positions = (np.array(predict_dataset.features[idx].input_ids) == mask_id).nonzero()[0]
            choices_for_mask = []

            for index, mask_index in enumerate(list(masked_positions)):
                mask_hidden_state = preds[mask_index]
                indices = torch.topk(mask_hidden_state, k=5, dim=0)[1]
                candidates = [self.masked_tokenizer.convert_ids_to_tokens(idx.item()).strip() for idx in indices]
                choices_for_mask.append(candidates[0])  # Get top-1 word from MLM

            masked_indices = [(i.start(), i.end()) for i in re.finditer('\[MASK\]', masked_sents[idx])]
            filled_sent = masked_sents[idx]

            # ThangPM: Reverse list to replace [MASK] token from right to left to preserve masked indices
            masked_indices = masked_indices[::-1]
            choices_for_mask = choices_for_mask[::-1]

            for idx, masked_index in enumerate(masked_indices):
                filled_sent = filled_sent[:masked_index[0]] + choices_for_mask[idx] + filled_sent[masked_index[1]:]

            assert len(masked_positions) == len(choices_for_mask) == len(masked_indices)
            filled_sents.append(filled_sent)

        return filled_sents

    def predict_multi_blanks_as_masked_lm_batch_encoded(self, masked_sents, max_length=128):
        filled_sents = []
        batch_text_or_text_pairs = []
        mask_id = self.masked_tokenizer.convert_tokens_to_ids(self.mask_token)

        for masked_sent in masked_sents:
            text_a = self.text_a if self.text_a else masked_sent
            text_b = self.text_b if self.text_b else masked_sent

            # SST/SST-2 WITH LIME AND LIME-BERT
            if self.data_args.task_name.lower() == "sst-2":
                text_b = ""

            batch_text_or_text_pairs.append((text_a, text_b))

        features = self.masked_tokenizer.batch_encode_plus(batch_text_or_text_pairs=batch_text_or_text_pairs,
                                                           return_tensors="pt",
                                                           max_length=max_length,
                                                           pad_to_max_length=True)

        features.data["input_ids"] = features.data["input_ids"].to(self.training_args.device)
        features.data["attention_mask"] = features.data["attention_mask"].to(self.training_args.device)
        # Masked LM roberta-base has no 'token_type_ids'
        if self.data_args.masked_lm.find("bert-base-uncased") != -1:
            features.data["token_type_ids"] = features.data["token_type_ids"].to(self.training_args.device)

        with torch.no_grad():
            outputs = self.masked_model(**features)[0]

        for idx, last_hidden_state in enumerate(outputs):
            masked_positions = (features.data["input_ids"].squeeze() == mask_id).nonzero()
            masked_positions = [mask.item() for mask in masked_positions]
            choices_for_mask = []

            for index, mask_index in enumerate(list(masked_positions)):
                mask_hidden_state = last_hidden_state[mask_index]
                indices = torch.topk(mask_hidden_state, k=5, dim=0)[1]
                candidates = [self.masked_tokenizer.convert_ids_to_tokens(idx.item()).strip() for idx in indices]
                choices_for_mask.append(candidates[0])  # Get top-1 word from MLM

            masked_indices = [(i.start(), i.end()) for i in re.finditer('\[MASK\]', masked_sents[idx])]
            filled_sent = masked_sents[idx]

            # ThangPM: Reverse list to replace [MASK] token from right to left to preserve masked indices
            masked_indices = masked_indices[::-1]
            choices_for_mask = choices_for_mask[::-1]

            for idx, masked_index in enumerate(masked_indices):
                filled_sent = filled_sent[:masked_index[0]] + choices_for_mask[idx] + filled_sent[masked_index[1]:]

            assert len(masked_positions) == len(choices_for_mask) == len(masked_indices)
            filled_sents.append(filled_sent)

        return filled_sents

    def predict_multi_blanks_as_masked_lm_batch_conditionally(self, masked_sents, max_length=128):
        filled_sents = []
        mask_id = self.masked_tokenizer.convert_tokens_to_ids(self.mask_token)

        for masked_sent in masked_sents:
            text_a = self.text_a if self.text_a else masked_sent
            text_b = self.text_b if self.text_b else masked_sent

            # SST/SST-2 WITH LIME AND LIME-BERT
            if self.data_args.task_name.lower() == "sst-2":
                text_b = ""

            # Fill in [MASK] tokens conditionally: One at a time
            filled_sent = text_a
            while filled_sent.find(self.mask_token) != -1:

                # max_length = 512 only for MultiRC
                feature = self.masked_tokenizer.encode_plus(text=filled_sent, text_pair=text_b,
                                                            return_tensors="pt",
                                                            max_length=max_length,
                                                            pad_to_max_length=True)

                feature.data["input_ids"] = feature.data["input_ids"].to(self.training_args.device)
                feature.data["attention_mask"] = feature.data["attention_mask"].to(self.training_args.device)
                # Masked LM roberta-base has no 'token_type_ids'
                if self.data_args.masked_lm.find("bert-base-uncased") != -1:
                    feature.data["token_type_ids"] = feature.data["token_type_ids"].to(self.training_args.device)

                with torch.no_grad():
                    output = self.masked_model(**feature)
                    last_hidden_state = output[0].squeeze()

                masked_positions = (feature.data["input_ids"].squeeze() == mask_id).nonzero()
                mask_index = [mask.item() for mask in masked_positions][0]
                mask_hidden_state = last_hidden_state[mask_index]

                indices = torch.topk(mask_hidden_state, k=1, dim=0)[1]
                candidates = [self.masked_tokenizer.convert_ids_to_tokens(idx.item()).strip() for idx in indices]
                choices_for_mask = candidates[0]  # Get top-1 word from MLM

                # Get index of the first [MASK] to update `filled_sent`
                masked_index = [(i.start(), i.end()) for i in re.finditer('\[MASK\]', filled_sent)][0]
                filled_sent = filled_sent[:masked_index[0]] + choices_for_mask + filled_sent[masked_index[1]:]

            filled_sents.append(filled_sent)

        return filled_sents

    # ------------------------------------------------------------------------------------------
    #   HELPER FUNCTIONS
    # ------------------------------------------------------------------------------------------

    def set_index(self, index):
        self.index = index
    def get_example_by_idx(self, idx):
        _, _, current_dataset = self.get_dataset_info()
        return current_dataset.examples[idx]
    def get_sentence_by_idx(self, idx, attribute):
        _, _, current_dataset = self.get_dataset_info()
        if attribute == "text_b":
            return current_dataset.examples[idx].text_b

        return current_dataset.examples[idx].text_a
    def get_tokens_by_idx(self, idx):
        _, _, current_dataset = self.get_dataset_info()
        return current_dataset.examples[idx].text_a.strip().split(" ")
    def get_label_id_by_idx(self, idx):
        _, _, current_dataset = self.get_dataset_info()
        return current_dataset.features[idx].label
    def get_dataset_info(self):

        if self.training_args.do_eval:
            dataset = self.eval_dataset
        elif self.training_args.do_predict:
            dataset = self.test_dataset
        else:
            dataset = self.train_dataset

        labels = dataset.get_labels()
        data_size = len(dataset.features)

        return labels, data_size, dataset
    def get_labels(self):
        return self.labels
    def get_data_args(self):
        return self.data_args
    def set_text_a(self, text_a):
        self.text_a = text_a
    def set_text_b(self, text_b):
        self.text_b = text_b

    # ------------------------------------------------------------------------------------------

