from os.path import exists
from tqdm import tqdm
import numpy as np

from analyzers import Analyzer
from analyzers import MASKED_EXAMPLES, INPUT_EXAMPLES, ALL_CONF_SCORES

from lime import LimeTextExplainer


class LimeAnalyzer(Analyzer):
    def __init__(self, model_wrapper, task_name, analyzer, pickle_fn, processed_pickle_fn=MASKED_EXAMPLES, replaced_token=""):
        super().__init__(model_wrapper, task_name, analyzer, pickle_fn, processed_pickle_fn)
        self.replaced_token = replaced_token

    def prepare_examples_for_analysis(self):
        self.dev_set = self.load_dev_set_from_file(self.pickle_fn, self.processed_pickle_fn)

        # Convert AttributionExamples objects to InputExample objects for prediction
        for idx in tqdm(range(len(self.dev_set))):
            original_example = self.dev_set[idx]["ori_example"]
            self.examples.append(original_example.get_input_example())
            self.chunk_size_list.append(len(self.examples))

    def run(self):
        # Create a LIME explainer object
        explainer = LimeTextExplainer(class_names=self.model_wrapper.get_labels())

        # Load a pickle file of all confidence scores if existed for faster analyzing.
        lite_pickle_fb = self.prefix + ALL_CONF_SCORES + self.suffix
        if not exists(lite_pickle_fb):

            # Load a pickle file of all input examples if existed for faster analyzing.
            lite_pickle_fb = self.prefix + INPUT_EXAMPLES + self.suffix
            if not exists(lite_pickle_fb):
                self.prepare_examples_for_analysis()
                self.save_examples()
                self.dev_set.clear()  # No longer needed --> Save memory
            else:
                self.load_examples()
                print("***** LOADING PRE-GENERATED EXAMPLES *****")

            self.all_conf_scores, _ = self.model_wrapper.predict_proba_with_examples(self.examples)
            self.save_all_conf_scores()
        else:
            self.load_chunk_size_list()
            self.load_all_conf_scores()
            print("***** LOADING PRE-GENERATED CONFIDENCE SCORES *****")

        # Load dev_set for computing attribution scores
        self.load_dev_set()

        # For debugging only
        # self.dev_set = self.dev_set[:10]
        # self.all_conf_scores = self.all_conf_scores[:10]

        for example, logits in tqdm(zip(self.dev_set, self.all_conf_scores)):
            pred_label = np.argmax(logits)
            conf_score = logits[pred_label]

            # Generate heatmap for text_a first
            self.model_wrapper.set_text_a("")
            self.model_wrapper.set_text_b(example["ori_example"].get_text_b())
            attr_scores_text_a = self.get_lime_score(explainer, self.model_wrapper, example["ori_example"].get_text_a(),
                                                     pred_label=pred_label, use_mlm=self.analyzer == "LIME-BERT", num_samples=1000)

            # Then, generate heatmap for text_b if the task is ESNLI or MultiRC
            attr_scores_text_b = []
            if self.task_name == "esnli" or self.task_name == "multirc":
                self.model_wrapper.set_text_a(example["ori_example"].get_text_a())
                self.model_wrapper.set_text_b("")
                attr_scores_text_b = self.get_lime_score(explainer, self.model_wrapper, example["ori_example"].get_text_b(),
                                                         pred_label=pred_label, use_mlm=self.analyzer == "LIME-BERT", num_samples=1000)

            attr_scores = attr_scores_text_a + (attr_scores_text_b if self.task_name == "esnli" or self.task_name == "multirc" else [])

            # Update results for the original example
            example["ori_example"].set_pred_label(pred_label)
            example["ori_example"].set_confidence_score(conf_score)
            example["ori_example"].set_attribution_scores(np.array(attr_scores))

        # Save the latest dev_set for visualization
        self.save_dev_set()

    def get_lime_score(self, explainer, model_wrapper, text_instance, pred_label, use_mlm, num_samples=1000):

        LIME_scores = []
        lime_explanations = explainer.explain_instance(text_instance=text_instance,
                                                       classifier_fn=model_wrapper.predict_proba,
                                                       top_labels=len(model_wrapper.get_labels()),
                                                       num_features=len(text_instance.split(' ')),
                                                       num_samples=num_samples,
                                                       use_mlm=use_mlm,
                                                       model_wrapper=model_wrapper)

        for token in text_instance.split(" "):
            t_score = 0
            for word, score in lime_explanations.as_list(label=pred_label):
                if token.lower() == word.lower():
                    t_score = score

            if t_score == 0:
                # Remove special characters and get the score one more time
                new_token = ''.join(c for c in token if c.isalnum())
                for word, score in lime_explanations.as_list(label=pred_label):
                    if new_token.lower() == word.lower():
                        t_score = score

            # Remember: DO NOT round score here. Just round score for visualization only
            LIME_scores.append(t_score)

        return LIME_scores



