from os.path import exists
from tqdm import tqdm
import numpy as np

from analyzers import Analyzer
from analyzers import MASKED_EXAMPLES, INPUT_EXAMPLES, ALL_CONF_SCORES


class InputMarginalizationAnalyzer(Analyzer):
    def __init__(self, model_wrapper, task_name, analyzer, pickle_fn, processed_pickle_fn=MASKED_EXAMPLES, threshold=10e-5):
        super().__init__(model_wrapper, task_name, analyzer, pickle_fn, processed_pickle_fn)
        self.threshold = threshold

    def prepare_examples_for_analysis(self):
        self.dev_set = self.load_dev_set_from_file(self.pickle_fn, self.processed_pickle_fn)

        # Convert AttributionExamples objects to InputExample objects for prediction
        for idx in tqdm(range(len(self.dev_set))):
            original_example = self.dev_set[idx]["ori_example"]
            self.examples.append(original_example.get_input_example())

            masked_examples = self.dev_set[idx]["masked_examples"]

            # ThangPM-NOTE: FIXED using top_N=10 since it's better than threshold
            filled_examples = original_example.generate_candidates_for_masked_token(masked_examples, self.model_wrapper, threshold=self.threshold, top_N=10)
            self.examples.extend(filled_examples)

            self.chunk_size_list.append(len(self.examples))

    def run(self):

        # Load a pickle file of all confidence scores if existed for faster analyzing.
        lite_pickle_fb = self.prefix + ALL_CONF_SCORES + self.suffix
        if not exists(lite_pickle_fb) or self.model_wrapper.data_args.overwrite_results:

            # Load a pickle file of all input examples if existed for faster analyzing.
            # We do not need to re-generate masked examples if the pickle file exists since it's always the same.
            lite_pickle_fb = self.prefix + INPUT_EXAMPLES + self.suffix
            if not exists(lite_pickle_fb):
                self.prepare_examples_for_analysis()
                self.save_examples()
                self.dev_set.clear()  # No longer needed --> Save memory
            else:
                self.load_examples()
                print("***** LOADING PRE-GENERATED EXAMPLES *****")

            # split can be adjust to > 5 if the number of generated examples are large.
            split = 5 if self.task_name == "multirc" else 1
            chunk_size = int(len(self.examples) / split)

            # split+1 for handling the remaining examples
            for i in range(split+1):
                if len(self.examples[:chunk_size]) > 0:
                    conf_scores, _ = self.model_wrapper.predict_proba_with_examples(self.examples[:chunk_size])
                    self.all_conf_scores.extend(conf_scores)

                    # No longer needed --> Save memory
                    del self.examples[:chunk_size]

            self.save_all_conf_scores()

            # Load dev_set again for computing attribution scores
            self.load_dev_set()

            for idx in tqdm(range(len(self.chunk_size_list))):
                start = 0 if idx == 0 else self.chunk_size_list[idx-1]
                end = self.chunk_size_list[idx]

                conf_scores = self.all_conf_scores[start:end]
                ori_pred = np.argmax(conf_scores[0])

                # Compute attribution score
                masked_examples = self.dev_set[idx]["masked_examples"]
                attr_scores = self.compute_input_margination(masked_examples, conf_scores)

                # Update results for the original example
                self.dev_set[idx]["ori_example"].set_pred_label(ori_pred)
                self.dev_set[idx]["ori_example"].set_confidence_score(conf_scores[0][ori_pred])
                self.dev_set[idx]["ori_example"].set_attribution_scores(attr_scores)

            # Save the latest dev_set for visualization
            self.save_dev_set()

    def compute_input_margination(self, masked_examples, conf_scores, topN=None):

        def odds(score):
            return score / (1.0 - score)

        ori_pred = np.argmax(conf_scores[0])

        # Confidence scores
        ori_conf_score = conf_scores[0][ori_pred]
        all_masked_conf_scores = [conf_score[ori_pred] for conf_score in conf_scores[1:]]

        # Get probability scores for the candidate.
        all_mlm_probs = [masked_example.get_candidates()[1] for masked_example in masked_examples]
        start, end = 0, len(all_mlm_probs[0])
        masked_probs = []

        for idx, mlm_probs in enumerate(all_mlm_probs):
            if idx > 0:
                start += len(all_mlm_probs[idx - 1])
                end += len(all_mlm_probs[idx])

            masked_conf_scores = all_masked_conf_scores[start:end]
            assert (len(mlm_probs) == len(masked_conf_scores))

            if topN and len(mlm_probs) >= topN:
                mlm_probs = mlm_probs[:topN]
                masked_conf_scores = masked_conf_scores[:topN]

            masked_prob = [masked_conf_score * mlm_prob for masked_conf_score, mlm_prob in zip(masked_conf_scores, mlm_probs)]
            masked_prob = sum(masked_prob) / sum(mlm_probs)
            masked_probs.append(masked_prob)

        attr_scores = np.log2(odds(ori_conf_score)) - np.log2([odds(prob) for prob in masked_probs])
        return attr_scores


