from os.path import exists
from tqdm import tqdm
import numpy as np

from analyzers import Analyzer
from analyzers import MASKED_EXAMPLES, INPUT_EXAMPLES, ALL_CONF_SCORES


class OcclusionAnalyzer(Analyzer):
    def __init__(self, model_wrapper, task_name, analyzer, pickle_fn, processed_pickle_fn=MASKED_EXAMPLES, replaced_token=""):
        super().__init__(model_wrapper, task_name, analyzer, pickle_fn, processed_pickle_fn)
        self.replaced_token = replaced_token

    def prepare_examples_for_analysis(self):
        self.dev_set = self.load_dev_set_from_file(self.pickle_fn, self.processed_pickle_fn)

        # Convert AttributionExamples objects to InputExample objects for prediction
        for idx in tqdm(range(len(self.dev_set))):
            original_example = self.dev_set[idx]["ori_example"]
            self.examples.append(original_example.get_input_example())

            masked_examples = self.dev_set[idx]["masked_examples"]
            filled_examples = original_example.generate_candidates_for_occ_token(masked_examples, self.replaced_token)
            self.examples.extend(filled_examples)

            self.chunk_size_list.append(len(self.examples))

    def run(self):

        def odds(score):
            return score / (1.0 - score)

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

        # Kim et al. 2020: Interpretation of NLP models through input marginalization
        for idx in tqdm(range(len(self.chunk_size_list))):
            start = 0 if idx == 0 else self.chunk_size_list[idx-1]
            end = self.chunk_size_list[idx]

            conf_scores = self.all_conf_scores[start:end]
            ori_pred = np.argmax(conf_scores[0])
            ori_conf_score = conf_scores[0][ori_pred]
            masked_conf_scores = [conf_score[ori_pred] for conf_score in conf_scores[1:]]

            # Compute attribution score
            attr_scores = np.log2(odds(ori_conf_score)) - np.log2([odds(conf_score) for conf_score in masked_conf_scores])

            # Update results for the original example
            self.dev_set[idx]["ori_example"].set_pred_label(ori_pred)
            self.dev_set[idx]["ori_example"].set_confidence_score(conf_scores[0][ori_pred])
            self.dev_set[idx]["ori_example"].set_attribution_scores(attr_scores)

        # Save the latest dev_set for visualization
        self.save_dev_set()


