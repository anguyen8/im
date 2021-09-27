

class AttributionVisualization(object):

    def __init__(self, tokens, aspect, predicted_label, confidence_scores, ground_truth, attribution_scores, att_weights, grad_norms,
                 LIME_scores, occlusion_unk_scores, occlusion_empty_scores):

        self.tokens = tokens # dictionary
        self.all_tokens = [] # list

        self.aspect = aspect
        self.ground_truth = ground_truth
        self.predicted_label = predicted_label
        self.confidence_scores = confidence_scores

        self.attribution_scores = attribution_scores
        self.att_weights = att_weights
        self.grad_norms = grad_norms

        self.LIME_scores = LIME_scores  # dictionary
        self.all_LIME_scores = []       # list

        self.most_common_words = []
        self.cos_sim_scores = []
        self.shuffled_indices = []

        self.occlusion_unk_scores = occlusion_unk_scores
        self.occlusion_empty_scores = occlusion_empty_scores

        self.ris_explanation_list = []

        # Top matches based on highest self-attention weights
        self.top_matches = None

        # Only for testing purpose
        self.corr_score = None

    def get_word_list(self):
        return self.tokens
    def set_word_list(self, tokens):
        self.tokens = tokens
    def set_word_list_with_attribute(self, tokens, attribute):
        self.tokens[attribute] = tokens

    def get_all_word_list(self):
        if not self.all_tokens:
            self.all_tokens = self.get_word_list()["text_a"]

            if "text_b" in self.get_word_list() and self.get_word_list()["text_b"]:
                self.all_tokens = self.all_tokens + self.get_word_list()["text_b"]

        return self.all_tokens

    def set_all_word_list(self, all_tokens):
        self.all_tokens = all_tokens

    def get_category(self):
        return self.aspect

    def get_ground_truth(self, escape=False):
        if escape and self.ground_truth.find("_") != -1:
            return self.ground_truth.replace("_", "\_")

        return self.ground_truth

    def get_prediction(self, escape=False):
        if escape and self.predicted_label.find("_") != -1:
            return self.predicted_label.replace("_", "\_")

        return self.predicted_label

    def get_confidence_scores(self):
        return self.confidence_scores

    def set_confidence_scores(self, confidence_scores):
        self.confidence_scores = confidence_scores

    def get_ris_scores(self):
        return self.attribution_scores

    def get_att_weights(self):
        return self.att_weights

    def get_grad_scores(self):
        return self.grad_norms

    def get_lime_scores(self):
        return self.LIME_scores
    def set_lime_scores(self, LIME_scores):
        self.LIME_scores = LIME_scores

    def get_all_lime_scores(self):
        if not self.all_LIME_scores:
            self.all_LIME_scores = self.get_lime_scores()["text_a"]

            if "text_b" in self.get_lime_scores() and self.get_lime_scores()["text_b"]:
                self.all_LIME_scores = self.all_LIME_scores + self.get_lime_scores()["text_b"]

        return self.all_LIME_scores

    def set_all_lime_scores(self, all_LIME_scores):
        self.all_LIME_scores = all_LIME_scores

    def get_most_common_words(self):
        return self.most_common_words
    def set_most_common_words(self, most_common_words):
        self.most_common_words = most_common_words

    def get_cos_sim_scores(self):
        return self.cos_sim_scores
    def set_cos_sim_scores(self, cos_sim_scores):
        self.cos_sim_scores = cos_sim_scores

    def get_shuffled_indices(self):
        return self.shuffled_indices
    def set_shuffled_indices(self, shuffled_indices):
        self.shuffled_indices = shuffled_indices

    def get_top_matches(self):
        if hasattr(self, "top_matches"):
            return self.top_matches
        return None
    def set_top_matches(self, top_matches):
        self.top_matches = top_matches

    def get_occlusion_unk_scores(self):
        return self.occlusion_unk_scores

    def get_occlusion_empty_scores(self):
        return self.occlusion_empty_scores

    def get_ris_explanation_list(self):
        return self.ris_explanation_list

    def set_ris_explanation_list(self, ris_explanation_list):
        self.ris_explanation_list = ris_explanation_list

    def to_dict(self):

        att_weights = str(self.att_weights) if isinstance(self.att_weights, list) else str(self.att_weights.tolist())
        grad_norms = str(self.grad_norms) if isinstance(self.grad_norms, list) else str(self.grad_norms.tolist())

        return {"tokens": self.tokens,
                "aspect": self.aspect,
                "ground_truth": self.ground_truth,
                "predicted_label": self.predicted_label,
                "confidence_scores": str(self.confidence_scores.tolist()),
                "attribution_scores": str(self.attribution_scores),
                "att_weights": att_weights,
                "grad_norms": grad_norms,
                "LIME_scores": str(self.LIME_scores),
                "occlusion_unk_scores": str(self.occlusion_unk_scores),
                "occlusion_empty_scores": str(self.occlusion_empty_scores),
                "ris_explanation_list": str([item.to_dict() for item in self.ris_explanation_list])
                }

    def to_string(self):

        print(" ".join(self.tokens))
        print(self.tokens)
        print("Aspect: " + self.aspect + " --- Ground Truth: " + self.ground_truth + " --- Prediction: " + self.predicted_label)
        print("Confident score (pos, neg, neu, conflict): " + str(self.confidence_scores))
        print("\n")
