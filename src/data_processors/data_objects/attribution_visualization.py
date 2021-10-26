

class AttributionVisualization(object):

    def __init__(self, tokens, predicted_label, confidence_scores, ground_truth, looEmpty_scores, im_scores):

        self.tokens = tokens # dictionary
        self.all_tokens = [] # list

        self.ground_truth = ground_truth
        self.predicted_label = predicted_label
        self.confidence_scores = confidence_scores

        self.looEmpty_scores = looEmpty_scores
        self.im_scores = im_scores

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

    def get_looEmpty_scores(self):
        return self.looEmpty_scores

    def get_im_scores(self):
        return self.im_scores

