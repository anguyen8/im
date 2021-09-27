

class ML_Object(object):
    def __init__(self, guid, tokens, input_ids, sentence_representation, attention_weights, confidence_score, pred_label, ground_truth):
        self.guid = guid
        self.tokens = tokens
        self.input_ids = input_ids
        self.sentence_representation = sentence_representation
        self.attention_weights = attention_weights
        self.confidence_score = confidence_score
        self.pred_label = pred_label
        self.ground_truth = ground_truth

        # Cosine similarity between shuffled vs original sentences.
        self.cos_sim = 0

    def get_guid(self):
        return self.guid

    def get_sentences(self):
        sentences = {}
        sentences["text_a"] = " ".join(self.tokens["text_a"])
        if len(self.tokens["text_b"]) > 0:
            sentences["text_b"] = " ".join(self.tokens["text_b"])

        return sentences

    def get_tokens(self):
        return self.tokens

    def get_original_indices(self):
        if isinstance(self.tokens[0], tuple):
            return [token[1] for token in self.tokens]

        return []

    def get_input_ids(self):
        return self.input_ids

    def get_sentence_representation(self):
        return self.sentence_representation

    def get_attention_weights(self):
        return self.attention_weights

    def get_confidence_score(self):
        return self.confidence_score

    def get_pred_label(self):
        return self.pred_label

    def get_ground_truth(self):
        return self.ground_truth

    def get_cos_sim(self):
        return self.cos_sim

    def set_cos_sim(self, cos_sim):
        self.cos_sim = cos_sim

    def __lt__(self, other):
        # return self.cos_sim < other.cos_sim
        return len(self.get_tokens()) < len(other.get_tokens())