from transformers.processor_utils import InputExample

import spacy
nlp = spacy.load("en_core_web_sm")


class AttrToken(object):
    def __init__(self, token, score=0):
        self.token = token
        self.attr_score = score

        # ONLY FOR IM
        # List of candidates proposed by Masked Language Model (e.g., BERT, RoBERTa)
        self.candidates = []

    def get_token(self):
        return self.token

    def set_token(self, token):
        self.token = token

    def get_attr_score(self):
        return self.attr_score

    def set_attr_score(self, score):
        self.attr_score = score

    def set_candidates(self, candidates):
        self.candidates = candidates

    def get_candidates(self):
        return self.candidates


def parse_text(text):
    attr_token_list = []

    if text:
        doc = nlp(text)
        for token in doc:
            attr_token = AttrToken(token=token.text, score=0)
            attr_token_list.append(attr_token)

    return attr_token_list


class AttrExample(object):
    def __init__(self, guid="", text_a=None, text_b=None, label="", parsed_text_a=None):
        self.guid = guid
        self.text_a = parse_text(text_a) if parsed_text_a == None else parsed_text_a   # List of AttrToken objects
        self.text_b = parse_text(text_b)   # List of AttrToken objects
        self.label = label                 # Ground truth
        self.pred_label = ""               # Prediction
        self.conf_score = 0                # Confidence score
        self.attr_scores = []

    def get_attr_token_list_from_text_a(self):
        return self.text_a

    def get_text_a(self):
        return " ".join([attr_token.get_token() for attr_token in self.text_a])

    def get_attr_token_list_from_text_b(self):
        return self.text_b

    def get_text_b(self):
        return " ".join([attr_token.get_token() for attr_token in self.text_b])

    def get_all_attr_tokens(self):
        return self.get_attr_token_list_from_text_a() + self.get_attr_token_list_from_text_b()

    def get_ground_truth(self):
        return self.label

    def get_pred_label(self):
        return self.pred_label

    def set_pred_label(self, pred_label):
        self.pred_label = pred_label

    def get_confidence_score(self):
        return self.conf_score

    def set_confidence_score(self, conf_score):
        self.conf_score = conf_score

    def get_input_example(self, without_label=False):
        # without_label = True is for Masked Language Model ONLY
        return InputExample(text_a=self.get_text_a(), text_b=self.get_text_b(),
                            label=None if without_label else self.label, guid=self.guid)

    def get_input_example_for_deletion_method(self, del_flags):
        text_a = [token for idx, token in enumerate(self.text_a) if del_flags[:self.get_length_text_a()][idx]]
        text_b = [token for idx, token in enumerate(self.text_b) if del_flags[self.get_length_text_a():self.get_length()][idx]]

        text_a = " ".join([attr_token.get_token() for attr_token in text_a])
        text_b = " ".join([attr_token.get_token() for attr_token in text_b])

        return InputExample(text_a=text_a, text_b=text_b, label=self.label, guid=self.guid)

    def get_input_example_for_bert_based_deletion_method(self, model_wrapper, del_flags):
        # IMPORTANT: Currently, this function is used for SST-2 task only

        masked_token = model_wrapper.mask_token
        text_a = [token.get_token() if del_flags[:self.get_length_text_a()][idx] else masked_token
                  for idx, token in enumerate(self.text_a)]
        text_b = [token.get_token() if del_flags[self.get_length_text_a():self.get_length()][idx] else masked_token
                  for idx, token in enumerate(self.text_b)]

        text_a = model_wrapper.predict_multi_blanks_as_masked_lm_batch_encoded([" ".join(text_a)])[0]
        text_b = " ".join(text_b)

        return InputExample(text_a=text_a, text_b=text_b, label=self.label, guid=self.guid)

    def get_input_example_for_replacement_method(self, model_wrapper, masked_idx):
        masked_token = AttrToken(model_wrapper.mask_token)

        if masked_idx < len(self.text_a):
            old_attr_token = self.text_a[masked_idx]
            self.text_a[masked_idx] = masked_token
        else:
            old_attr_token = self.text_b[masked_idx - len(self.text_a)]
            self.text_b[masked_idx - len(self.text_a)] = masked_token

        # Get outputs for all masked examples at once.
        masked_outputs = model_wrapper.predict_as_masked_lm(examples=[self.get_input_example()], top_N=2)
        candidates = masked_outputs[0]          # Since the input list has 1 example only

        # candidates[0] is a list of words
        # candidates[1] is a list of their corresponding likelihoods.
        for replaced_word in candidates[0]:
            if replaced_word != old_attr_token.get_token():
                masked_token.set_token(replaced_word)
                break

        return self.get_input_example()

    def get_length(self):
        return len(self.text_a) + len(self.text_b)

    def get_length_text_a(self):
        return len(self.text_a)

    def get_length_text_b(self):
        return len(self.text_b)

    def set_masked_token(self, idx, mask="[MASK]"):
        token = (self.text_a + self.text_b)[idx]
        token.set_token(mask)

    def get_attribution_scores(self):
        return self.attr_scores

    def set_attribution_scores(self, attr_scores):
        assert len(attr_scores) == self.get_length()
        self.attr_scores = attr_scores

        for idx, token in enumerate(self.text_a + self.text_b):
            token.set_attr_score(attr_scores[idx])

    def generate_candidates_for_masked_token(self, masked_examples, model_wrapper, top_N=None, threshold=10e-5):
        # Find all the masked tokens
        masked_tokens = []
        input_examples = []

        for masked_example in masked_examples:
            for idx, token in enumerate(masked_example.get_attr_token_list_from_text_a() +
                                        masked_example.get_attr_token_list_from_text_b()):
                if token.get_token() in ["[MASK]", "<mask>"]:
                    masked_token = token
                    masked_token.set_token(model_wrapper.mask_token)
                    masked_tokens.append(masked_token)
                    break

            # Generate number (top_N) of input examples for a masked sentence
            input_examples.append(masked_example.get_input_example(without_label=True))

        # Get outputs for all masked examples at once.
        masked_outputs = model_wrapper.predict_as_masked_lm(examples=input_examples, top_N=top_N, threshold=threshold)

        input_examples.clear()
        for idx, candidates in enumerate(masked_outputs):
            # Set a list of candidates for the masked token and original token
            masked_tokens[idx].set_candidates(candidates)               # mask
            self.get_all_attr_tokens()[idx].set_candidates(candidates)  # original

            # Prepare input examples for the masked token
            for word, prob in zip(candidates[0], candidates[1]):
                text_a = masked_examples[idx].get_text_a().replace(model_wrapper.mask_token, word)
                text_b = masked_examples[idx].get_text_b().replace(model_wrapper.mask_token, word)
                input_example = InputExample(text_a=text_a, text_b=text_b, label=self.label, guid=self.guid)
                input_examples.append(input_example)

        return input_examples

    def generate_candidates_for_occ_token(self, masked_examples, replaced_token):
        # Find all the masked tokens
        input_examples = []

        for masked_example in masked_examples:
            for idx, token in enumerate(masked_example.get_attr_token_list_from_text_a() +
                                        masked_example.get_attr_token_list_from_text_b()):
                if token.get_token() in ["[MASK]", "<mask>"]:
                    token.set_token(replaced_token)
                    break

            # Generate number (top_N) of input examples for a masked sentence
            input_examples.append(masked_example.get_input_example())

        return input_examples

    def get_candidates(self):
        for token in self.text_a + self.text_b:
            if token.get_token() in ["[MASK]", "<mask>"]:
                return token.get_candidates()

        return []

