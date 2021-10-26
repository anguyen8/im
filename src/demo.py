import argparse
import copy
import numpy as np
import datetime

from data_processors.data_objects.attribution_example import AttrExample
from data_processors.data_objects.attribution_visualization import AttributionVisualization
from src.model_wrapper import ModelWrapper

from pdflatex import PDFLaTeX


# Compute attribution score
def odds(score):
    return score / (1.0 - score)


def compute_input_margination(masked_examples, conf_scores, topN=None):
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
    return attr_scores, ori_pred, ori_conf_score


def highlight_text_binary_mode(analyzer, tokens, high_attr_tokens, target_label):

    # token's foreground is black, token's background is orange/yellow
    format = ';'.join([str(0), str(30), str(43)])
    highlight_tokens = copy.deepcopy(tokens)

    for idx, token in enumerate(tokens):
        highlight_tokens[idx] = '\x1b[%sm%s\x1b[0m' % (format, token) if idx in high_attr_tokens else token

    print("\nTarget label: {}".format(target_label))
    print(analyzer + ": " + " ".join(highlight_tokens) + "\n")     # highlight important words


def highlight_text_real_valued_mode(ori_example, target_label, looEmpty_scores, im_scores):
    def isEnglish(s):
        try:
            s.encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            return False
        else:
            return True

    def clean_word_list_for_latex(word_list):
        new_word_list = []

        for word in word_list:
            if word == "":
                word = " "

            if not isEnglish(word):
                word = "non_english"

            for latex_sensitive in ["\\", "%", "&", "^", "#", "_", "{", "}", "$", "#"]:
                if latex_sensitive in word:
                    word = word.replace(latex_sensitive, '\\' + latex_sensitive)

            new_word_list.append(word)

        return new_word_list

    def get_latex_string_words_with_weights(title, rows, cols, attr_viz, attribute="", print_weights=True):

        if isinstance(attr_viz.get_word_list(), dict):
            word_list = attr_viz.get_word_list()[attribute] if attribute in attr_viz.get_word_list() else []
        else:
            word_list = attr_viz.get_word_list()
        word_num = len(word_list)

        if title == "LOOEmpty":
            weights = attr_viz.get_looEmpty_scores()
            if attribute == "text_b":
                weights = weights[len(attr_viz.get_word_list()["text_a"]):]
                assert len(weights) == len(attr_viz.get_word_list()[attribute])
        elif title == "IM":
            weights = attr_viz.get_im_scores()
            if attribute == "text_b":
                weights = weights[len(attr_viz.get_word_list()["text_a"]):]
                assert len(weights) == len(attr_viz.get_word_list()[attribute])
        else:
            print("WRONG TITLE")
            return ""

        if weights is None or len(weights) == 0:
            return ""

        # Normalize and round numbers in weights
        weights = weights / max(abs(weights))
        weights = [round(num, 5) for num in weights]

        total_rows = rows * 2 if print_weights else rows
        string = "\\multirow{" + str(total_rows) + "}{*}{" + title + "\n" + attribute.replace("_", "\_") + "}"
        for iRow in range(total_rows):
            if print_weights:
                start_index = cols * (iRow // 2)
                end_index = min(word_num, cols * (iRow // 2 + 1))
            else:
                start_index = cols * iRow
                end_index = min(word_num, cols * (iRow + 1))

            idx = start_index

            # Print words if iRow is an even
            if iRow % 2 == 0 or not print_weights:

                while idx < end_index:
                    if weights[idx] >= 0:
                        string += " & " + "\\colorbox{%s!%s}{" % ('orange', weights[idx] * 100) + "\\strut " + word_list[idx] + "}"
                    else:
                        string += " & " + "\\colorbox{%s!%s}{" % ('blue', weights[idx] * (-100)) + "\\strut " + word_list[idx] + "}"

                    idx += 1
                string += " \\" + "\\" + "\n"

            # Print weights if iRow is an odd
            else:
                while idx < end_index:
                    string += " & " + str(weights[idx])
                    idx += 1
                string += " \\" + "\\ "

                if iRow < total_rows - 1:
                    string += "\\cline{2-" + str(cols + 1) + "}" + "\n"
                else:
                    string += "\\hline" + "\n\n"

            if not print_weights:
                if iRow < total_rows - 1:
                    string += "\\cline{2-" + str(cols + 1) + "}" + "\n"
                else:
                    string += "\\hline" + "\n"

        return string

    def generate_multiple_sentences_with_weights_for_analyzers(attribution_list, latex_file):
        with open(latex_file, 'w') as f:
            f.write(r'''\documentclass{article}
    \usepackage{geometry}
    \geometry{a4paper, portrait, margin=0.5in}
    \usepackage{color}
    \usepackage{tcolorbox}
    \usepackage{CJK}
    \usepackage{adjustbox}
    % Please add the following required packages to your document preamble:
    \usepackage{multirow}
    \usepackage{graphicx}
    \usepackage{float}
    \newcommand{\mybox}[2]{{\color{#1}\fbox{\normalcolor#2}}}
    \tcbset{width=0.85\textwidth,boxrule=0pt,colback=red,arc=0pt,auto outer arc,left=0pt,right=0pt,boxsep=5pt}
    \begin{document}
    \begin{CJK*}{UTF8}{gbsn}''' + '\n')

            string = ""

            for i in range(len(attribution_list)):

                attr_viz = attribution_list[i]

                string += "\\textbf{\\textcolor{red}{Example " + str(i) + "}}\n\n"
                string += "Prediction: " + str(attr_viz.get_prediction(escape=True)) + "\n\n"
                string += "Confidence scores: " + str(round(attr_viz.get_confidence_scores(), 4)) + "\n\n"

                attributes = ["text_a", "text_b"]
                for attribute in attributes:
                    word_num = len(attr_viz.get_word_list()[attribute]) if attribute in attr_viz.get_word_list() else 0
                    if word_num == 0:
                        continue

                    cols = 15 if word_num >= 15 else word_num
                    rows = word_num // cols + (0 if word_num % cols == 0 else 1)

                    attr_viz.set_word_list_with_attribute(
                        clean_word_list_for_latex(attr_viz.get_word_list()[attribute]), attribute)

                    string += "\\begin{table}[H]" + "\n"
                    string += "\\centering" + "\n"
                    string += "\\def\\arraystretch{1.5}" + "\n"
                    string += "\\resizebox{\\textwidth}{!}{%" + "\n"
                    string += "\\begin{tabular}{|l|" + "c|" * cols + "}" + "\n"
                    string += "\\hline" + "\n"

                    # Attribution scores
                    string += get_latex_string_words_with_weights("LOOEmpty", rows, cols, attr_viz, attribute=attribute)
                    string += get_latex_string_words_with_weights("IM", rows, cols, attr_viz, attribute=attribute)

                    string += "\\end{tabular}%" + "\n"
                    string += "}" + "\n"
                    string += "\\end{table}"

                    string += "\n\n"

            f.write(string + '\n')
            f.write(r'''\end{CJK*}
    \end{document}''')

    all_attr_scores, all_tokens = [], []
    logits, pred_labels, ground_truths = [], [], []
    attr_scores, tokens = [], {"text_a": [], "text_b": []}

    for attr_token in ori_example.get_attr_token_list_from_text_a():
        attr_scores.append(attr_token.get_attr_score())
        tokens["text_a"].append(attr_token.get_token())

    for attr_token in ori_example.get_attr_token_list_from_text_b():
        attr_scores.append(attr_token.get_attr_score())
        tokens["text_b"].append(attr_token.get_token())

    all_attr_scores.append(attr_scores)
    all_tokens.append(tokens)
    logits.append(ori_example.get_confidence_score())
    pred_labels.append(target_label)

    attribution_viz_list = []
    for idx, (tokens, pred_label, logit, attr_scores) in enumerate(zip(all_tokens, pred_labels, logits, all_attr_scores)):
        attribution_viz = AttributionVisualization(tokens, pred_label, logit, "", looEmpty_scores, im_scores)
        attribution_viz_list.append(attribution_viz)

    # Visualize results
    file_path = "../data/attribution_maps/" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + ".tex"
    generate_multiple_sentences_with_weights_for_analyzers(attribution_viz_list, file_path)
    print("Finish generating latex file of the real-valued attribution maps. \n"
          "You need to convert the file to PDF to view it")

    '''
    ThangPM: Currently, we cannot support output pdf file since there's an issue with the library `pdflatex`
    Thus, you need to convert the tex file to pdf file by your own to see the real-valued heatmap.
    We will try to fix it as soon as possible.
    '''
    # print("Converting tex to pdf file... " + file_path)
    # pdfl = PDFLaTeX.from_texfile(file_path)
    # pdfl.add_args({"-output-directory": "../data/attribution_maps/"})
    # pdfl.create_pdf(keep_pdf_file=True, keep_log_file=False)
    # os.remove(file_path) # Remove tex file


def visualize_attribution_map(analyzer, ori_example, masked_examples, model_wrapper, theta):
    if analyzer not in ["LOOEmpty", "IM"]:
        raise "Attribution method {} is not supported".format(analyzer)

    task_name = model_wrapper.data_args.task_name
    label_list = {"sst": ["negative", "positive"],
                  "sst-2": ["negative", "positive"],
                  "esnli": ["entailment", "neutral", "contradiction"],}

    input_examples = [ori_example.get_input_example()]

    # LOOEmpty
    if analyzer == "LOOEmpty":
        filled_examples = ori_example.generate_candidates_for_LOO_token(masked_examples, replaced_token="")
    # IM
    else:
        filled_examples = ori_example.generate_candidates_for_masked_token(masked_examples, model_wrapper, threshold=10e-5, top_N=10)

    input_examples.extend(filled_examples)
    conf_scores = model_wrapper.predict_proba_with_examples(input_examples)

    if analyzer == "LOOEmpty":
        ori_pred = np.argmax(conf_scores[0])
        ori_conf_score = conf_scores[0][ori_pred]
        masked_conf_scores = [conf_score[ori_pred] for conf_score in conf_scores[1:]]
        attr_scores = np.log2(odds(ori_conf_score)) - np.log2([odds(conf_score) for conf_score in masked_conf_scores])
    else:
        attr_scores, ori_pred, ori_conf_score = compute_input_margination(masked_examples, conf_scores)

    # Normalize + Zero-out and binarize
    if max(attr_scores) > 0:
        normalized_attr_scores = attr_scores / max(attr_scores)
        normalized_attr_scores = [1 if score >= theta else 0 for score in normalized_attr_scores]
    else:
        normalized_attr_scores = [0] * len(attr_scores)

    target_label = label_list[task_name][ori_pred]
    high_attr_tokens = [idx for idx, attr_score in enumerate(normalized_attr_scores) if attr_score == 1]
    highlight_text_binary_mode(analyzer=analyzer, tokens=all_tokens, high_attr_tokens=high_attr_tokens, target_label=target_label)

    return attr_scores, target_label, ori_conf_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive demo for attribution methods LOOEmpty and IM')
    parser.add_argument('--text_a', dest='text_a', required=True,
                        help='Provide the first input for running attribution methods')
    parser.add_argument('--text_b', dest='text_b', default="",
                        help='Provide the second input for running attribution methods')
    parser.add_argument('--theta', dest='theta', default=0.05,
                        help='Provide a threshold to binarize attribution maps')

    args, unknown_args = parser.parse_known_args()

    ori_example = AttrExample(text_a=args.text_a, text_b=args.text_b, label=None, guid=0)
    all_tokens = [x.token for x in ori_example.get_all_attr_tokens()]

    masked_examples = []
    for idx in range(ori_example.get_length()):
        masked_attr_example = copy.deepcopy(ori_example)
        masked_attr_example.set_masked_token(idx, mask="[MASK]")
        masked_examples.append(masked_attr_example)

    model_wrapper = ModelWrapper()
    looEmpty_scores, target_label, conf_score = visualize_attribution_map(analyzer="LOOEmpty",
                                                                          ori_example=copy.deepcopy(ori_example),
                                                                          masked_examples=copy.deepcopy(masked_examples),
                                                                          model_wrapper=model_wrapper,
                                                                          theta=args.theta)

    im_scores, _, _ = visualize_attribution_map(analyzer="IM",
                                                ori_example=copy.deepcopy(ori_example),
                                                masked_examples=copy.deepcopy(masked_examples),
                                                model_wrapper=model_wrapper,
                                                theta=args.theta)

    ori_example.set_confidence_score(conf_score)
    highlight_text_real_valued_mode(ori_example=copy.deepcopy(ori_example),
                                    target_label=target_label,
                                    looEmpty_scores=looEmpty_scores,
                                    im_scores=im_scores,)

