""""
This scripts is to replicate the error analyses conducted in /notebooks/spaCy_NER_on_CS.ipynb in the Repo.

Running process:
    - the data specified as file_path argument will be loaded on start
    - the tagging process starts consequently
    - choose the error analysis you want to run from list
    - type >> to end the script

Usage:
   python spaCy_NER_on_CS.py
"""

import pandas as pd
import spacy

from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")


def parse_file(file_path):
    # empty list to store DataFrames for each sentence
    corpus = []

    # read the CoNLL-U file line by line
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

        current_sentence = []
        columns = ["word", "lang", "entity_type"]
        for line in lines:
            if line.startswith("# sent_enum"):
                # if a new sentence begins, process the current one
                if current_sentence:
                    df = pd.DataFrame(current_sentence, columns=columns)
                    corpus.append(df)
                    current_sentence = []
            else:
                # append each line to the current sentence
                current_sentence.append(line.strip().split("\t"))

    # last sentence in the file
    if current_sentence:
        df = pd.DataFrame(current_sentence, columns=columns)
        corpus.append(df)

    return corpus


def most_frequent_element(lang_tags):
    # find the L1 by counting the most frequent lang tag in a sentence
    most_frequent = max(set(lang_tags), key=lang_tags.count)

    return most_frequent


def tag_eng_sent(model_eng, corpus, sent_idx):
    """tag a sentence with English as L1
    return a dictionary with language tags, gold NE tags and spacy NER results"""
    sent_df = corpus[sent_idx][:-1]  # remove last row resulted by CoNLL-U seperator

    sentence_text = sent_df["word"].str.cat(sep=" ")
    doc = model_eng(sentence_text)

    # run function to get result dictionary for current sentence
    results = extract_results(sent_df=sent_df, doc=doc)

    return results


def tag_spa_sent(model_spa, corpus, sent_idx):
    """tag a sentence with Spanish as L1
    return a dictionary with language tags, gold NE tags and spacy NER results"""
    sent_df = corpus[sent_idx][:-1]  # remove last row resulted by CoNLL-U seperator

    sentence_text = sent_df["word"].str.cat(sep=" ")
    doc = model_spa(sentence_text)

    # run function to get result dictionary for current sentence
    results = extract_results(sent_df=sent_df, doc=doc)

    return results


def extract_results(sent_df, doc):
    # extract all pre-processed tokens to a list
    gold_tokens = list(sent_df["word"])
    # regularize gold NER tags, save to list
    gold_tags = ["Yes" if tag != "O" else "O" for tag in list(sent_df["entity_type"])]
    # also save language tags
    gold_langs = list(sent_df["lang"])

    nes = [i.text for i in doc.ents]
    # flat the nes tokens
    nes_tokens = [
        item for sublist in [item.split() for item in nes] for item in sublist
    ]

    if len(nes_tokens) == 0:  # check if spaCy found any NE
        spacy_tags = ["O"] * len(sent_df)
    else:
        spacy_tags = []  # list to store spaCy NER results
        for token in gold_tokens:
            if len(nes_tokens) != 0:
                if token in nes_tokens[0] or nes_tokens[0] in token:
                    spacy_tags.append("Yes")
                    nes_tokens = nes_tokens[1:]
                else:
                    spacy_tags.append("O")
            else:
                spacy_tags.append("O")

    # format results to a dictionary
    results = {
        "mlang": "spa",
        "lang": gold_langs,
        "true_ne": gold_tags,
        "spacy_ne": spacy_tags,
    }

    return results


def l2_as_fne(ner_results):
    # target_word_idxs: list of indices of inserted L2 tokens that are not NEs in each sentence
    target_word_idxs = []
    for result in ner_results:
        if result["mlang"] == "eng":
            # get CS Spanish token index
            cs_idx = [
                i for i in range(len(result["lang"])) if result["lang"][i] == "lang2"
            ]
            # remove CS Spanish tokens that are actually NEs
            cs_ne_idx = [idx for idx in cs_idx if result["true_ne"][idx] == "O"]
            target_word_idxs.append(cs_ne_idx)
        elif result["mlang"] == "spa":
            # get CS English token index
            cs_idx = [
                i for i in range(len(result["lang"])) if result["lang"][i] == "lang1"
            ]
            # remove CS English tokens that are actually NEs
            cs_ne_idx = [idx for idx in cs_idx if result["true_ne"][idx] == "O"]
            target_word_idxs.append(cs_ne_idx)

    cs_fauxne = []  # [(CS tokens count, CS tokens tagged as NE count) of sent_1, ...]
    # get from spaCy falsely tagged inserted L2 tokens
    for i in range(len(target_word_idxs)):
        if len(target_word_idxs[i]) > 0:
            cs_count = len(target_word_idxs[i])
            cs_as_ne_count = len(
                [j for j in target_word_idxs[i] if ner_results[i]["spacy_ne"][j] != "O"]
            )
            cs_fauxne.append((cs_count, cs_as_ne_count))

    all_cs_count = sum(t[0] for t in cs_fauxne)
    all_cs_as_ne_count = sum(t[1] for t in cs_fauxne)

    print(all_cs_as_ne_count / all_cs_count)

    return True


def fne_is_l2(ner_results):
    # target_word_idxs: list of indices of falsely tagged tokens by spaCy
    target_word_idxs = []
    for result in ner_results:
        spacy_wrong_ne_idx = [
            i
            for i, (elem1, elem2) in enumerate(
                zip(result["spacy_ne"], result["true_ne"])
            )
            if elem1 != elem2
        ]
        target_word_idxs.append(spacy_wrong_ne_idx)

    fauxne_at_cs = (
        []
    )  # [(falsely tagged NE count, error on CS position count) of sent_1, ...]
    for i in range(len(target_word_idxs)):
        if len(target_word_idxs[i]) > 0:
            fauxne_count = len(target_word_idxs[i])
            sentence = ner_results[i]

            if sentence["mlang"] == "eng":
                fauxne_at_cs_count = len(
                    [j for j in target_word_idxs[i] if sentence["lang"][j] == "lang2"]
                )
            elif sentence["mlang"] == "spa":
                fauxne_at_cs_count = len(
                    [j for j in target_word_idxs[i] if sentence["lang"][j] == "lang1"]
                )
            fauxne_at_cs.append((fauxne_count, fauxne_at_cs_count))

    all_fauxne_count = sum(t[0] for t in fauxne_at_cs)
    all_fauxne_at_cs_count = sum(t[1] for t in fauxne_at_cs)

    print(all_fauxne_at_cs_count / all_fauxne_count)

    return True


def l2ne_as_ne(ner_results):
    # target_word_idxs: list of indices of inserted L2 words that are NEs by gold standard
    target_word_idxs = []
    for result in ner_results:
        if result["mlang"] == "eng":
            # get CS Spanish token index
            cs_idx = [
                i for i in range(len(result["lang"])) if result["lang"][i] == "lang2"
            ]
            # keep CS Spanish tokens that are actually NEs
            cs_ne_idx = [idx for idx in cs_idx if result["true_ne"][idx] != "O"]
            target_word_idxs.append(cs_ne_idx)
        elif result["mlang"] == "spa":
            # get CS English token index
            cs_idx = [
                i for i in range(len(result["lang"])) if result["lang"][i] == "lang1"
            ]
            # keep CS English tokens that are actually NEs
            cs_ne_idx = [idx for idx in cs_idx if result["true_ne"][idx] != "O"]
            target_word_idxs.append(cs_ne_idx)

    csne_as_ne = []  # [(L2 tokens = NE count, NE-L2 tokens as NE count) of sent_1, ...]
    for i in range(len(ner_results)):
        if len(target_word_idxs[i]) > 0:
            l2ne_count = len(target_word_idxs[i])
            l2ne_as_ne_count = len(
                [
                    j
                    for j in target_word_idxs[i]
                    if ner_results[i]["spacy_ne"][j] == "Yes"
                ]
            )

            csne_as_ne.append((l2ne_count, l2ne_as_ne_count))
    all_l2ne_count = sum(t[0] for t in csne_as_ne)
    all_l2ne_as_ne_count = sum(t[1] for t in csne_as_ne)

    print(all_l2ne_as_ne_count / all_l2ne_count)

    return True


def select_error_an(ner_results):
    function_selection = input(
        """\nRun an error analysis by number, "all" to run all, ">>" to quit:
1. How many inserted normal non-NE L2 words are falsely tagged as named entities? 
2. How many falsely tagged tokens are actually normal inserted non-NE L2 words?
3. How many inserted L2 tokens that are actually NEs are successfully extracted by L1 model?
Run: """
    )
    if function_selection not in ">>>>>" and function_selection[0].lower() != "a":
        try:
            function_selection = int(function_selection)
            # only three analyses are avaiable
            if function_selection < 4 and function_selection > 0:
                if int(function_selection) == 1:
                    l2_as_fne(ner_results=ner_results)
                elif int(function_selection) == 2:
                    fne_is_l2(ner_results=ner_results)
                elif int(function_selection) == 3:
                    l2ne_as_ne(ner_results=ner_results)
            else:
                print("Input out of range!")
        except ValueError:
            print("Invalid input!")
        return False
    elif function_selection[0].lower() == "a":
        # directly call all three functions
        l2_as_fne(ner_results=ner_results)
        fne_is_l2(ner_results=ner_results)
        l2ne_as_ne(ner_results=ner_results)
        return False
    else:
        # stop script if has >> as input
        return True


if __name__ == "__main__":
    # read in file, parse as a list of dataframe
    file_path = "../data/train.conll"
    print(f"reading in file: {file_path}")
    corpus = parse_file(file_path=file_path)
    # each sentence cann now be called by corpus[idx]

    # load spaCy models for both target languages
    model_eng = spacy.load("en_core_web_sm")
    model_spa = spacy.load("es_core_news_sm")

    # run spaCy model depending on the L1 to extract NEs
    ner_results = []
    print("starting named entity extraction ...")
    for i in tqdm(range(len(corpus)), desc="Processing"):
    # for i in tqdm(range(1000), desc="Processing"):
        lang_tags = list(corpus[i]["lang"])
        # make sure the sentence is code-mixed
        if "lang1" in lang_tags and "lang2" in lang_tags:
            # find the dominant language (lang1=eng, lang2=spa)
            mlang = most_frequent_element(lang_tags=lang_tags)
            if mlang == "lang1":
                ner_results.append(
                    tag_eng_sent(model_eng=model_eng, corpus=corpus, sent_idx=i)
                )
            else:
                ner_results.append(
                    tag_spa_sent(model_spa=model_spa, corpus=corpus, sent_idx=i)
                )

    # choose an error analysis function
    while True:
        to_stop = select_error_an(ner_results=ner_results)
        if to_stop == True:
            break
