""""
This script transforms original Wiki comment sentences to certain selected dialects.

Usage:
    python transformToDialect.py
    NOTE: only run in multi-VALUE repo root directory (https://github.com/SALT-NLP/multi-value)
"""

import pandas as pd
import json
from tqdm import tqdm

from src.Dialects import AfricanAmericanVernacular
from src.Dialects import NigerianDialect
from src.Dialects import IndianDialect
from src.Dialects import ColloquialSingaporeDialect


def transform_to_dialect(dialect, dfc, dialect_name):
    sents = []  # {text: ..., rules: [...]}

    print(f"working on {dialect_name} ...")
    # for i in tqdm(range(len(dfc)), desc="Processing"):  # run all 159,686 sentences
    for i in tqdm(range(500), desc="Processing"):  # sample size defined in range()
        wiki_sent = dfc["comment"][i]  # load original sentece

        sent_dict = {}
        sent_dict["text"] = dialect.convert_sae_to_dialect(wiki_sent)
        sent_dict["rules"] = list(
            set([i["type"] for i in dialect.executed_rules.values()])
        )

        sents.append(sent_dict)

    with open(f"{dialect_name}.jsonl", "w") as outfile:
        for entry in sents:
            json.dump(entry, outfile)
            outfile.write("\n")

    return True


if __name__ == "__main__":
    # read in original Wiki comments date with raw text
    dfc = pd.read_csv("toxicity_annotated_comments.tsv", sep="\t")

    # load and run AAVE transform module, save results
    aave = AfricanAmericanVernacular()
    transform_to_dialect(dialect=aave, dfc=dfc, dialect_name="aave")

    # load and run Nigerian dialect transform module, save results
    ngd = NigerianDialect()
    transform_to_dialect(dialect=ngd, dfc=dfc, dialect_name="nigerianD")

    # load and run Indian dialect transform module, save results
    indd = IndianDialect()
    transform_to_dialect(dialect=indd, dfc=dfc, dialect_name="indianD")

    # load and run Singlish dialect transform module, save results
    csgd = ColloquialSingaporeDialect()
    transform_to_dialect(dialect=csgd, dfc=dfc, dialect_name="singlish")
