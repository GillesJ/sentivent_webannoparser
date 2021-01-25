'''
Script to analyse Dependency parses and PoS patterns in annotations using Spacy.
'''
from parse_project import parse_process_project
import settings
import spacy
import random
from spacy.symbols import nsubj, VERB, NOUN
from spacy import displacy
from spacy.tokens import Span
import pandas as pd
import numpy as np
from ast import literal_eval
from scipy import stats

random.seed(42)
Span.set_extension('unit', default=None, force=True)

def get_processed_span(project, ann_name):

    annotations = []
    if ann_name == "event_triggers":
        for event in project.get_events(subset="dev"):

            if event.friendly_id() not in ["jefdhondt_gs03_s5_FinancialReport.None-reports"]:
                # clean some errors
                # get trigger with discont tokens
                trigger_tokens = event.get_extent_tokens(extent=["discontiguous_triggers"])

                doc = event.get_processed_sentence()[0]

                spacy_span = doc[
                    trigger_tokens[0].index_sentence : trigger_tokens[-1].index_sentence + 1
                ]

                spacy_span._.unit = event

                yield spacy_span
            else: # skip the errors
                continue
    elif ann_name == "arguments":
        for argument in project.get_arguments(subset="dev"):
            doc = argument.get_processed_sentence()[0]
            span_tokens = argument.tokens
            spacy_span = doc[
                    span_tokens[0].index_sentence : span_tokens[-1].index_sentence + 1
                ]

            spacy_span._.unit = argument
            yield spacy_span

def get_info(span, unit):

    record = {}
    if unit == "event_trigger":
        record["event_type"] = span._.unit.event_type
    elif unit == "argument":
        record["role"] = span._.unit.role
    record["id"] = f"{span._.unit.friendly_id()}"
    spacy_attribs = ["text", "pos_", "tag_", "dep_", "shape_"]
    # for all tokens in span
    root = span.root
    for attrib in spacy_attribs:
        record[f"root-{attrib.replace('_', '')}"] = getattr(root, attrib)
        for t in span:
            record.setdefault(f"span-{attrib.replace('_', '')}", []).append(getattr(t, attrib))

    return record

def literal_eval_safe(x):
    try:
        return literal_eval(x)
    except Exception as e:
        print(e)
        return(np.nan)


if __name__ == "__main__":

    RELOAD = True
    # choices to make: parse from WebAnno annotation or with preproc -> reuse preproc code by loading as module
    # parse from raw Webanno and process with Spacy pipeline
    units = ["event_triggers", "arguments"]
    dfs = {}
    if RELOAD:
        print(f"Parsing raw WebAnno data in {settings.MASTER_DIRP}")
        project = parse_process_project(settings.MASTER_DIRP, from_scratch=False)

        for unit in units:
            parsed_spans = list(get_processed_span(project, unit))
            df = pd.DataFrame.from_records([get_info(s, unit) for s in parsed_spans])
            df.to_csv(f"{unit}_spans_allinfo.tsv", sep="\t")
            dfs[unit] = df

    for unit in units:
        dfs[unit] = pd.read_csv(f"{unit}_spans_allinfo.tsv", sep="\t",
                                converters={"span-text": literal_eval_safe,
                                            "span-dep": literal_eval_safe,
                                            "span-pos": literal_eval_safe,
                                            })
    # Get PoS + DeP info for tokens in annotation unit

    # get trigger stats
    for unit, df in dfs.items():
        print(f"ANALYSING {unit.upper()}")
        df_lengths = df["span-text"].apply(lambda x: len(x))
        print("- Span length: percentile")
        for i in range(df_lengths.max()):
            if (i+1 - 9) < 0 or (i+1) % 8 == 0 or i+1 == df_lengths.max():
                print(f"  - {i+1}: {stats.percentileofscore(df_lengths, i+1)}")

        # join span_dep
        df["span-dep_pos"] = [[f"{d}_{p}" for d, p in zip(*x)] for x in zip(df["span-dep"].tolist(), df["span-pos"].tolist())]
        df["root-dep_pos"] = df["root-dep"] + "_" + df["root-pos"]

        # create filter file
        df["root-dep_pos"].unique()
        # count values as frequencies
        # df = df.applymap(str)
        cols = ["span-dep", "span-pos", "span-dep_pos", "root-dep", "root-pos", "root-dep_pos"]
        cnts = {}
        for c in cols:
            df_cnt = df[c].value_counts(normalize=True)
            df_cnt.to_csv(f"{unit}_{c}.tsv", sep="\t")
            cnts[c] = df_cnt.shape[0]

        print(cnts)