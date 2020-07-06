#!/usr/bin/env python3
"""
Inspect made annotations and check for common annotation mistakes and issues.
inspect_sentiment_annotations.py
webannoparser
5/13/20
Copyright (c) Gilles Jacobs. All rights reserved.
"""
import sys

sys.path.append("/home/gilles/repos/")

from sentivent_webannoparser.parser import WebannoProject
import sentivent_webannoparser.util as util
import sentivent_webannoparser.settings as settings
import sentivent_webannoparser.parse_project as pp
from itertools import groupby
from collections import Counter
import pandas as pd


def to_data(u):
    ud = u.__dict__
    to_add = {
        "annotation_unit": type(u).__name__,
        "object": u,
    }
    ud.update(to_add)
    return ud


def print_dict_itemlist(d, level=0):
    for k, v in d.items():
        indent = 2 * level * " "
        if not indent:  # prettify top level
            print("-------")
        if isinstance(v, dict):
            print(f"{indent}- {k}:")
            print_dict_itemlist(v, level=level + 1)
        else:
            print(f"{indent}- {k}: {', '.join(v)}")


def add_issue(issues, unit, issue_descr):
    if not isinstance(unit, str):
        doc_id = " ".join(unit.friendly_id().split(" ")[:2])
        u_id = unit.friendly_id().split(" ", 2)[-1].split("(", 1)[0]
    else:
        doc_id = unit.split("_")[0]
        u_id = unit.split("_")[1]
    issues.setdefault(doc_id, dict())
    issues[doc_id].setdefault(u_id, set()).add(issue_descr)


if __name__ == "__main__":

    # Create full clean corpus
    # project = pp.parse_project(settings.SENTIMENT_IAA)

    project = pp.parse_project(settings.SENTIMENT_ANNO)

    annotated_by_gilles_doc_ids = [
        "wfc00",
        "wfc01",
        "wfc02",
        "wfc03",
        "wmt03",
        "pg03",
        "pg04",
        "pg05",
    ]
    project.annotation_documents = [
        d
        for d in project.annotation_documents
        if d.annotator_id != "gilles" or d.document_id in annotated_by_gilles_doc_ids
    ]

    # dict for counting issues
    issues = {}  # {doc_id: anno_id: set(issues)}

    all_se = list(project.get_sentiment_expressions())
    all_ev = list(project.get_events())
    all_inst = all_se + all_ev

    # # check doubly annotated docs: None found
    # print("------\nDouble annotated docs:")
    # key_f = lambda x: x.title
    # i = 0
    # for k, g in groupby(sorted(list(project.annotation_documents), key=key_f), key_f):
    #     g = list(g)
    #     if len(g) > 1:
    #         annos = [d.annotator_id for d in g]
    #         i += 1
    #         print(f"{i}. {annos} annotated {k}")

    # Check Low annotation density (n anno units/n sentences):
    units = ["sentiment_expressions", "events"]
    for d in project.annotation_documents:
        cnt_u = [len(getattr(d, u)) for u in units]
        n_sen = len(d.sentences)
        den = sum(cnt_u) / n_sen
        doc_id = f"{d.annotator_id} {d.document_id}"
        if den <= 0.75:  # threshold manually set (1.0 was too sensitive)
            t = f"{', '.join(f'{c} {u}' for u, c in zip(units, cnt_u))}, {n_sen} sentences._{round(den,2)})"
            add_issue(issues, f"{doc_id}_doc-level", t)

    # print summary:
    # sort dict
    print("----------\nCheck manually for low annotation density:")
    kf = lambda i: list(list(i[1].values())[0])[0].split("_")[-1]
    issues = {k: v for k, v in sorted(issues.items(), key=kf)}
    print_dict_itemlist(issues)
    # check event and sentiment polarity annotation complete
    for x in all_inst:
        if not x.polarity_sentiment:
            add_issue(issues, x, "polarity_missing")

    # check all sentiment expressions have a target
    for se in all_se:
        if not se.targets:
            add_issue(issues, se, "target_missing")

    # check common token-based mistakes: polarity label it should be > tokens
    token_issues = {
        "positive": ["oversold", "hold", "increase", "benefit", "advantage"],
        "negative": ["danger" "overbought", "sell", "decrease", "issue", "problem"],
    }
    for correct_pol, tokens in token_issues.items():
        filter_f = lambda x: x.polarity_sentiment_scoped != correct_pol
        for x in filter(filter_f, all_inst):
            expr_pol = x.polarity_sentiment
            text = x.get_extent_text(extent=[]).lower()
            if any(token in text for token in tokens) and correct_pol != expr_pol:
                add_issue(
                    issues, x, f"possible polarity mismatch: {expr_pol}>{correct_pol}"
                )

    # print summary:
    # sort dict
    kf = lambda i: i[0].split(" ")[0] and sum(1 for x in i[1].values() for s in x)
    issues = {k: v for k, v in sorted(issues.items(), key=kf, reverse=True)}
    print_dict_itemlist(issues)

    #
    stats_df = get_stats(project)
