#!/usr/bin/env python3
'''
iaa_eval.py
webannoparser 
10/23/18
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import numpy as np
import itertools
from pathlib import Path
import pickle

# represent corpus as list of token ids:

# compute dice coefficient


def dice_coefficient(a, b):
    print(a, b)
    if not len(a) or not len(b): return 0.0
    """ quick case for true duplicates """
    if a == b: return 1.0
    """ if a != b, and a or b are single chars, then they can't possibly match """
    if len(a) == 1 or len(b) == 1: return 0.0

    """ use python list comprehension, preferred over list.append() """

    # assignments to save function calls
    lena = len(a)
    lenb = len(b)
    # initialize match counters
    matches = i = j = 0
    while (i < lena and j < lenb):
        if a[i] == b[j]:
            matches += 2
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    score = float(matches) / float(lena + lenb)
    return score

if __name__ == "__main__":


    FROM_SCRATCH = False
    opt_fp = "sentivent_en_webanno_project_iaa.pickle"

    iaa_ids = [
        "aapl14_iphone-x-s-dangerous-choice-of-market-share-or-profit.txt",
        "aapl15_here-s-how-apple-gets-to-a-2-trillion-market-value.txt",
        "aapl16_apple-s-app-store-generated-over-11-billion-in-revenue-for-th.txt",
        "amzn12_is-amazon-getting-into-the-pharmacy-business-this-is-what-you.txt",
        "amzn13_five-reasons-amazon-can-reach-1500.txt",
        "amzn14_amazon-sold-more-echo-dots-than-any-other-item-over-the-holidays.txt",
        "ba14_boeing-s-low-altitude-bid.txt",
        "ba15_should-boeing-buy-ge-aviation.txt",
        "ba16_boeing-s-stock-contributes-about-10-of-the-dow-s-1030-point.txt",
        "bac04_bank-of-america-earnings-hurt-by-tax-related-charge.txt",
        "bac05_bofa-includes-bitcoin-trust-in-broader-ban-on-investments.txt",
        "bac06_bank-of-america-hires-law-firm-to-help-probe-292-million-loss.txt",
        "cvx04_strong-crude-oil-no-help-for-chevron-exxon-mobil.txt",
        "cvx05_chevron-s-10-k-puts-the-permian-on-a-pedestal.txt",
        "cvx06_chevron-s-debt-fell-in-4q17-what-to-expect-in-2018.txt",
        "duk05_duke-energy-says-some-customers-may-be-affected-by-data-breach.txt",
        "duk06_like-many-of-its-peers-duk-is-trading-in-the-oversold-zone.txt",
        "duk07_duke-energy-stock-is-at-its-most-oversold-level-in-5-years.txt",
        "f13_ford-rolls-out-a-hot-rod-suv-as-drivers-abandon-performance-cars.txt",
        "f14_ford-is-the-next-ge-and-shorts-should-be-salivating.txt",
        "f15_ford-is-at-a-crossroad-of-danger-and-opportunity-in-china.txt",
        "jnj04_johnson-johnson-earnings-when-strong-is-nt-strong-enough.txt",
        "jnj05_where-s-the-tylenol-jj-disappoints-and-frustrates.txt",
        "jnj06_what-to-expect-from-johnson-johnson-in-2018.txt",
        "nem04_analyst-insight-is-newmont-mining-warming-up-for-a-good-2018.txt",
        "nem05_newmont-barrick-race-for-top-gold-crown-comes-down-to-a-decimal.txt",
        "nem06_newmont-mining-is-investors-gold-stock-to-buy.txt",
        "wmt05_walmart-stock-nears-key-support-after-earnings-miss.txt",
        "wmt06_goldman-expects-wal-mart-s-fortunes-to-improve-alongside-the-co.txt",
        "wmt07_walmart-s-meal-kits-are-not-the-solution-to-fight-amazon.txt",
    ]

    project_dirp = "exports/XMI_SENTiVENT-event-english-1_2018-10-04_1236"
    iaa_filter_func = lambda x: Path(x).parts[3] in iaa_ids and "anno" in Path(x).stem

    if not Path(opt_fp).is_file() or FROM_SCRATCH:
        # parse project
        event_project = WebannoProject(project_dirp)
        # filter so only iaa docs will be parsed
        event_project.annotation_document_fps = list(filter(iaa_filter_func, event_project.annotation_document_fps))
        event_project.parse_annotation_project()
        event_project.dump_pickle(opt_fp)
    else:
        with open(opt_fp, "rb") as project_in:
            event_project = pickle.load(project_in)

    attrib = "event_extent"
    annotations = {}
    key = lambda x: x.title
    event_project.documents.sort(key=key)
    for title, docs in itertools.groupby(event_project.documents, key):
        for doc in docs:
            mention_spans = [(i, x.event_extent + x.discontiguous_trigger_extent) if x.discontiguous_trigger_extent else (i, x.event_extent) for i, x in enumerate(doc.tokens) if x.event_extent]
            annotations.setdefault(doc.annotator_id, []).append(doc.tokens)

    combo = itertools.product(mention_spans_a, mention_spans_b)
    print([dice_coefficient(a, b) for (a, b) in combo])
