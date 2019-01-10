#!/usr/bin/env python3
'''
corpus_stats_viz.py
webannoparser 
11/19/18
Copyright (c) Gilles Jacobs. All rights reserved.  
'''

from parser import *
import pygal
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import squarify  # pip install squarify (algorithm for treemap)
from functools import partial
from collections import Counter
from itertools import groupby
import pandas as pd
import util
from copy import deepcopy
import numpy as np
from pprint import pprint
from pygal.style import CleanStyle, DefaultStyle

pd.option_context('display.max_rows', None, 'display.max_columns', None)

def count_no_participants_defined(event_list):
    cnt = sum(1 for ev in event_list if not ev.participants)
    total = len(event_list)
    return (cnt, total, round(100 * float(cnt) / total, 2))

def check_role_in_participants(event, role):
    all_participants = []
    if event.participants is not None: all_participants.extend(event.participants)
    coref_participants = event.get_coref_attrib("participants")
    if coref_participants is not None:
        all_participants.extend(coref_participants)
    return any(p.role == role for p in all_participants)

def group_and_process(l, group_key, process_func):
    l_sorted = sorted(l, key=group_key)
    for k, g in groupby(l_sorted, key=group_key):
        g = list(g)
        yield k, process_func(g)

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

def median_index(l):
    avg = sum(l)/len(l)
    dist = [abs(x - avg) for x in l]
    idx = dist.index(min(dist))
    return idx

def get_percentage_counter(counter):

    n = sum(counter.values())
    return {k: {"pct": round(100.0 * v / n, 2), "n": v} for k, v in counter.items()}

def clean_project(proj):
    '''
    Clean the project from empty and redundant docs.
    :param proj:
    :return:
    '''
    # check documents with no annotations
    unclean_len = len(proj.annotation_documents)
    proj.annotation_documents = [d for d in proj.annotation_documents if d.events]
    clean_len = len(proj.annotation_documents)
    print(f"Removed {unclean_len - clean_len} docs without event annotations.")

    # check double titles (was an issue with opening documents to annotators in WebAnno interfaces)
    # title_cnt = Counter("".join(d.title.split("_")[1:]) for d in proj.annotation_documents)
    title_cnt = Counter(d.title for d in proj.annotation_documents)
    title_cnt = {k: v for k, v in title_cnt.items() if v > 1} # remove singles because not problematic
    # collect these double docs
    keep_docs = []
    single_annotated_docs = [] # for returning when manually correcting: we avoid manually correcting these until final selection method is decided.
    for title, docgroup in groupby(proj.annotation_documents, key=lambda x: x.title):
        docs = list(docgroup)
        if len(docs) > 1:
            docs.sort(key=lambda x: len(x.events), reverse=True)
            cnt = {d.annotator_id: len(d.events) for d in docs}
            # doc_most_events = max(docs, key=lambda x: len(x.events))
            # keep_docs.append(doc_most_events)
            # print(f"{title} selected {doc_most_events.annotator_id.upper()} {cnt}.")
            doc_average = docs[median_index([float(len(d.events)) for d in docs])]
            keep_docs.append(doc_average)
            print(f"{title} selected {doc_average.annotator_id.upper()} {cnt}.")
        else:
            keep_docs.append(docs[0])
            single_annotated_docs.append(docs[0])
    print(f"Removed {clean_len - len(keep_docs)} duplicate annotated docs by keeping docs with most events.")
    proj.annotation_documents = keep_docs

    return single_annotated_docs

def clean_events(evs):
    evs_to_clean = [deepcopy(ev) for ev in evs]
    type_replace = {"CapitalReturns": "Dividend", "FinancialResult": "FinancialReport"}
    clean_evs = []

    for ev in evs_to_clean:
    # TODO replace changed subtypes
        if ev.event_type is not None: # none types not included
            if ev.event_type in type_replace: # replace changed types
                print(f"WRONG TYPE FOUND: {ev.event_type} {ev.document_title}")
                ev.event_type = type_replace[ev.event_type]
            if ev.event_subtype is None: # replace none subtype by other
                ev.event_subtype = "Other"
            # # remove macroecon zonder participants
            # if not (ev.event_type == "Macroeconomics" and not check_role_in_participants(ev, "AffectedCompany")):
            #     clean_evs.append(ev)
            clean_evs.append(ev)

    return clean_evs


def plot_type_treemap_interactive(type_df, fp="type_treemap_pygal.svg"):
    # type treemap
    style = DefaultStyle(
        legend_font_size=12,
        tooltip_font_size=12,
    )
    treemap = pygal.Treemap(style=style, margin=0)
    treemap.title = "Event Type"

    for idx, row in type_df.iterrows():
        treemap.add(row.name, [row["pct"]])

    treemap.render_to_file(fp, print_values=False)

def plot_type_treemap_matplot(type_df, fp="type_treemap_matplot.png"):

    type_df["label"] = type_df.apply(lambda x: f"{x.name}\n{x['pct']}% (n={x['n'].astype('int')})", axis=1)

    figsize = [9, 5]
    plt.rcParams["figure.figsize"] = figsize
    cmap = plt.get_cmap("tab20", lut=len(type_df.index))
    # Change color
    fig = plt.figure(figsize=figsize, dpi=300)
    squarify.plot(sizes=type_df["n"], label=type_df["label"],
                  color=cmap.colors, alpha=.4, figure=fig)
    plt.title("Distribution of event categories in SENTiVENT English corpus.", fontsize=12, figure=fig)
    plt.axis('off', figure=fig)
    plt.show()
    fig.savefig(fp)

if __name__ == "__main__":

    # opt_fp = "sentivent_en_webanno_project_my_obj.pickle"
    opt_fp = "sentivent_en_webanno_correction.pickle"
    exclude_gilles = lambda x: "anno" in Path(x.path).stem

    with open(opt_fp, "rb") as project_in:
        event_project = pickle.load(project_in)

    single_annotated_docs = clean_project(event_project)

    avg_attribs = ["events", "sentences", "tokens", "participants", "fillers"]
    avg = {avg_attrib: count_avg([d for d in event_project.annotation_documents], avg_attrib, return_counts=True) for avg_attrib in avg_attribs}
    print(avg)

    all_events = []
    for d in event_project.annotation_documents:
        for ev in d.events:
            all_events.append(ev)

    all_events_clean = clean_events(all_events)

    avg_type_count = get_percentage_counter(Counter(ev.event_type for ev in all_events_clean))
    avg_subtype_count = get_percentage_counter(Counter(ev.event_subtype for ev in all_events_clean))
    for k, v in avg_subtype_count.items():
        v["type"] = k[0]
        v["subtype"] = k[1]
    print("Event types: ", avg_type_count)
    print("Event subtypes", avg_subtype_count)
    type_df = pd.DataFrame(avg_type_count).transpose().sort_values(by="pct", ascending=False)

    subtype_df = pd.DataFrame(avg_subtype_count).transpose().sort_values(by="pct")
    subtype_df[["type", "subtype", "n"]].to_csv("subtype_data.csv", index=False)

    plot_type_treemap_matplot(type_df)

    #create a list of event annotations that have changed and fix them

    event_getter = {"event_type": ["CapitalReturns", "FinancialResult",]}
    edits = [ev for ev in all_events if ev.event_type in ["CapitalReturns", "FinancialResult",]]
    gkey = lambda x: (x.document_title, x.annotator_id)
    doc_replace_cnt = [(title, len(list(g))) for title, g in groupby(sorted(edits, key=gkey), key=gkey)]
    for title, cnt in sorted(doc_replace_cnt, key=lambda x: x[1], reverse=True):
        print(title, cnt)

    # events with no participants
    cnt_no_participants = sum(1 for ev in all_events if not ev.participants)
    print(f"{cnt_no_participants}/{len(all_events)} ({round(100*float(cnt_no_participants)/len(all_events), 2)}%) event without participants.")
    group_key = lambda x: x.annotator_id
    all_events_sorted = sorted(all_events, key=group_key)
    for anid, evs in groupby(all_events_sorted, key=group_key):
        evs = list(evs)
        cnt_no_participants = sum(1 for ev in evs if not ev.participants)
        print(f"\t{anid} {cnt_no_participants}/{len(evs)} ({round(100 * float(cnt_no_participants) / len(evs),2)}%)")

    # parse histoplot of doc_length in sentences and words over amount of event annos
    df_event_cnt = pd.DataFrame({
        "title": [doc.title for doc in event_project.annotation_documents],
        "event_count": [len(doc.events) for doc in event_project.annotation_documents],
        "token_count": [len(doc.tokens) for doc in event_project.annotation_documents],
        "sentence_count": [len(doc.sentences) for doc in event_project.annotation_documents],
    })

    def get_company_info(document_title, column, company_df):
        if not document_title[:1].isdigit():
            tickersymbol = document_title.split("_")[0][:-2]
            company_info_row = company_df.loc[company_df['tickersymbol'] == tickersymbol]
            val = company_info_row[column].tolist()[0]
            return val
        else:
            return "Unresolved"
    comp_cnt = Counter(ev.document_title.split("_")[0][:-2] for ev in all_events)
    print(comp_cnt, len(comp_cnt))
    companies_df = pd.read_csv('corpus_companies.tsv', sep='\t', index_col=False)
    companies_df["tickersymbol"] = companies_df["corpussymbol"].str.lower()
    # count of events and docs per company
    # create document dataframe
    df_documents = pd.DataFrame({
        "document_title": [d.title for d in event_project.annotation_documents],
        "company": [get_company_info(d.title, "security", companies_df) for d in event_project.annotation_documents],
        "industry": [get_company_info(d.title, "industry", companies_df) for d in event_project.annotation_documents],
        "subindustry": [get_company_info(d.title, "subindustry", companies_df) for d in event_project.annotation_documents],
    })
    # create event dataframe
    df_events = pd.DataFrame({
        "event_type": [ev.event_type for ev in all_events],
        "event_subtype": [ev.event_subtype for ev in all_events],
        "annotator_id": [ev.annotator_id for ev in all_events],
        "document_title": [ev.document_title for ev in all_events],
        "company": [get_company_info(ev.document_title, "security", companies_df) for ev in all_events],
    # actually to many computations here use groupby to group tickers than look up once for company info
        "industry": [get_company_info(ev.document_title, "industry", companies_df) for ev in all_events],
        "subindustry": [get_company_info(ev.document_title, "subindustry", companies_df) for ev in all_events],
    })

    # count events and docs per industry
    df_industry_freq = pd.DataFrame({
        "doc_target_freq": {
            "Consumer Discretionary": 20.00,
            "Consumer Staples": 10.00,
            "Energy": 10.00,
            "Financials": 10.00,
            "Health Care": 10.00,
            "Industrials": 10.00,
            "Information Technology": 20.00,
            "Materials": 3.33,
            "Telecommunication Services": 3.33,
            "Utilities": 3.33,
        },
    })
    df_industry_freq["doc_counts"] = df_documents["industry"].value_counts()
    df_industry_freq["doc_freq"] = df_documents["industry"].value_counts(normalize=True) * 100
    df_industry_freq["doc_target_delta"] = df_industry_freq["doc_target_freq"] - df_industry_freq["doc_freq"]
    df_industry_freq["event_counts"] = df_events["industry"].value_counts()
    df_industry_freq["event_freq"] = df_events["industry"].value_counts(normalize=True) * 100
    # # count of events and docs per company
    df_company_freq = pd.DataFrame()
    df_company_freq["event_counts"] = df_documents["company"].value_counts()
    df_company_freq["event_freq"] = df_documents["company"].value_counts(normalize=True) * 100
    df_company_freq["doc_counts"] = df_documents["company"].value_counts()
    df_company_freq["doc_freq"] = df_documents["company"].value_counts(normalize=True) * 100

    print(df_company_freq)
    print(df_industry_freq)

    sns.set_style('darkgrid')

    sns.regplot(x="event_count", y="sentence_count", data=df_event_cnt, x_estimator = np.mean)
    plt.show()

    sns.regplot(x="event_count", y="sentence_count", data=df_event_cnt,
                x_estimator = np.mean, logx = True, truncate = True)
    plt.show()

    sns.regplot(x="event_count", y="token_count", data=df_event_cnt, x_estimator = np.mean)
    plt.show()

    sns.regplot(x="event_count", y="token_count", data=df_event_cnt,
                x_estimator = np.mean, logx = True, truncate = True)
    plt.show()

    sns.distplot(df_event_cnt["event_count"])
    plt.show()

    # TODO examine correlation of event_type and company

    # TODO examine correlation of event_type and industry

    # TODO examine correlation of event_type with event_type

