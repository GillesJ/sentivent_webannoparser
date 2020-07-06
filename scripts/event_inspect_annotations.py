#!/usr/bin/env python3
"""
Script to inspect common issues in annotation for aiding in manual corpus adjudication and correction.
event_inspect_annotations.py
webannoparser 
12/12/18
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
from pathlib import Path
import pickle
import corpus_stats_viz
from collections import Counter
from itertools import groupby
from parser import *
import util
import pandas as pd
import json
from fuzzywuzzy import fuzz
import settings
from parse_project import parse_project
from fuzzywuzzy.utils import full_process
from fuzzywuzzy import process as fuzzprocess

fully_corrected = [  # full manual correction and adjudication done on these docs
    "aal04_american-airlines-backtracks-lrb-a-bit-rrb-on-its-legroom-re.txt",
    "ba00_boeing-delivers-its-first-737-max-just-as-planned.txt",
    "celg00_why-shares-of-biopharma-giant-celgene-lrb-celg-rrb-tumbled-t.txt",
    "f03_may-auto-sales-results-reveal-a-new-leader-among-detroit-three.txt",
    "ge02_a-simple-ge-trade-for-long-suffering-investors.txt",
    "kmi02_why-long-term-investors-should-prefer-kinder-morgan-inc-over-ene.txt",
    "aal00_american-airlines-up-on-record-april-traffic-upbeat-q2-view.txt",
    "celg02_celgene-s-stock-may-be-incredibly-overvalued-right-now-here.txt",
    "celg04_why-celgene-is-bucking-biotech-weakness-today.txt",
    "f02_ford-ekes-out-a-sales-gain-on-pickups-and-fleet-deliveries.txt",
    "ge01_ge-drops-after-immelt-predicts-challenges-to-hit-2018-targets.txt",
    "ge04_ge-to-replace-jeff-immelt-with-ge-healthcare-exec-john-flannery.txt",
    "kmi01_what-enterprise-products-could-offer-income-investors.txt",
    "02_how-safe-is-chevrons-dividend---the-motley-fool.txt",
    "03_procter--gamble-co.s-proxy-fight-what-investors-need-to-know---the-motley-fool.txt",
    "05_netflixs-stock-is-worth-only-about-onethird-of-where-it-trades-today.txt",
    "aal03_american-airlines-reports-load-factor-increase-in-may-shares-g.txt",
    "dis01_this-hedge-fund-bought-250-million-of-disney-stock-should-yo.txt",
    "ge00_what-s-behind-ge-s-move-from-the-connecticut-suburbs-to-boston.txt",
    "nem01_inside-barrick-gold-s-production-growth.txt",
    "nem02_newmont-mining-ready-to-shine.txt",
    "nem03_newmont-s-lower-margins-understanding-the-analyst-predictions.txt",
    "fox03_weekend-box-office-wonder-woman-opened-even-bigger-than-we.txt",
    "cvx02_chevron-s-management-looks-like-it-s-changing-its-capital-spen.txt",
    "ge13_ge-is-beating-some-headwinds-but-is-it-enough.txt",
    "fb05_is-the-market-undervaluing-facebook-stock.txt",
    "gm02_general-motors-sales-slip-on-rental-fleet-cutbacks.txt",
    "gm03_general-motors-cadillac-sales-are-booming-in-china.txt",
    "nflx01_netflix-s-stock-is-worth-only-about-one-third-of-where-it-trade.txt",
    "dis02_disney-earnings-give-a-look-at-how-espn-is-facing-the-future.txt",
    "gm01_is-gm-abandoning-its-future-for-short-term-profits.txt",
    "dis10_stocks-in-red-that-might-produce-some-green.txt",
    "fb09_facebook-pitches-brand-safety-ahead-of-video-ad-push.txt",
    "abbv04_is-this-dividend-aristocrat-ridiculously-undervalued.txt",
    "aal12_airlines-have-a-cost-problem-lrb-and-it-s-not-12-snack-boxe.txt",
    "ba06_aircraft-leasing-firms-lift-boeing-s-dreamliner-higher-at-the-p.txt",
    "ba01_boeing-s-t-x-could-be-good-news-for-struggling-st-louis-econom.txt",
    "cvx00_chevron-investors-back-off-from-climate-change-proposal.txt",
    "gs03_goldman-sachs-continues-retail-banking-push-with-higher-savings.txt",
    "gs04_goldman-sachs-goes-mr-roboto-on-debt-sales.txt",
    "amzn02_3-reasons-why-amazon-is-scary-to-netflix.txt",
    "aapl13_stocks-mixed-this-california-builder-breaks-out-what-to-watc.txt",
    "fb06_facebook-posts-strong-q2-as-arpu-grows-across-geographies.txt",
    "wmt03_wal-mart-s-sam-s-club-reinvents-its-private-brand.txt",
    "fox02_fox-msnbc-neck-and-neck-in-ratings-as-viewers-monitor-trump.txt",
    "gm06_gm-doubles-down-on-diesel-as-2018-equinox-gets-39-mpg-rating-5.txt",
    "amd00_how-well-has-amd-delivered-on-their-2015-financial-analyst-day-p.txt",
    "ge03_why-ge-will-not-be-impacted-by-us-withdrawal-from-paris-climat.txt",
    "dis11_disney-earnings-buying-fox-would-nt-a-bad-idea.txt",
    "nem00_are-gold-miners-on-track-to-achieve-2017-production-guidance.txt",
    "dis12_disney-s-fiscal-2017-in-review.txt",
    "abbv11_abbvie-overpromises.txt",
    "amd11_reviews-confirm-with-radeon-rx-vega-amd-re-enters-the-performa.txt",  # from here manually checked with keywords and typology errors
    "dis08_disney-stock-defies-ho-hum-earnings.txt",
    "aapl03_will-the-iphone-8-help-apple-boost-its-market-share.txt",
    "aapl12_apple-s-stock-jumps-after-keybanc-upgrade-to-buy-rating.txt",
    "gm09_will-gm-s-top-selling-suv-take-a-hit-from-canadian-strike.txt",
    "abbv06_abbvie-stock-just-hit-a-52-week-high-here-s-why-it-should-go.txt",
    "cmg02_why-you-re-smart-to-buy-chipotle.txt",
    "amd01_why-amd-investors-should-nt-lose-faith.txt",
    "aapl01_here-are-the-six-dow-stocks-experts-say-have-the-most-upside.txt",
    "ge09_ge-s-expected-q3-profit-surge-may-be-overshadowed-by-dividend-f.txt",
    "cost00_costco-makes-gains-despite-retailer-gloom-with-upbeat-earnings.txt",
    "nflx11_netflix-stock-seesaws-after-third-quarter-subscriber-beat.txt",
    "nflx04_why-analysts-expect-netflix-s-global-membership-base-to-grow.txt",
    "amzn07_amazoncom-just-crushed-department-store-stocks-again-with-prime.txt",
    "dis07_disney-joins-ar-fray-with-200-star-wars-ar-headset.txt",
    "goog12_with-its-india-first-approach-google-is-trying-to-woo-the-comm.txt",
    "aapl14_iphone-x-s-dangerous-choice-of-market-share-or-profit.txt",  # gold standard
    "aapl15_here-s-how-apple-gets-to-a-2-trillion-market-value.txt",  # gold standard
    "aapl16_apple-s-app-store-generated-over-11-billion-in-revenue-for-th.txt",  # gold standard
    "amzn12_is-amazon-getting-into-the-pharmacy-business-this-is-what-you.txt",  # gold standard
    "amzn13_five-reasons-amazon-can-reach-1500.txt",  # gold standard
    "amzn14_amazon-sold-more-echo-dots-than-any-other-item-over-the-holidays.txt",  # gold standard
    "ba14_boeing-s-low-altitude-bid.txt",  # gold standard
    "ba15_should-boeing-buy-ge-aviation.txt",  # gold standard
    "ba16_boeing-s-stock-contributes-about-10-of-the-dow-s-1030-point.txt",  # gold standard
    "bac04_bank-of-america-earnings-hurt-by-tax-related-charge.txt",  # gold standard
    "bac05_bofa-includes-bitcoin-trust-in-broader-ban-on-investments.txt",  # gold standard
    "bac06_bank-of-america-hires-law-firm-to-help-probe-292-million-loss.txt",  # gold standard
    "cvx04_strong-crude-oil-no-help-for-chevron-exxon-mobil.txt",  # gold standard
    "cvx05_chevron-s-10-k-puts-the-permian-on-a-pedestal.txt",  # gold standard
    "cvx06_chevron-s-debt-fell-in-4q17-what-to-expect-in-2018.txt",  # gold standard
    "duk05_duke-energy-says-some-customers-may-be-affected-by-data-breach.txt",  # gold standard
    "duk06_like-many-of-its-peers-duk-is-trading-in-the-oversold-zone.txt",  # gold standard
    "duk07_duke-energy-stock-is-at-its-most-oversold-level-in-5-years.txt",  # gold standard
    "f13_ford-rolls-out-a-hot-rod-suv-as-drivers-abandon-performance-cars.txt",  # gold standard
    "f14_ford-is-the-next-ge-and-shorts-should-be-salivating.txt",  # gold standard
    "f15_ford-is-at-a-crossroad-of-danger-and-opportunity-in-china.txt",  # gold standard
    "jnj04_johnson-johnson-earnings-when-strong-is-nt-strong-enough.txt",  # gold standard
    "jnj05_where-s-the-tylenol-jj-disappoints-and-frustrates.txt",  # gold standard
    "jnj06_what-to-expect-from-johnson-johnson-in-2018.txt",  # gold standard
    "nem04_analyst-insight-is-newmont-mining-warming-up-for-a-good-2018.txt",  # gold standard
    "nem05_newmont-barrick-race-for-top-gold-crown-comes-down-to-a-decimal.txt",  # gold standard
    "nem06_newmont-mining-is-investors-gold-stock-to-buy.txt",  # gold standard
    "wmt05_walmart-stock-nears-key-support-after-earnings-miss.txt",  # gold standard
    "wmt06_goldman-expects-wal-mart-s-fortunes-to-improve-alongside-the-co.txt",  # gold standard
    "wmt07_walmart-s-meal-kits-are-not-the-solution-to-fight-amazon.txt",  # gold standard
    "bac04_bank-of-america-earnings-hurt-by-tax-related-charge.txt",
    "f03_may-auto-sales-results-reveal-a-new-leader-among-detroit-three.txt",
]


def match_string(string, documents):
    """
    Fuzzy matching of a string
    :param string:
    :param documents:
    :return:
    """

    def custom_full_process(token, **kwargs):

        try:
            s = token.text
        except Exception as e:
            s = str(token)
        return full_process(s, **kwargs)

    corpus_tokens = [tok for doc in documents for tok in doc.tokens]
    match_generator = fuzzprocess.extractWithoutOrder(
        string, corpus_tokens, processor=custom_full_process, score_cutoff=95
    )
    return match_generator


def retrieve_document_with_keyword(documents, keywords):

    docs = {}
    # for each token match see if event exists and if it exists check the type
    for event_type, keywords in keywords.items():
        for keyword in keywords:
            for match, score in match_string(keyword, documents):
                if not match.event_extent or event_type not in [
                    ev.event_type for ev in match.event_extent
                ]:
                    print(
                        f"KeywordMissed: {event_type} on {match} in {match.document_title} {match.annotator_id}"
                    )
                    docs.setdefault(
                        (match.document_title, match.annotator_id), {}
                    ).setdefault(event_type, []).append(match)
    docs_count = {
        doc_key: Counter({f"kw_{event_type}": len(matches)})
        for doc_key, keywords in docs.items()
        for event_type, matches in keywords.items()
    }

    return docs_count


def sum_counters(cs):
    sum_counter = Counter()
    for c in cs:
        sum_counter.update(c)
    sum_counter.update(Counter({"all": sum(sum_counter.values())}))
    return sum_counter


def check_typology(events, typology):
    """
    Checks an parsed event annotation for typology compatibility.

    :param events:
    :param typology:
    :return:
    """
    # check that type and subtype is n typology
    print(f"Checking annotated events for typology errors.")
    typology_maintypes = [ev["maintype_name"] for ev in typology]
    typology_subtypes = [ev["subtype_name"] for ev in typology]
    typology_subtypes.append(None)
    typology_participants = util.flatten(
        [ev["participant"] for ev in typology if ev["participant"]]
    )
    typology_fillers = util.flatten([ev["filler"] for ev in typology if ev["filler"]])
    # check that participants and fillers are allowed
    for event in events:
        event.typology_error = []
        allowed_part = list(
            filter(lambda p: p["on_type"] == event.event_type, typology_participants)
        )
        allowed_fill = list(
            filter(lambda p: p["on_type"] == event.event_type, typology_fillers)
        )
        # print(allowed_part)
        # print(allowed_fill)
        if event.event_subtype:
            subtype_name = event.event_subtype.split("_")[0]
            allowed_part = list(
                filter(lambda p: p["on_subtype"] == subtype_name, allowed_part)
            )
            allowed_fill = list(
                filter(lambda p: p["on_subtype"] == subtype_name, allowed_fill)
            )
            # print(allowed_part)
            # print(allowed_fill)
        allowed_part = [p["name"].split("_")[0] for p in allowed_part]
        allowed_fill = [f["name"].split("_")[0] for f in allowed_fill] + [
            "TIME",
            "CAPITAL",
            "PLACE",
        ]
        # print(allowed_part)
        # print(allowed_fill)
        if event.event_type not in typology_maintypes:
            # print(f"TypeError on {event}")
            event.typology_error.append((event.event_type, "TypeError", event))
        if event.event_subtype not in typology_subtypes:
            # print(f"SubtypeError on {event}")
            event.typology_error.append((event.event_subtype, "SubtypeError", event))
        if event.participants:
            for p in event.participants:
                role = p.role.split("_")[0]
                if role not in allowed_part:
                    # print(f"ParticipantError on {event} | {event.event_type} {event.event_subtype}: {role} | {allowed_part}")
                    event.typology_error.append((p.role, "ParticipantError", p))
        if event.fillers:
            for f in event.fillers:
                if f.role not in allowed_fill:
                    # print(f"FILLERError on {event} | {event.event_type} {event.event_subtype}: {role} | {allowed_part}")
                    event.typology_error.append((f.role, "FILLERError", f))


def clean_project(proj):
    """
    Clean the project from empty and redundant docs.
    :param proj:
    :return:
    """
    # check documents with no annotations
    unclean_len = len(proj.annotation_documents)
    proj.annotation_documents = [d for d in proj.annotation_documents if d.events]
    clean_len = len(proj.annotation_documents)
    print(f"Removed {unclean_len - clean_len} docs without event annotations.")

    # check double titles (was an issue with opening documents to annotators in WebAnno interfaces)
    keep_docs = []
    single_annotated_docs = (
        []
    )  # for returning when manually correcting: we avoid manually correcting these until final selection method is decided.
    for title, docgroup in groupby(proj.annotation_documents, key=lambda x: x.title):
        docs = list(docgroup)
        if len(docs) > 1:
            docs.sort(key=lambda x: len(x.events), reverse=True)
            cnt = {d.annotator_id: len(d.events) for d in docs}
            # most_events = max(docs, key=lambda x: len(x.events))
            # keep_docs.append(most_events)
            # print(f"{title} selected {most_events.annotator_id.upper()} {cnt}.")
            average = docs[
                corpus_stats_viz.median_index([float(len(d.events)) for d in docs])
            ]
            keep_docs.append(average)
            print(f"{title} selected {average.annotator_id.upper()} {cnt}.")
        else:
            keep_docs.append(docs[0])
            single_annotated_docs.append(docs[0])
    print(
        f"Removed {clean_len - len(keep_docs)} duplicate annotated docs by keeping docs with most events."
    )
    proj.annotation_documents = keep_docs

    return single_annotated_docs


def old_main():
    """
    Main function used in first inspection run.
    """

    project_pickle_fp = settings.ALL_ANNOTATIONS_PARSER_OPT
    typology_fp = settings.TYPOLOGY_FP

    exclude_gilles = lambda x: "anno" in Path(x.path).stem

    with open(project_pickle_fp, "rb") as project_in, open(
        typology_fp, "rt"
    ) as typology_in:
        event_project = pickle.load(project_in)
        typology = json.load(typology_in)

    single_annotated_docs = clean_project(event_project)
    # collect all events
    all_events = [ev for d in event_project.annotation_documents for ev in d.events]
    # count pct of events that have already been fully corrected
    event_corrected_cnt = sum(
        1 for event in all_events if event.document_title in fully_corrected
    )
    event_corrected_pct = round(100 * event_corrected_cnt / len(all_events), 2)
    print(
        f"{event_corrected_pct}% of events ({event_corrected_cnt}/{len(all_events)}) manually corrected in "
        f"{len(fully_corrected)} documents"
    )
    # collect and count typology violations
    check_typology(all_events, typology)
    event_typology_errors = sorted(
        [ev for ev in all_events if ev.typology_error],
        key=lambda x: (x.document_title, x.annotator_id),
    )
    for doc_title, evs in groupby(
        event_typology_errors, lambda x: (x.document_title, x.annotator_id)
    ):
        print("Typology Errors in", doc_title)
        for ev in evs:
            for err in ev.typology_error:
                print(
                    f"\t{err[1]} for {err[0]} on {ev.event_type}.{ev.event_subtype} in {str(ev.in_sentence[0])[:50]}"
                )

    # keyword check: check keywords for event types
    keywords = {
        "Dividend": ["yield"],
        "Profit/Loss": [
            "earnings",
            "profit",
            "loss",
            "income",
            "EPS",
            "earnings per share",
        ],
        "Revenue": ["revenue",],
        "Expense": ["cost", "expense"],
        "Product/Service": [
            "launch",
            "release",
            "trial",
            "foray",
            "product",
            "production",
            "pricing",
        ],
        "SecurityValue": [
            "undervalue",
            "overvalue",
            "underweight",
            "overweight",
            "underbought",
            "overbought",
            "oversold",
            "undersold",
        ],  # these are ambiguous terms that are a subjective measure of value
        "Rating": [
            "rating",
            "undervalue",
            "overvalue",
            "underweight",
            "overweight",
            "underbought",
            "overbought",
            "oversold",
            "undersold",
        ],
        "SalesVolume": ["sales", "sold"],
        "FinancialReport": [
            "guidance",
            "forecast",
            "projected",
            "E/P",
            "EP",
            "P/E",
            "PE",
        ],
        "generic_event": ["generate", "report"],
    }
    cnt_keyword = retrieve_document_with_keyword(
        event_project.annotation_documents, keywords
    )

    # issues on document level using document counts
    # docs_events_lt_10 = {(doc.title, doc.annotator_id): {"ev": len(doc.events), "sen": len(doc.sentences), "tok": len(doc.tokens)} for doc in iaa_project.annotation_documents if len(doc.events) < 10} # was used for determining a cutoff
    docs_events_per_senttok = {
        (doc.title, doc.annotator_id): {
            "sen/ev": len(doc.sentences) / len(doc.events),
            "tok/ev": len(doc.tokens) / len(doc.events),
            "ev": len(doc.events),
            "sen": len(doc.sentences),
            "tok": len(doc.tokens),
        }
        for doc in event_project.annotation_documents
    }

    # cutoff of 5 sentences per event is intuitively reasonable for inspection
    docs_too_little_events = {
        k: v for k, v in docs_events_per_senttok.items() if v["sen/ev"] > 5
    }

    # issues on event level using event attributes counts
    # collect macroeconomic issues
    macroecon_no_part = list(
        filter(
            lambda x: x.event_type == "Macroeconomics" and not x.participants,
            all_events,
        )
    )
    macroecon_no_affectedcompany = list(
        filter(
            lambda x: x.event_type == "Macroeconomics"
            and not corpus_stats_viz.check_role_in_participants(x, "AffectedCompany"),
            all_events,
        )
    )
    gkey = lambda x: (x.document_title, x.annotator_id)

    # other counters
    cnt_typology_error = {
        k: Counter("typology_error" for ev in g)
        for k, g in groupby(sorted(event_typology_errors, key=gkey), key=gkey)
    }
    cnt_no_part = {
        k: Counter("no_part" for ev in g)
        for k, g in groupby(
            sorted(list(filter(lambda x: not x.participants, all_events)), key=gkey),
            key=gkey,
        )
    }
    cnt_weak = {
        k: Counter("weak" for ev in g)
        for k, g in groupby(
            sorted(
                list(filter(lambda x: not x.participants or not x.fillers, all_events)),
                key=gkey,
            ),
            key=gkey,
        )
    }
    cnt_productservice_no_part = {
        k: Counter(ev.event_type + "_no_part" for ev in g)
        for k, g in groupby(
            sorted(
                list(
                    filter(
                        lambda x: not x.participants
                        and x.event_type == "Product/Service",
                        all_events,
                    )
                ),
                key=gkey,
            ),
            key=gkey,
        )
    }
    cnt_macroecon_no_part = {
        k: Counter(ev.event_type + "_no_part" for ev in g)
        for k, g in groupby(sorted(macroecon_no_part, key=gkey), key=gkey)
    }
    cnt_macroecon_no_affected = {
        k: Counter(ev.event_type + "_no_affcomp" for ev in g)
        for k, g in groupby(sorted(macroecon_no_affectedcompany, key=gkey), key=gkey)
    }
    cnt_problem_type = {
        k: Counter(ev.event_type for ev in g)
        for k, g in groupby(
            sorted(
                list(
                    filter(
                        lambda x: x.event_type
                        in [
                            "Macroeconomics",
                            "FinancialReport",
                            "Revenue",
                            "CSR/Brand",
                            "Profit/Loss",
                            "SalesVolume",
                        ],
                        all_events,
                    )
                ),
                key=gkey,
            ),
            key=gkey,
        )
    }
    cnt_all = util.dict_zip(
        cnt_typology_error,
        cnt_keyword,
        cnt_productservice_no_part,
        cnt_no_part,
        cnt_macroecon_no_affected,
        cnt_macroecon_no_part,
        cnt_problem_type,
        cnt_weak,
        fillvalue=None,
    )
    cnt_all = {k: sum_counters(v) for k, v in cnt_all.items()}
    single_annotated_titles = [d.title for d in single_annotated_docs]

    print(
        "+------------------------+\n|  DOCUMENTS TO INSPECT  |\n+------------------------+"
    )
    i = 0
    for k, cnt in sorted(
        cnt_all.items(), key=lambda x: sum(x[1].values()), reverse=True
    ):
        (title, anno) = k
        if (
            title in single_annotated_titles and title not in fully_corrected
        ):  # because we have not yet completed selection of multiple annotated documents
            if k in docs_too_little_events:
                print(
                    i,
                    title,
                    anno,
                    f"!!!!!!!!!!! <= CHECK IF FULLY ANNOTATED {docs_too_little_events[k]}",
                )
            else:
                print(i, title, anno)
            print(cnt)
            i += 1

    #
    print(
        "+------------------------+\n|  KEYWORDS DOCUMENTS TO INSPECT  |\n+------------------------+"
    )
    i = 0
    for k, cnt in sorted(
        cnt_keyword.items(), key=lambda x: sum(x[1].values()), reverse=True
    ):
        (title, anno) = k
        if (
            title in single_annotated_titles
        ):  # because we have not yet completed selection of multiple annotated documents
            if k in docs_too_little_events:
                print(
                    i,
                    title,
                    anno,
                    f"!!!!!!!!!!! <= CHECK IF FULLY ANNOTATED {docs_too_little_events[k]}",
                )
            else:
                print(i, title, anno)
            print(cnt)
            i += 1

    # check docs with more than 50 sentences due to Webanno settings that shows only 50 sentences
    print(
        "+------------------------+\n|  DOCUMENTS +50 SEN  |\n+------------------------+"
    )
    for doc in event_project.annotation_documents:
        if len(doc.sentences) > 50:
            if (
                doc.title in single_annotated_titles
                and doc.title not in fully_corrected
            ):
                print(doc.title, doc.annotator_id)


def inspect_annotations(project, typology, events=None):

    # collect all events
    if events:
        all_events = events
    else:
        all_events = [
            ev for d in project.annotation_documents if d.events for ev in d.events
        ]

    # collect and count typology violations
    check_typology(all_events, typology)
    event_typology_errors = sorted(
        [ev for ev in all_events if ev.typology_error],
        key=lambda x: (x.document_title, x.annotator_id),
    )
    for doc_title, evs in groupby(
        event_typology_errors, lambda x: (x.document_title, x.annotator_id)
    ):
        print("Typology Errors in", doc_title)
        for ev in evs:
            for err in ev.typology_error:
                print(
                    f"\t{err[1]} for {err[0]} on {ev.event_type}.{ev.event_subtype} in {str(ev.in_sentence[0])[:50]}"
                )

    # keyword check: check keywords for event types
    keywords = {
        "Dividend": ["yield"],
        "Profit/Loss": [
            "earnings",
            "profit",
            "loss",
            "income",
            "EPS",
            "earnings per share",
        ],
        "Revenue": ["revenue",],
        "Expense": ["cost", "expense"],
        "Product/Service": [
            "launch",
            "release",
            "trial",
            "foray",
            "product",
            "production",
            "pricing",
        ],
        "SecurityValue": [
            "undervalue",
            "overvalue",
            "underweight",
            "overweight",
            "underbought",
            "overbought",
            "oversold",
            "undersold",
        ],  # these are ambiguous terms that are a subjective measure of value
        "Rating": [
            "rating",
            "undervalue",
            "overvalue",
            "underweight",
            "overweight",
            "underbought",
            "overbought",
            "oversold",
            "undersold",
        ],
        "SalesVolume": ["sales", "sold"],
        "FinancialReport": [
            "guidance",
            "forecast",
            "projected",
            "E/P",
            "EP",
            "P/E",
            "PE",
        ],
        "generic_event": ["generate", "report"],
    }
    cnt_keyword = retrieve_document_with_keyword(project.annotation_documents, keywords)

    # issues on document level using document counts
    # docs_events_lt_10 = {(doc.title, doc.annotator_id): {"ev": len(doc.events), "sen": len(doc.sentences), "tok": len(doc.tokens)} for doc in iaa_project.annotation_documents if len(doc.events) < 10} # was used for determining a cutoff
    docs_events_per_senttok = {}
    for doc in project.annotation_documents:
        docs_events_per_senttok[(doc.title, doc.annotator_id)] = {}
        n_sen = len(doc.sentences) if doc.sentences else 0.0
        n_ev = len(doc.events) if doc.events else 0.0
        n_tok = len(doc.tokens) if doc.tokens else 0.0
        sen_ev = n_sen / n_ev if n_ev else 0.0
        tok_ev = n_tok / n_ev if n_ev else 0.0

        docs_events_per_senttok[(doc.title, doc.annotator_id)]["sen"] = n_sen
        docs_events_per_senttok[(doc.title, doc.annotator_id)]["ev"] = n_ev
        docs_events_per_senttok[(doc.title, doc.annotator_id)]["tok"] = n_tok
        docs_events_per_senttok[(doc.title, doc.annotator_id)]["sen/ev"] = sen_ev
        docs_events_per_senttok[(doc.title, doc.annotator_id)]["tok/ev"] = tok_ev

    # cutoff of 5 sentences per event is intuitively reasonable for inspection
    docs_too_little_events = {
        k: v for k, v in docs_events_per_senttok.items() if v["sen/ev"] > 5
    }

    # issues on event level using event attributes counts
    # collect macroeconomic issues
    csrbrand_no_part = list(
        filter(
            lambda x: x.event_type == "CSR/Brand" and not x.participants, all_events,
        )
    )

    macroecon_no_part = list(
        filter(
            lambda x: x.event_type == "Macroeconomics" and not x.participants,
            all_events,
        )
    )
    macroecon_no_affectedcompany = list(
        filter(
            lambda x: x.event_type == "Macroeconomics"
            and not corpus_stats_viz.check_role_in_participants(x, "AffectedCompany"),
            all_events,
        )
    )
    gkey = lambda x: (x.document_title, x.annotator_id)

    # other counters
    cnt_typology_error = {
        k: Counter("typology_error" for ev in g)
        for k, g in groupby(sorted(event_typology_errors, key=gkey), key=gkey)
    }
    cnt_no_part = {
        k: Counter("no_part" for ev in g)
        for k, g in groupby(
            sorted(list(filter(lambda x: not x.participants, all_events)), key=gkey),
            key=gkey,
        )
    }
    cnt_productservice_no_part = {
        k: Counter(ev.event_type + "_no_part" for ev in g)
        for k, g in groupby(
            sorted(
                list(
                    filter(
                        lambda x: not x.participants
                        and x.event_type == "Product/Service",
                        all_events,
                    )
                ),
                key=gkey,
            ),
            key=gkey,
        )
    }
    cnt_macroecon_no_part = {
        k: Counter(ev.event_type + "_no_part" for ev in g)
        for k, g in groupby(sorted(macroecon_no_part, key=gkey), key=gkey)
    }

    cnt_csrbrand_no_part = {
        k: Counter(ev.event_type + "_no_part" for ev in g)
        for k, g in groupby(sorted(csrbrand_no_part, key=gkey), key=gkey)
    }

    cnt_macroecon_no_affected = {
        k: Counter(ev.event_type + "_no_affcomp" for ev in g)
        for k, g in groupby(sorted(macroecon_no_affectedcompany, key=gkey), key=gkey)
    }
    cnt_problem_type = {
        k: Counter(ev.event_type for ev in g)
        for k, g in groupby(
            sorted(
                list(
                    filter(
                        lambda x: x.event_type
                        in [
                            "Macroeconomics",
                            "FinancialReport",
                            "Revenue",
                            "CSR/Brand",
                            "Profit/Loss",
                            "SalesVolume",
                        ],
                        all_events,
                    )
                ),
                key=gkey,
            ),
            key=gkey,
        )
    }
    cnt_all = util.dict_zip(
        cnt_typology_error,
        cnt_keyword,
        cnt_productservice_no_part,
        cnt_no_part,
        cnt_macroecon_no_affected,
        cnt_macroecon_no_part,
        cnt_problem_type,
        cnt_csrbrand_no_part,
        fillvalue=None,
    )
    cnt_all = {k: sum_counters(v) for k, v in cnt_all.items()}
    # single_annotated_titles = [d.title for d in single_annotated_docs]

    print(
        "+------------------------+\n|  DOCUMENTS TO INSPECT  |\n+------------------------+"
    )
    i = 0
    for k, cnt in sorted(
        cnt_all.items(), key=lambda x: sum(x[1].values()), reverse=True
    ):
        (title, anno) = k
        print(title, anno, cnt)
        # if title in single_annotated_titles and title not in fully_corrected:  # because we have not yet completed selection of multiple annotated documents
        #     if k in docs_too_little_events:
        #         print(i, title, anno, f"!!!!!!!!!!! <= CHECK IF FULLY ANNOTATED {docs_too_little_events[k]}")
        #     else:
        #         print(i, title, anno)
        #     print(cnt)
        #     i += 1

    #
    print(
        "+------------------------+\n|  KEYWORDS DOCUMENTS TO INSPECT  |\n+------------------------+"
    )
    i = 0
    for k, cnt in sorted(
        cnt_keyword.items(), key=lambda x: sum(x[1].values()), reverse=True
    ):
        (title, anno) = k
        print(title, anno, cnt)
        # if title in single_annotated_titles:  # because we have not yet completed selection of multiple annotated documents
        #     if k in docs_too_little_events:
        #         print(i, title, anno, f"!!!!!!!!!!! <= CHECK IF FULLY ANNOTATED {docs_too_little_events[k]}")
        #     else:
        #         print(i, title, anno)
        #     print(cnt)
        #     i += 1

    # check docs with more than 50 sentences due to Webanno settings that shows only 50 sentences
    print(
        "+------------------------+\n|  DOCUMENTS +50 SEN  |\n+------------------------+"
    )
    for doc in project.annotation_documents:
        if len(doc.sentences) > 50:
            # if doc.title in single_annotated_titles and doc.title not in fully_corrected:
            print(doc.title, doc.annotator_id)


if __name__ == "__main__":

    # load typology for typology checking
    with open(settings.TYPOLOGY_FP, "rt") as typology_in:
        typology = json.load(typology_in)

    # Inspect SENTiVENT-english-event-1.0-clean
    clean_project = parse_project(settings.CLEAN_XMI_DIRP)
    # remove empty docs
    clean_project.annotation_documents = [
        d
        for d in clean_project.annotation_documents
        if d.annotator_id == settings.MOD_ID
    ]

    inspect_annotations(clean_project, typology)

    # # INSPECT IAA STUDY
    # iaa_project = util.unpickle_webanno_project(settings.IAA_PARSER_OPT)
    #
    # # filter every file not from moderator_id
    # iaa_project.annotation_documents = [d for d in iaa_project.annotation_documents if d.annotator_id == settings.MOD_ID]
    #
    # inspect_annotations(iaa_project, typology)
    #
    # # INSPECT MAIN CORPUS
    # main_project = util.unpickle_webanno_project(settings.MAIN_PARSER_OPT)
    #
    # # filter every file not from moderator_id
    # main_project.annotation_documents = [d for d in main_project.annotation_documents
    #                                      if d.annotator_id != settings.MOD_ID
    #                                      and not d.title[0:1].isdigit()]
    #
    # inspect_annotations(main_project, typology)
