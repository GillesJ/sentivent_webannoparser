#!/usr/bin/env python3
"""
Settings file with run settings and shared variables.

settings.py
sentivent_webannoparser
10/4/18
Copyright (c) Gilles Jacobs. All rights reserved.
"""
from pathlib import Path

IAA_XMI_DIRP= "/home/gilles/sentivent-phd/sentivent-resources-dataset/webanno-project-export/sentivent-english-event-latest/XMI-SENTiVENT-event-english-1.0-iaastudy_2019-12-03_1545" # IAA adjudicated gold-standard
MAIN_XMI_DIRP = "/home/gilles/sentivent-phd/sentivent-resources-dataset/webanno-project-export/sentivent-english-event-latest/XMI-SENTiVENT-event-english-1.0-main-corpus_2019-03-12_1759" # Main corpus DIRP"
FINAL_XMI_DIRP = "/home/gilles/sentivent-phd/sentivent-resources-dataset/webanno-project-export/sentivent-english-event-latest/XMI-SENTiVENT-event-english-final-1_2019-12-02_1529" # Final corpus
TYPOLOGY_FP = "/home/gilles/sentivent-phd/webanno-english-event-annotation-preparation-implementation/webanno-event-implementation/scripts/sentivent_en_event_typology.json"

MOD_ID = "gilles"  # annotator id of the judge who adjudicated for gold standard

OPT_DIRP = "./output/"
Path(OPT_DIRP).mkdir(parents=True, exist_ok=True)

MAIN_PARSER_OPT = str(Path(OPT_DIRP) / "en_event_all_annotations.pickle")
IAA_PARSER_OPT = str(Path(OPT_DIRP) / "iaa_annotations.pickle")

PLOT_DIRP = str(Path(OPT_DIRP) / "plots")
Path(PLOT_DIRP).mkdir(parents=True, exist_ok=True)

IA_IDS = [
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

PARSER_MULTIPROC = True # set True to use faster multiprocessing for parsing files, False to parse iteratively