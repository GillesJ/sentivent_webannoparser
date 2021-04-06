#!/usr/bin/env python3
"""
Settings file with run settings and shared variables.

settings.py
sentivent_webannoparser
10/4/18
Copyright (c) Gilles Jacobs. All rights reserved.
"""
from pathlib import Path

MASTER_DIRP = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-final_project_2021-02-10_1537/"
MASTER_DIRP_BEFORE_FEBR21 = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-final_project_2020-09-16_1116"
MASTER_DIRP_BEFORE_SEPT = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-final_project_2020-07-03_1739/"

# SENTIMENT_IAA = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-iaa_project_2020-05-19_1028"
SENTIMENT_IAA = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-iaa_project_2020-05-26_1450"
SENTIMENT_ANNO = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-anno_project_2020-05-27_0931"

IAA_XMI_DIRP = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-event-en/XMI-SENTiVENT-event-english-1.0-iaastudy_2019-12-03_1545"  # IAA adjudicated gold-standard
MAIN_XMI_DIRP = "/home/gilles/sentivent-phd/sentivent-resources-dataset/webanno-project-export/sentivent-english-event-latest/XMI-SENTiVENT-event-english-1.0-main-corpus_2019-03-12_1759"  # Main corpus DIRP"
PREFINAL_XMI_DIRP = "/home/gilles/sentivent-phd/sentivent-resources-dataset/webanno-project-export/sentivent-english-event-latest/XMI-SENTiVENT-event-english-final-1_2019-12-02_1529"  # PREFinal corpus
CLEAN_XMI_DIRP = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-event-en/XMI-SENTiVENT-event-english-1.0-clean_2020-02-11_1354"  # clean project
TEST_XMI_DIRP = "/home/gilles/sentivent-phd/sentivent-resources-dataset/webanno-project-export/testing/TEST_SENTiVENT-event-english-1_2020-02-11_1150"

TYPOLOGY_FP = "/home/gilles/sentivent-phd/annotation/event-annotation-preparation-implementation-webanno-english/webanno-event-implementation/scripts/sentivent_en_event_typology.json"
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

SPLITS_DOC_EXPERIMENTS = {"test": ["aapl14", "aapl15", "aapl16", "amzn12", "amzn13", "amzn14", "ba14", "ba15", "ba16", "bac04", "bac05", "bac06", "cvx04", "cvx05", "cvx06", "duk05", "duk06", "duk07", "f13", "f14", "f15", "jnj04", "jnj05", "jnj06", "nem04", "nem05", "nem06", "wmt05", "wmt06", "wmt07"], "dev": ["aal10", "aapl01", "abbv00", "amd11", "amzn04", "ba03", "bac01", "celg01", "chk00", "cmg04", "cost00", "cvx07", "dis06", "duk00", "f00", "fb01", "fox01", "ge03", "gm09", "goog09", "gs00", "jnj01", "kmi04", "msft04", "nem01", "nflx13", "pg04", "t04", "wfc00", "wmt01"], "train": ["aal00", "aal01", "aal02", "aal03", "aal04", "aal05", "aal06", "aal07", "aal08", "aal09", "aal11", "aal12", "aal13", "aal14", "aapl00", "aapl02", "aapl03", "aapl04", "aapl05", "aapl06", "aapl07", "aapl08", "aapl09", "aapl10", "aapl11", "aapl12", "aapl13", "abbv01", "abbv02", "abbv03", "abbv04", "abbv05", "abbv06", "abbv07", "abbv08", "abbv09", "abbv10", "abbv11", "abbv12", "abbv13", "amd00", "amd01", "amd02", "amd03", "amd04", "amd05", "amd06", "amd07", "amd08", "amd09", "amd10", "amd12", "amd13", "amd14", "amzn00", "amzn01", "amzn02", "amzn03", "amzn05", "amzn06", "amzn07", "amzn08", "amzn09", "amzn10", "amzn11", "ba00", "ba01", "ba02", "ba04", "ba05", "ba06", "ba07", "ba08", "ba09", "ba10", "ba11", "ba12", "ba13", "bac00", "bac02", "bac03", "celg00", "celg02", "celg03", "celg04", "chk01", "chk02", "chk03", "cmg00", "cmg01", "cmg02", "cmg03", "cost01", "cost02", "cost03", "cvx00", "cvx01", "cvx02", "cvx03", "dis00", "dis01", "dis02", "dis03", "dis04", "dis05", "dis07", "dis08", "dis09", "dis10", "dis11", "dis12", "duk01", "duk02", "duk03", "duk04", "f01", "f02", "f03", "f04", "f05", "f06", "f07", "f08", "f09", "f10", "f11", "f12", "fb00", "fb02", "fb03", "fb04", "fb05", "fb06", "fb07", "fb08", "fb09", "fb10", "fb11", "fb12", "fb13", "fb14", "fox00", "fox02", "fox03", "fox04", "ge00", "ge01", "ge02", "ge04", "ge05", "ge06", "ge07", "ge08", "ge09", "ge10", "ge11", "ge12", "ge13", "gm00", "gm01", "gm02", "gm03", "gm04", "gm05", "gm06", "gm07", "gm10", "gm11", "gm12", "goog00", "goog01", "goog02", "goog03", "goog04", "goog05", "goog06", "goog07", "goog08", "goog10", "goog11", "goog12", "goog13", "gs01", "gs02", "gs03", "gs04", "jnj00", "jnj02", "jnj03", "kmi00", "kmi01", "kmi02", "kmi03", "msft01", "msft02", "msft03", "nem00", "nem02", "nem03", "nflx00", "nflx01", "nflx02", "nflx03", "nflx04", "nflx11", "nflx12", "nflx14", "pg00", "pg01", "pg02", "pg03", "pg05", "t00", "t01", "t02", "t03", "t05", "t06", "t12", "t13", "t14", "wfc01", "wfc02", "wfc03", "wmt00", "wmt02", "wmt03", "wmt04"]}

PARSER_MULTIPROC = True  # set True to use faster multiprocessing for parsing files, False to parse iteratively
