#!/usr/bin/env python3
"""
Settings file with run settings and shared variables.

settings.py
sentivent-webannoparser
10/4/18
Copyright (c) Gilles Jacobs. All rights reserved.
"""
from pathlib import Path

IAA_XMI_DIRP= "/home/gilles/00-sentivent-fwosb-phd-2017-2020/sentivent-resources-dataset/webanno-project-export/sentivent-english-event-latest/XMI-SENTiVENT-event-english-1.0-iaastudy_2019-04-02_0951" # Main corpus DIRP
MAIN_XMI_DIRP = "/home/gilles/00-sentivent-fwosb-phd-2017-2020/sentivent-resources-dataset/webanno-project-export/sentivent-english-event-latest/XMI-SENTiVENT-event-english-1.0-main-corpus_2019-03-12_1759" # IAA gold standard test set DIRP with adjudicated data
TYPOLOGY_FP = "/home/gilles/00-sentivent-fwosb-phd-2017-2020/00-sentivent-event-annotation-preperation-implementation/webanno-event-implementation/scripts/sentivent_en_event_typology.json"

MOD_ID = "gilles"

OPT_DIRP = "./output/"
Path(OPT_DIRP).mkdir(parents=True, exist_ok=True)

MAIN_PARSER_OPT = str(Path(OPT_DIRP) / "en_event_all_annotations.pickle")
IAA_PARSER_OPT = str(Path(OPT_DIRP) / "iaa_annotations.pickle")

PLOT_DIRP = str(Path(OPT_DIRP) / "plots")
Path(PLOT_DIRP).mkdir(parents=True, exist_ok=True)
