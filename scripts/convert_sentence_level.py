#!/usr/bin/env python3
'''
convert_sentence_level.py
sentivent_webannoparser 
11/13/19
Copyright (c) Gilles Jacobs. All rights reserved.  
'''

from parser import WebannoProject
import pandas
import util
import settings
from pathlib import Path

def parse_and_pickle(project_dirp, opt_fp):

    project = WebannoProject(project_dirp)
    project.parse_annotation_project()

    records = []
    for doc in project.annotation_documents:
        for sen in sentences:
            
            record = {
                "doc_id": doc.,
                "ann_id": ,
                "sentence_text": ,
                "sentence_idx": ,
                "labels_true_event_maintype": ,
                "labels_true_event_subtypes": ,
                "y_true_maintype_multihot": ,
                "y_true_subtype_multihot": ,
            }

    pass

if __name__ == "__main__":

    parse_and_pickle(settings.MAIN_XMI_DIRP, str(Path(settings.OPT_DIRP) / "sentence-level.csv"))