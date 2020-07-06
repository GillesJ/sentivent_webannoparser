#!/usr/bin/env python3
"""
test_parser.py
sentivent_webannoparser 
2/11/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
from parse_project import parse_project
import settings
from collections import Counter
from itertools import groupby
from corpus_stats_viz import is_weakly_realized

if __name__ == "__main__":
    proj = parse_project(settings.TEST_XMI_DIRP, from_scratch=True)
    all_docs = proj.annotation_documents

    all_events = []
    for d in all_docs:
        for ev in d.events:
            all_events.append(ev)

    # count weakly realized events
    weak_events = [ev for ev in all_events if is_weakly_realized(ev)]
    weak_docs = []
    for title, evs in groupby(
        sorted(weak_events, key=lambda x: x.document_title),
        key=lambda x: x.document_title,
    ):
        evs = list(evs)
        weak_docs.append((title, len(evs), evs))
    weak_docs.sort(key=lambda x: x[1], reverse=True)
    weak_cnt = len(weak_events)
    weak_types = Counter(ev.event_type for ev in weak_events)
    pass
