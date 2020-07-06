#!/usr/bin/env python3
"""
samples_to_latex.py
sentivent_webannoparser 
2/10/20
Copyright (c) Gilles Jacobs. All rights reserved.  
"""
from parser import *
from parse_project import parse_project
import settings


def to_latex(event, tag_type=True):
    """
    Format the sentence as LaTeX representation in which
    event trigger = bold
    full underline = participant
    dashed underline = filler
    :return:
    """

    return s


def filter_example(ev):
    if (
        ev.event_subtype
        and ev.participants
        and len(ev.in_sentence[0].tokens) < 20
        and ev.polarity_negation == "negative"
    ):
        return True


if __name__ == "__main__":
    proj = parse_project(settings.CLEAN_XMI_DIRP)

    example = list(filter(filter_example, proj.get_events()))

    docs = proj.annotation_documents
    for doc in docs:
        for ev in doc.events:
            if "SecurityValue" in ev.event_type and "Increase" in str(ev.event_subtype):
                part_roles = (
                    [p.role for p in ev.participants] if ev.participants else []
                )
                # print(part_roles)
                if "Price" in part_roles:
                    trig = ev.text
                    trig_fmt = f"\\anntrg{{{trig}}}"
                    for sen in ev.in_sentence:
                        if len(str(sen)) < 80:
                            s = str(sen).replace(trig, trig_fmt)
                            id = f"{doc.document_id} s{sen.element_id:02d}"
                            print(s + f"~\\footnotesize{{[{id}]}}")
                            pass
