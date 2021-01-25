#!/usr/bin/env python3
"""
Simple parsing test https://github.com/dkpro/dkpro-cassis repo on Sentivent WebAnno project XMI CAS export.

webannoparser 
1/14/19
"""
from cassis import *
from pathlib import Path
import settings
import zipfile

xmi_export_dirp = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-iaa_project_2020-05-07_0858"

for zip_fp in Path(xmi_export_dirp).glob("annotation/*/*.zip"):
    doc_n = zip_fp.parts[-2]
    with zipfile.ZipFile(zip_fp, "r") as zip_in:
        xmi_fn = next(f for f in zip_in.filelist if f.filename.endswith("xmi"))
        ann_id = xmi_fn.filename.strip(".xmi")
        xmi_str = zip_in.read(xmi_fn).decode("utf-8")
        typesystem = load_typesystem(zip_in.read("TypeSystem.xml").decode("utf-8"))
        cas = load_cas_from_xmi(xmi_str, typesystem=typesystem)

        custom_types = sorted([t for t in typesystem._types if "custom" in t])

        events = list(cas.select("webanno.custom.A_Event"))
        for event in events:

            # get sentence
            for sentence in cas.select_covering(
                "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence", event
            ):
                print(sentence)
            # get tokens
            for token in cas.select_covered(
                "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token", event
            ):
                print(token)

            if not event.i_Polarity:
                print(event, " missing polarity_negation annotation")
            # print(unit.get_covered_text())
            # for d in unit.f_Discontiguous:
            #     print("Discont.: ", d.target.get_covered_text())
        pass
pass
