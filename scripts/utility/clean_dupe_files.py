#!/usr/bin/env python3
'''
Clean up residual doubly annotated files in the final corpus export.
These files are due to artifacts of the annotation process.

clean_dupe_files.py in sentivent_webannoparser
7/3/20 Copyright (c) Gilles Jacobs
'''
from pathlib import Path
from itertools import groupby
from parse_project import parse_project
import json


prefinal_dir = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-final_project_2020-07-03_1527"
anno_dir = Path(prefinal_dir) / 'annotation'
ser_dir = Path(prefinal_dir) / "annotation_ser"
metadata_fp = next(Path(prefinal_dir).glob("exported*.json"))
with open(metadata_fp, "rt") as meta_in:
    metadata = json.load(meta_in)

proj = parse_project(prefinal_dir)

key_f = lambda x: x.title
delete_from_metadata = []

for title, g in groupby(sorted(proj.annotation_documents, key=key_f), key_f):
    g = list(g)
    if len(g) > 1:
        print(f"--{title}--")
        for doc in g:
            print(doc.annotator_id, len(doc.sentiment_expressions))
            if doc.annotator_id == "gilles" and len(doc.sentiment_expressions) == 0:
                fn = Path(doc.path).parts[-3]
                anno_del_fp = Path(doc.path).parent # remove gilles annotation file
                ser_del_fp = ser_dir / fn / "gilles.ser" # remove gilles .ser file
                anno_del_fp.unlink()
                ser_del_fp.unlink()
                delete_from_metadata.append((fn, doc.annotator_id))
                print(f"Removed {doc.annotator_id}")

# delete the entry from the metadata
for (name, user) in delete_from_metadata:
    metadata["annotation_documents"] = [x for x in metadata["annotation_documents"] if not (x["name"] == name and x["user"] == user)]

with open(metadata_fp, "wt") as meta_out:
    json.dump(metadata, meta_out, indent=2)
