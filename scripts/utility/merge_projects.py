#!/usr/bin/env python3
"""
Utility script to merge gold-standard adjudicated IAA files in XMI export format with correct titles.

merge_projects.py in sentivent_webannoparser
7/3/20 Copyright (c) Gilles Jacobs
"""
from pathlib import Path
from zipfile import ZipFile

iaa_dir = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-iaa_project_2020-07-03_1221"
anno_dir = Path(iaa_dir) / "annotation"
anno_id_keep = "gilles"
opt_dir = Path(iaa_dir) / "to_upload"
opt_dir.mkdir(exist_ok=True)

for fp in anno_dir.glob("**/*.zip"):
    with ZipFile(fp) as zip:
        fns = zip.namelist()
        id_in_zip = [fn for fn in fns if anno_id_keep in fn]
        if id_in_zip:
            with zip.open(id_in_zip[0]) as f_in:
                content = f_in.read()
                opt_fn = Path(fp).parts[-2]
                with open(opt_dir / opt_fn, "wb") as f_out:
                    f_out.write(content)
