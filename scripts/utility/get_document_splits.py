#!/usr/bin/env python3
'''
Get the standard train-dev-test split document ids from the DYGIEPP experiments for use in other experiments.

get_document_splits.py in sentivent_webannoparser
2/2/21 Copyright (c) Gilles Jacobs
'''
from pathlib import Path
import json


data_dir = "/home/gilles/repos/dygiepp/data/sentivent/preproc_full_trunc-5-7/"
data_dp = Path(data_dir)

splits = {}

for fp in data_dp.rglob("*.jsonl"):
    split_name = fp.stem
    with open(fp, "rt") as json_in:
        doc_ids = [json.loads(l)["doc_key"] for l in json_in.readlines()]
        splits[split_name] = doc_ids

split_out_fp = "sentivent_train_dev_test_split_document_ids.json"
with open(split_out_fp, "wt") as split_out:
    json.dump(splits, split_out)