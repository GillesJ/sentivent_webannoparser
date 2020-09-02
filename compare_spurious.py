#!/usr/bin/env python3
'''
Script to compare spurious trigger errors due to single-token data transformation.



compare_spurious.py in sentivent_webannoparser
8/28/20 Copyright (c) Gilles Jacobs
'''

import pandas as pd
from parse_project import parse_project

errors_fp = "/home/gilles/repos/dygiepp/scripts/analysis/errors_trigger.csv"
project_fp = "/home/gilles/sentivent-phd/resources-dataset-guidelines/sentivent-webanno-project-export/sentivent-sentiment-en/XMI-SENTiVENT-sentiment-en-final_project_2020-07-03_1739/"

df_err = pd.read_csv(errors_fp)
project = parse_project(project_fp, from_scratch=False)

ref_events = {}
for doc in project.test:
    for sen in doc.sentences:
        id = f"{doc.document_id}_{sen.element_id}"
        if id in df_err["id"].values:
            ref_events[id] = [(ev.get_extent_token_ids(), ev.event_type) for ev in sen.events]
    # lookup doc_id (docid_senix)

spur_err = 0
single_tok_errors = 0
single_tok_error_type = []
df_err_analy = pd.DataFrame(columns=list(df_err.columns) + ["1_tok_error", "1_tok_error_missclf"])
for i, row in df_err.iterrows():
    if row["error_type"] == "spurious":
        spur_err += 1
        pred_ix = int(row["pred_ix"])
        pred_type = row["pred_label"]
        ref = ref_events[row["id"]]
        for (ref_ic, ref_type) in ref:
            if ref_ic[0] <= pred_ix <= ref_ic[-1]:
                single_tok_errors += 1
                row["1_tok_error"] = True
                if pred_type not in ref_type:
                    single_tok_error_type.append((ref_ic, ref_type, pred_ix, pred_type))
                    row["1_tok_error_missclf"] = True
    df_err_analy.loc[i] = row

df_err_analy.to_csv(f"{errors_fp.split('.csv')[0]}_anno.csv", index=False)
stok_pct =  round(100 * single_tok_errors / spur_err, 2)
misclf_pct = round(100 * len(single_tok_error_type) / single_tok_errors, 2)
print(f"Spurious Errors due to single-token preprocessing: {single_tok_errors}/{len(df_err)} {stok_pct}%")
print(f"Of which {len(single_tok_error_type)}/{single_tok_errors} {misclf_pct}% misclass.")