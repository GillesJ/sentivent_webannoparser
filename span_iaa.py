#!/usr/bin/env python3
"""
span_iaa.py
webannoparser
02/06/20
Copyright (c) Gilles Jacobs. All rights reserved.

Match and align annotation spans in a pair-wise manner for agreement + evaluation in the SENTiVENT project.
For all span annotations including events and sentiment expressions.
Outputs files for use with Agreestat360.com
"""
import types
from nltk.metrics.agreement import AnnotationTask
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
from parser import *
from parse_project import parse_project, parse_corpus
from functools import partial
import re
import util
from pathlib import Path
from itertools import groupby, product, combinations, chain
from collections import Counter
import settings
import pandas as pd
import numpy as np
import numpy.testing as npt

def make_csv(data, opt_dirp="agreestat-sentiment-span-iaa-files"):

    Path(opt_dirp).mkdir(parents=True, exist_ok=True)

    d = {}

    for anno_id, itm, label in data:
        d.setdefault(itm, []).append({anno_id: label})

    df = pd.Series(d).apply(lambda x: pd.Series({ k: v for y in x for k, v in y.items() }))
    df.to_csv(Path(opt_dirp) / "agreestat_interrater_data.csv", index=False)

class CustomAnnotationTask(AnnotationTask):
    """
    Wrapper object aorund nltk.agreement.AnnotationTask object that allows for frp metrics to be computed.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.label_encoder = LabelEncoder().fit(np.array(list(self.K)))
        self.sk_labels = self._get_scikit_labels()

        self.metrics = {
            "Fleiss' kappa": self.multi_kappa,
            "Cohen's kappa": self.kappa,
            "Krippendorff's alpha": self.alpha,
            "Weighted kappa": self.weighted_kappa,
            "S-score": self.S,
            "Scott's pi (multi)": self.pi,
            "F1-score": f1_score,
            "Precision": precision_score,
            "Recall": recall_score,
            # "Accuracy": accuracy_score
        }

        # set distance func
        if isinstance(
                self.distance, tuple
        ):  # if string it should be a function of this obj
            func_name, dist_kwargs = self.distance[0], self.distance[1]
            self.distance = partial(getattr(self, func_name), **dist_kwargs)
        elif callable(self.distance):  # else it should be a passed function
            pass
        else:
            raise ValueError(
                f'{self.distance} should be a tuple or a dict ("func name of class method", kwargs).'
            )

    def load_array(self, array):
        """Load an sequence of annotation results, appending to any data already loaded.

        The argument is a sequence of 3-tuples, each representing a coder's labeling of an item:
            (coder,item,label)
        """
        for coder, item, labels in array:
            if isinstance(self.distance, tuple) and "windowed" in self.distance[0]:
                labels = labels[0]
            self.C.add(coder)
            self.K.add(labels)
            self.I.add(item)
            self.data.append({"coder": coder, "labels": labels, "item": item})

    def compute_all(self, average="binary"):
        all_results = {}
        for name, func in self.metrics.items():
            if isinstance(func, types.MethodType):
                all_results[name] = func()
            else:
                all_results[name] = self.scikit_metric_pairwise(func, average=average)

        return all_results

    def _get_scikit_labels(self):
        sk_labels = []
        key = lambda x: x["coder"]
        data = self.data[:]
        data.sort(key=key)
        for item, item_data in groupby(data, key=key):
            labels_ann = self.label_encoder.transform(
                [
                    idat["labels"][0]
                    if isinstance(idat["labels"], tuple)
                    else idat["labels"]
                    for idat in item_data
                ]
            )
            sk_labels.append(labels_ann)
        return sk_labels

    def scikit_metric_pairwise(self, func, **kwargs):
        total = []
        s = self.sk_labels[:]
        for lab1 in self.sk_labels:
            s.remove(lab1)
            for lab2 in s:
                total.append(func(lab1, lab2, **kwargs))
        ret = np.mean(total, axis=0)
        return ret

def dice_coefficient(a, b):
    """
    :param a: list of items
    :param b: list of items
    :return: dice coefficient score
    """

    a.sort()
    b.sort()
    if not len(a) or not len(b):
        return 0.0
    """ quick case for true duplicates """
    if a == b:
        return 1.0

    # assignments to save function calls
    lena = len(a)
    lenb = len(b)

    # count match
    matches = len(set(a) & set(b))

    score = 2 * float(matches) / float(lena + lenb)
    return score

def compute_overlap_stats(proj):

    exact_overlaps = [find_exact_overlap(d.events) for d in proj.annotation_documents]
    exact_overlaps = [x for x in exact_overlaps if x]
    exact_overlaps = util.flatten(exact_overlaps)

    overlaps_cnt = len(exact_overlaps)
    overlap_events_cnt = sum(len(x) for x in exact_overlaps)
    all_ev = []
    for d in proj.annotation_documents:
        for ev in d.events:
            all_ev.append(ev)
    all_ev_cnt = len(all_ev)
    overlap_pct = round(100 * overlap_events_cnt / all_ev_cnt, 1)
    return {
        "overlap_pct": overlap_pct,
        "overlaps_cnt": overlaps_cnt,
        "overlapping_event_cnt": overlap_events_cnt,
    }

def score_dice_pairwise(annotations_x, annotations_y):

    pairwise_dice_scores = np.zeros(shape=(len(annotations_x),len(annotations_y)))

    # score all events with dice coefficient
    for i, x in enumerate(annotations_x):

        for j, y in enumerate(annotations_y):

            dice_score = dice_coefficient(x, y)

            pairwise_dice_scores[i, j] = dice_score

    return pairwise_dice_scores

def unit_test():

    # test pairwise dice scoring
    x_y_token_ids = [
        # ([["1"]], [["1"]], [[1.]]), # full match
        # ([["1", "2"]], [["2"]], [[0.66]]), # test scoring
        # ([["1"], ["2"]], [["3"], ["4"]], [[0., 0.], [0., 0.]]), # no match multiple
        # ([["1"], ["2"]], [["2"], ["3"], ["4"]], [[0., 0., 0.], [1., 0., 0.]]), # one match, multiple
        ([["1", "2"], ["2"]], [["1"], ["2"]], [[0.66, 0.66], [0., 1.]]), # x self-overlaps" solution 0>0, 1>1 REQUIRES RESOLUTION
        ([["1"], ["2"]], [["1", "2"], ["2"]], [[0.66, 0.], [0.66, 1.]]), # y self-overlaps" solution 0>0, 1>1 REQUIRES RESOLUTION
        ([["1"], ["2"]], [["1", "2"], ["2", "3"]], [[0.66, 0.], [0.66, 0.66]]), # y-self overlaps, different boundaries: REQUIRES label check
    ]
    for x, y, correct in x_y_token_ids:
        dice_scores = score_dice_pairwise(x, y)
        dice_scores_orig = dice_scores.copy()
        npt.assert_almost_equal(np.array(correct), dice_scores, decimal=2) # assert equal with decimal precision 2

        mask = (dice_scores == dice_scores.max(axis=0, keepdims=True))
        dice_scores[~mask] = 0.
        mask2 = (dice_scores == dice_scores.max(axis=1, keepdims=True))
        dice_scores[~mask2] = 0.
        pass
    pass

def flatten(l):
    return [item for sublist in l for item in sublist]

def merge_tuples(edges):

    from collections import defaultdict

    def dfs(adj_list, visited, vertex, result, key):
        visited.add(vertex)
        result[key].append(vertex)
        for neighbor in adj_list[vertex]:
            if neighbor not in visited:
                dfs(adj_list, visited, neighbor, result, key)

    adj_list = defaultdict(list)
    for x, y in edges:
        adj_list[x].append(y)
        adj_list[y].append(x)

    result = defaultdict(list)
    visited = set()
    for vertex in adj_list:
        if vertex not in visited:
            dfs(adj_list, visited, vertex, result, vertex)

    return list(result.values())

def resolve_self_overlap(df, category_name, extent):
    # issue assign group by overlap

    # 1. isolate

    # get self-overlap: anno_id has multiple same group_id
    self_overlap = df.duplicated(subset=["anno_id", "group_id"], keep=False)
    if self_overlap.any():
        problem_groups = df[self_overlap].group_id.unique()
        df_resolve = df[df.group_id.isin(problem_groups)].sort_values("group_in_doc")
        pass
        for group_id, group_df in df_resolve.groupby(["group_in_doc"]):
            # if in same overlap-group: split group by most overlap per anno
            new_groups = []
            for (anno_id1, anno_group1), (anno_id2, anno_group2) in combinations(group_df.groupby(["anno_id"]), 2):
                pairwise_dice = pd.DataFrame(index=anno_group1.index, columns=anno_group2.index, dtype=float)
                for x in pairwise_dice.index: # pairwise matching
                    for y in pairwise_dice.columns:
                        x_token_ids = group_df.unit.loc[x].get_extent_token_ids(extent=[])
                        y_token_ids = group_df.unit.loc[y].get_extent_token_ids(extent=[])
                        dice_score = dice_coefficient(x_token_ids, y_token_ids)
                        pairwise_dice.loc[x, y] = dice_score

                # select max dicescore as match, if multiple max -> need to look at label to disambiguate
                match_idc = pairwise_dice[pairwise_dice == pairwise_dice.values.max()].stack().index.tolist()

                if len(match_idc) > 1 and match_idc == pairwise_dice.stack().index.tolist(): # multiple max -> need resolution of labels
                    for (x, y) in match_idc:
                        if df.loc[x][category_name] == df.loc[y][category_name]:
                            new_groups.append((x, y))
                else:
                    new_groups.extend(match_idc)

            new_groups = merge_tuples(new_groups)
            # exceptional case with split annotations after merge: same group [[1-2_anno_x]] [[3-4 anno_x]][[1-4 anno_y]]:
            # > create 2 groups
            newer_groups = []
            for newgroup in new_groups:
                newgroup_df = df.loc[newgroup].sort_values("anno_id")
                split_overlap = newgroup_df.duplicated(subset=["anno_id"], keep=False)
                if split_overlap.any():
                    x_idc = split_overlap.index[split_overlap].tolist()
                    y_idc = split_overlap.index[~split_overlap].tolist()
                    # select the largest overlap
                    mean_dice_coefs = []
                    for x in x_idc:
                        x_token_ids = newgroup_df.unit.loc[x].get_extent_token_ids(extent=extent)
                        dice_scores = [dice_coefficient(x_token_ids, newgroup_df.unit.loc[y].get_extent_token_ids(extent=extent)) for y in y_idc]
                        mean_dice_coefs.append(np.mean(dice_scores))
                    keep = x_idc[np.argmax(mean_dice_coefs)]
                    newgroup = [keep] + y_idc
                newer_groups.append(newgroup)

            # add every unmatched as new group
            for idx in group_df.index:
                if idx not in flatten(newer_groups):
                    newer_groups.append([idx])

            # change group_id
            for i, g in enumerate(newer_groups):
                for idx in g:
                    df.loc[idx,"group_id"] = df.loc[idx, "group_id"] + "_" + str(i+1)
        return df
    else:
        return df

def overlap_match_to_agreestat(proj, unit_name, extent=[], category_names=["polarity_sentiment"]):

    def union(data):
        from intervaltree import IntervalTree, Interval

        t = IntervalTree().from_tuples((begin, end+1) for begin, end in data)
        t.merge_overlaps(strict=True)
        return sorted(t.all_intervals)

    corpus_dfs = []
    doc_dfs = {cat: [] for cat in category_names}

    kf = lambda x: x.title
    for doc_id, docs in groupby(sorted(proj.annotation_documents, key=kf), kf):
        units = [] # collect all units from all annotators across the document group
        for doc in docs:
            units.extend(getattr(doc, unit_name))

        data = {
            "anno_id": [],
            "doc_id": [],
            "text": [],
            "begin": [],
            "end": [],
        }

        data["unit"] = units

        for u in units:
            tokens = u.get_extent_tokens(extent=extent)
            data["anno_id"].append(u.annotator_id)
            data["doc_id"].append(u.document_title.split("_")[0])
            data["text"].append(u.text)
            data["begin"].append(tokens[0].index)
            data["end"].append(tokens[-1].index)

            for cat in category_names: # add category label values
                data.setdefault(cat, []).append(getattr(u, cat))

        df = pd.DataFrame(data)
        # remove any exactly overlapping annos from same anno (redundant)
        # df = df.drop_duplicates(subset=["anno_id", "begin", "end", cat])
        # Create a list of intervals
        df['begin_end'] = df[['begin', 'end']].apply(list, axis=1)
        intervals = union(df.begin_end)

        # Add a group column
        df['group_in_doc'] = df['begin'].apply(lambda x: next(g for g,l in enumerate(intervals) if
                                                  l.contains_point(x)))

        df["group_id"] = df["doc_id"] + "_" + df['group_in_doc'].astype(str)

        corpus_dfs.append(df) # add to corpus overview



        # make Agreestat output dfs by category label
        for cat in category_names:

            # resolve self-overlap, i.e. same group within annotator, in exceptional cases the label is used
            df = resolve_self_overlap(df, cat, extent)
            doc_df = df.pivot(index="group_id", columns="anno_id", values=cat)
            doc_dfs[cat].append(doc_df)

    corpus_df = pd.concat(corpus_dfs) # combine all dfs into one corpus-level df
    corpus_df.to_csv("all_annotations_alignment_info.csv", sep="\t")
    for gn, gdf in corpus_df.groupby("group_id"):
        if len(gdf) >2 and gdf["end"].is_unique and gdf["begin"].is_unique:
            pass
            print(gdf)
    # write csv's for agreestat
    cat_dfs = {} # for output
    for cat, doc_dfs in doc_dfs.items():
        all_df = pd.concat(doc_dfs) # join all docs into corpus-level overview df

        cat_dfs[cat] = all_df
        fp = Path(f"agreestat-iaa-files/{unit_name}-{cat}.csv") # missing data as blank (default Agreestat)
        all_df.to_csv(fp, index=False)

        # Also output missing data as Missing label
        all_df_missing_labeled = all_df.fillna("Missing")
        fp = Path(f"agreestat-iaa-files/{unit_name}-{cat}-missing-labeled.csv") # missing data as blank (default Agreestat)
        all_df_missing_labeled.to_csv(fp, index=False)

    return cat_dfs

def pair_wise_missing_as_label_analysis(project):
    '''
    NOT USED in publication.
    Use NLTK agreement study package that handles missing labels as its own category. > bad approach
    In pairwise manner.
    :param project: WebAnno project parsed
    :return:
    '''

    # extract relevant annotations as document representation. (doc_id, anno_id, annotations_to_align)
    data = [[d.document_id, d.annotator_id, d.sentiment_expressions] for d in project.annotation_documents]

    data_iaa = {}

    kf = lambda x: x[0] # groupby doc_id
    for doc_id, docs in groupby(sorted(data, key=kf), kf):
        docs = list(docs)
        for d1, d2 in combinations(docs, 2): # pairwise matching
            anno_key = "-".join(sorted([d1[1], d2[1]]))
            matches = []
            unmatched = d1[2] + d2[2]
            x_token_ids = [x.get_extent_token_ids(extent=[]) for x in d1[2]]
            y_token_ids = [y.get_extent_token_ids(extent=[]) for y in d2[2]]
            dice_scores = score_dice_pairwise(x_token_ids, y_token_ids)

            for (x_idx, y_idx) in zip(*np.nonzero(dice_scores)):
                x = d1[2][x_idx]
                y = d2[2][y_idx]
                match = (x, y)
                try:
                    unmatched.remove(x)
                    unmatched.remove(y)
                except ValueError as e:
                    print(x, y, e)
                matches.append(match)

            print("--------Matched--------")
            for m1,m2 in matches:
                print(f"{m1.get_extent_text(extent=[])}.{x.polarity_sentiment} = {m2.get_extent_text(extent=[])}.{x.polarity_sentiment}")
            print("--------Unmatched--------")
            for x in unmatched:
                print(f"{x.get_extent_text(extent=[])}.{x.polarity_sentiment} -  {x.annotator_id}")

            # create data for custom anno task
            for i, (m1,m2) in enumerate(matches):
                data_iaa.setdefault(anno_key, []).append((m1.annotator_id, f"{doc_id}_{i}", m1.polarity_sentiment))
                data_iaa.setdefault(anno_key, []).append((m2.annotator_id, f"{doc_id}_{i}", m2.polarity_sentiment))
            for i, x in enumerate(unmatched):
                data_iaa.setdefault(anno_key, []).append((x.annotator_id, f"{doc_id}_{i+len(matches)}", x.polarity_sentiment))
                other_anno = [a for a in [d1[1], d2[1]] if a != x.annotator_id ][0]
                data_iaa.setdefault(anno_key, []).append((other_anno, f"{doc_id}_{i+len(matches)}", "NaN"))

    # 1.2. alignments = match_alignment(doc1, doc2, criteria=["full_overlap", "boundary", "partial_overlap"])
    # 1.3. write output file with matched + unmatched units

    # 2. compute same metrics with 1 reference (not correct handling of missing values)
    for anno_pair, data in data_iaa.items():
        t = CustomAnnotationTask(data)
        results = t.compute_all(average="micro")
        print(anno_pair, results)

def write_agreestat_token(project, extents = ['event_extent', 'participant_extent', 'filler_extent', 'canonical_referent_extent', 'discontiguous_trigger_extent', 'sentiment_expression_extent']):
    '''
    Writes agreestat data file for each type of extent with an entry for each token.
    Extents are identified by attributes on token containing extent.
    :param project:
    :return:
    '''

    def join_extents(data, to_join=(), new_name="new"):
        for anno_id in data[to_join[0]].keys():
            first = data[to_join[0]][anno_id]
            second = data[to_join[1]][anno_id]
            new = [new_name if x != None or y != None else None for x, y in zip(first, second)]
            data.setdefault(new_name, dict())[anno_id] = new
        for k in to_join:
            data.pop(k, None)
        return data

    data = {ext: {} for ext in extents}
    all_tokens = [t for d in project.annotation_documents for t in d.tokens]
    for doc in project.annotation_documents:
        for token in doc.tokens:
            for ext in extents:
                res = getattr(token, ext)
                if res:
                    anno = ext.split("_")[0]
                else:
                    anno = None
                data[ext].setdefault(token.annotator_id, []).append(anno)

    # join discontiguous and event extents into one trigger annotation
    if "discontiguous_trigger_extent" in extents and "event_extent" in extents:
        data = join_extents(data, to_join=("event_extent", "discontiguous_trigger_extent"), new_name="trigger_event")

    for extent_n, annotators in data.items():
        df = pd.DataFrame(annotators)
        dirp = Path(f"agreestat-iaa-files/token-identification")
        # df.to_csv(dirp /  (extent_n + ".csv"), index=False)
        df_no_anno = df.fillna("no_anno")
        df_no_anno.to_csv(dirp / (extent_n + "_no_labeled.csv"), index=False)

if __name__ == "__main__":

    # # Events: load EVENT IAA STUDY
    event_proj = parse_project(settings.IAA_XMI_DIRP, from_scratch=False)
    event_proj.annotation_documents = [d for d in event_proj.annotation_documents if d.annotator_id != "gilles"]

    categories = ["event_type", "event_fulltype", "polarity_negation", "modality"]
    matched_dfs = overlap_match_to_agreestat(event_proj, "events", extent=["discontiguous_triggers"], category_names=categories)

    # naive token approach for comparison:
    write_agreestat_token(event_proj, extents= ['event_extent', 'participant_extent', 'filler_extent', 'discontiguous_trigger_extent'])

    # Sentiment: load the final SENTIMENT IAA study project and set gold standard
    sent_proj = parse_project(settings.SENTIMENT_IAA)
    sent_proj.annotation_documents = [d for d in sent_proj.annotation_documents if d.annotator_id != "gilles"]

    # match overlap groups
    categories = ["polarity_sentiment", "polarity_sentiment_scoped", "negated", "uncertain"]
    matched_dfs = overlap_match_to_agreestat(sent_proj, "sentiment_expressions", extent=[], category_names=categories)

    # naive token approach for comparison:
    write_agreestat_token(sent_proj, extents=["sentiment_expression_extent"])