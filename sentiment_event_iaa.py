#!/usr/bin/env python3
"""
Perform IAA scoring on SENTIMENT POLARITY FOR EVENTS in the SENTiVENT sentiment corpus + make output file for agreestat360.com.
sentiment_event_iaa.py
webannoparser
5/13/20
Copyright (c) Gilles Jacobs. All rights reserved.
"""
import sys

sys.path.append("/home/gilles/repos/")

from nltk.metrics.agreement import AnnotationTask
from sklearn.preprocessing import LabelEncoder
from itertools import groupby, combinations
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import types
import sentivent_webannoparser.settings as settings
import sentivent_webannoparser.parse_project as pp
from itertools import groupby, combinations, chain
from pathlib import Path

def make_csv(data, opt_dirp="agreestat-iaa-files"):

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

def compute_percentage_agreement(data_clean):

    def check_equal(lst):
        return not lst or lst.count(lst[0]) == len(lst)

    key_func = lambda x: x[1]
    all_labels = [[x[2] for x in g] for k, g in groupby(sorted(data_clean, key=key_func), key_func)]
    agreed = []
    for l in all_labels:
        combos = list(combinations(l, 2))
        for pair in combos:
            if check_equal(pair):
                agreed.append(1.0 / len(combos))
            else:
                agreed.append(0.0)
    agreed = sum(agreed)
    agreed_pct = round(100 * agreed / len(all_labels), 2)
    agreed_all_raters = [check_equal(l) for l in all_labels]
    agreed_all_raters_pct = round(100 * sum(1 for x in agreed_all_raters if x) / len(all_labels), 2)
    print(f"{agreed_pct}% percentage agreement. {agreed_all_raters_pct}% (strict) of instances all raters agree.")
    return agreed_pct

def match_sentiment_expressions(project):

    # ("rater", "item", "label")
    # item > doc_id_itemid
    # matching based on token overlap by token_id

    from sympy import Interval, Union

    df = pd.DataFrame({'left': [0,5,10,3,12,13,18,31],
                       'right':[4,8,13,7,19,16,23,35]})

    def union(data):
        """ Union of a list of intervals e.g. [(1,2),(3,4)] """
        intervals = [Interval(begin, end) for (begin, end) in data]
        u = Union(*intervals)
        return [u] if isinstance(u, Interval) \
            else list(u.args)

    # Create a list of intervals
    df['left_right'] = df[['left', 'right']].apply(list, axis=1)
    intervals = union(df.left_right)

    # Add a group column
    df['group'] = df['left'].apply(lambda x: [g for g,l in enumerate(intervals) if
                                              l.contains(x)][0])
    pass



def parse_to_gamma(project, allowed, opt_dirp):
    '''
    Parse the annotation documents in a project to the format required by the Gamma computation tool [1].
    Format is cvs with each entry: anno_id, label, begin_index, end_index.
    Our base unit is the token and we evaluate at document level
    1. http://gamma.greyc.fr
    :param project:
    :return:
    '''
    Path(opt_dirp).mkdir(parents=True, exist_ok=True)

    pol_label_map = {"negative": 0, "neutral": 1, "positive": 3}

    docs = project.annotation_documents
    key_f = lambda x: x.document_id

    corpus_fp = Path(opt_dirp) / f"full_corpus.csv"
    offset = 0
    with(open(corpus_fp, "wt")) as corpus_out:

        for doc_id, doc_g in groupby(sorted(docs, key=key_f), key_f):
            data = []
            doc_fp = Path(opt_dirp) / f"{doc_id}.csv"

            with open(doc_fp, "wt") as f_out:
                unq_id = 0
                for doc in doc_g:
                    anno_id = doc.annotator_id
                    if anno_id in allowed:
                        seq_length = len(doc.tokens)
                        for i, se in enumerate(doc.sentiment_expressions):
                            begin = se.tokens[0].index
                            end = se.tokens[-1].index + 1
                            label = se.polarity_sentiment
                            unq_id += 1
                            f_out.write(f"{doc_id}_{unq_id},{anno_id},{label},,{begin},{end}\n")

                            # write corpus level
                            corpus_out.write(f"{doc_id}_{unq_id},{anno_id},{label},,{begin+offset},{end+offset}\n")
            offset += seq_length + 10 # add 10 extra offset for doc boundaries

        # df = pd.DataFrame(data)
        # df.to_csv(doc_fp, index=False, header=False)
        pass




if __name__ == "__main__":

    anno_id_allowed = ["jefdhondt", "elinevandewalle", "haiyanhuang"]
    # anno_id_allowed = ["elinevandewalle", "haiyanhuang"]
    # Create full clean corpus
    project = pp.parse_project(settings.SENTIMENT_IAA)
    project.annotation_documents = [d for d in project.annotation_documents if d.annotator_id in anno_id_allowed]

    # parse to gamma tool
    parse_to_gamma(project, anno_id_allowed, "gamma_iaa_files")

    # Sentiment Expression IAA:
    # se_all = list(project.get_annotation_from_documents("sentiment_expressions"))
    # data_matched = match_sentiment_expressions(project)

    # Polarity on Event IAA
    all_events = [ev for ev in project.get_events() if ev.annotator_id in anno_id_allowed]
    data = [(ev.annotator_id, ev.document_title.split("_")[0] + "_" + str(ev.element_id), str(ev.polarity_sentiment)) for ev in all_events]

    # unit test check they all match
    key_func = lambda x: x[1]
    data = sorted(data, key=key_func)
    data_clean = []
    missed = [] # in cvx05 jef accidentally deleted an event, messing up the ids. Disregard the mismatches
    for k, g in groupby(data, key_func):
        g = list(g)
        if len(g) !=len(anno_id_allowed):
            missed.extend(g)
        else:
            data_clean.extend(g)

    # fix cvx_05 missed jefdhondt manually
    missed.sort(key = lambda x: int(x[1].split("_")[1]))
    to_fix = list(filter(lambda x: "cvx05" in x[1], missed))
    for i in range(0, len(to_fix), len(anno_id_allowed)):
        same_item_candidate = to_fix[i:i + len(anno_id_allowed)]
        jef_item = next(x for x in same_item_candidate if x[0] == "jefdhondt")
        jef_id = next(int(x[1].split("_")[1]) for x in same_item_candidate if x[0] == "jefdhondt")
        correct_id = next(int(x[1].split("_")[1]) for x in same_item_candidate if x[0] == "elinevandewalle")
        if correct_id == jef_id + 1 or correct_id == jef_id - 1:
            jef_item_fixed = (jef_item[0], f"cvx05_{correct_id}", jef_item[2])
            same_item_fixed = [x for x in same_item_candidate if x != jef_item]
            same_item_fixed.append(jef_item_fixed)
            same_item_fixed = tuple(same_item_fixed)
            print(f"Fixed ids {same_item_candidate} > {same_item_fixed}")
            data_clean.extend(same_item_fixed)
            for i in same_item_candidate: missed.remove(i)

    # for i in missed:
    #     print(i)
    print(f"{len(missed)}/{len(all_events)} ({round(len(missed)*100/len(all_events),2)}%)")

    # make_csv for agreestat360.com
    make_csv(data_clean + missed)

    # compute own results (they do correspond to agreestat360.com, so implementation checks out).
    t = CustomAnnotationTask(data_clean)
    results = t.compute_all(average="micro")
    print(results)

    # let's check percentage aggreement acc here
    pct_agr = compute_percentage_agreement(data_clean)

    # print disagreements for examples
    disagrees = []
    key_func = lambda x: x[1]
    for k, u in groupby(sorted(data_clean, key=key_func), key_func):
        u = list(u)
        if len(set(x[2] for x in u)) != 1:
            disagrees.append(sorted(u, key=lambda x: x[0]))

    # match new sentiment expression strings
    # load the final IAA study project and set gold standard