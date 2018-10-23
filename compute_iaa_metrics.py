#!/usr/bin/env python3
"""
compute_iaa_metrics.py
sentivent-webannoparser
10/10/18
Copyright (c) Gilles Jacobs. All rights reserved.
"""

from nltk.metrics.agreement import AnnotationTask
from util import count_avg
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from itertools import groupby, combinations
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import types
from functools import partial
import pickle
from parser import WebannoProject
from dataclasses import dataclass

pd.set_option('display.expand_frame_repr', False)


class CustomAnnotationTask(AnnotationTask):
    '''
    Wrapper object aorund nltk.agreement.AnnotationTask object that allows for frp metrics to be computed.
    '''

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
        if isinstance(self.distance, tuple):  # if string it should be a function of this obj
            func_name, dist_kwargs = self.distance[0], self.distance[1]
            self.distance = partial(getattr(self, func_name), **dist_kwargs)
        elif callable(self.distance):  # else it should be a passed function
            pass
        else:
            raise ValueError(
                f"{self.distance} should be a tuple or a dict (\"func name of class method\", kwargs).")

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
            self.data.append({'coder': coder, 'labels': labels, 'item': item})

    def windowed_label_distance(self, label1, label2, window=1):
        pass

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
            labels_ann = self.label_encoder.transform([idat["labels"][0] if isinstance(idat["labels"], tuple)
                                                       else idat["labels"] for idat in item_data])
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


if __name__ == "__main__":

    FROM_SCRATCH = False
    opt_fp = "sentivent_en_webanno_project_iaa.pickle"

    iaa_ids = [
        "aapl14_iphone-x-s-dangerous-choice-of-market-share-or-profit.txt",
        "aapl15_here-s-how-apple-gets-to-a-2-trillion-market-value.txt",
        "aapl16_apple-s-app-store-generated-over-11-billion-in-revenue-for-th.txt",
        "amzn12_is-amazon-getting-into-the-pharmacy-business-this-is-what-you.txt",
        "amzn13_five-reasons-amazon-can-reach-1500.txt",
        "amzn14_amazon-sold-more-echo-dots-than-any-other-item-over-the-holidays.txt",
        "ba14_boeing-s-low-altitude-bid.txt",
        "ba15_should-boeing-buy-ge-aviation.txt",
        "ba16_boeing-s-stock-contributes-about-10-of-the-dow-s-1030-point.txt",
        "bac04_bank-of-america-earnings-hurt-by-tax-related-charge.txt",
        "bac05_bofa-includes-bitcoin-trust-in-broader-ban-on-investments.txt",
        "bac06_bank-of-america-hires-law-firm-to-help-probe-292-million-loss.txt",
        "cvx04_strong-crude-oil-no-help-for-chevron-exxon-mobil.txt",
        "cvx05_chevron-s-10-k-puts-the-permian-on-a-pedestal.txt",
        "cvx06_chevron-s-debt-fell-in-4q17-what-to-expect-in-2018.txt",
        "duk05_duke-energy-says-some-customers-may-be-affected-by-data-breach.txt",
        "duk06_like-many-of-its-peers-duk-is-trading-in-the-oversold-zone.txt",
        "duk07_duke-energy-stock-is-at-its-most-oversold-level-in-5-years.txt",
        "f13_ford-rolls-out-a-hot-rod-suv-as-drivers-abandon-performance-cars.txt",
        "f14_ford-is-the-next-ge-and-shorts-should-be-salivating.txt",
        "f15_ford-is-at-a-crossroad-of-danger-and-opportunity-in-china.txt",
        "jnj04_johnson-johnson-earnings-when-strong-is-nt-strong-enough.txt",
        "jnj05_where-s-the-tylenol-jj-disappoints-and-frustrates.txt",
        "jnj06_what-to-expect-from-johnson-johnson-in-2018.txt",
        "nem04_analyst-insight-is-newmont-mining-warming-up-for-a-good-2018.txt",
        "nem05_newmont-barrick-race-for-top-gold-crown-comes-down-to-a-decimal.txt",
        "nem06_newmont-mining-is-investors-gold-stock-to-buy.txt",
        "wmt05_walmart-stock-nears-key-support-after-earnings-miss.txt",
        "wmt06_goldman-expects-wal-mart-s-fortunes-to-improve-alongside-the-co.txt",
        "wmt07_walmart-s-meal-kits-are-not-the-solution-to-fight-amazon.txt",
    ]

    project_dirp = "exports/XMI_SENTiVENT-event-english-1_2018-10-04_1236"
    iaa_filter_func = lambda x: Path(x).parts[3] in iaa_ids and "anno" in Path(x).stem

    if not Path(opt_fp).is_file() or FROM_SCRATCH:
        # parse project
        event_project = WebannoProject(project_dirp)
        # filter so only iaa docs will be parsed
        event_project.annotation_document_fps = list(filter(iaa_filter_func, event_project.annotation_document_fps))
        event_project.parse_annotation_project()
        event_project.dump_pickle(opt_fp)
    else:
        with open(opt_fp, "rb") as project_in:
            event_project = pickle.load(project_in)

    avg_attribs = ["events", "sentences", "tokens", "participants", "fillers"]
    avg = {avg_attrib: count_avg(event_project.documents, avg_attrib, return_counts=True) for avg_attrib in avg_attribs}
    print(avg)

    all_event_types = []
    all_event_subtypes = []
    for doc in event_project.documents:
        for ev in doc.events:
            all_event_types.append(ev.event_type)
            all_event_subtypes.append(f"{ev.event_type}.{ev.event_subtype}")

    from collections import Counter

    print("Event types: ", Counter(all_event_types))
    print("Event subtypes", Counter(all_event_subtypes))

    # use nltk.agreement AnnotationTask and customize
    # annotation unit will be determined in several ways
    # nltk.metrics.agreement data format [(coder, item, labels)]

    token_level_data = []
    token_level_label = {}
    extent_attribs = ["event_extent", "participant_extent", "filler_extent", "canonical_referent_extent"]
    for doc in event_project.documents:
        doc_id = doc.title.split("_")[0]
        for token in doc.tokens:
            token_level_data.append((doc.annotator_id, f"{doc_id}_{token.index}"))
            for attrib in extent_attribs:
                label = -1 if getattr(token, attrib) is None else 1
                token_level_label.setdefault(attrib, []).append(label)

    anns = ["anno_01", "anno_02", "anno_03"]
    combo = list(combinations(anns, 2))
    combo.append(tuple(anns))
    for anno_pair in combo:
        print(anno_pair)
        all_ann_results = {}
        for annotation, labels in token_level_label.items():
            # data = [(ann_id, tokidx, (l,i)) for i, ((ann_id, tokidx), l) in
            #         enumerate(zip(token_level_data, labels)) if ann_id in anno_pair]
            data = [(ann_id, tokidx, l) for (ann_id, tokidx), l in
                    zip(token_level_data, labels) if ann_id in anno_pair]

            # t = CustomAnnotationTask(data, distance=("windowed_label_distance", {"window": 2}))
            t = CustomAnnotationTask(data)
            # all_ann_results[annotation] = {"Weighted Cohen": t.weighted_kappa()}
            all_ann_results[annotation] = t.compute_all()

        iaa_df = pd.DataFrame(all_ann_results)
        print(iaa_df)
    # parse to nltk data format at token-level

    # parse to

    # trigger scoring: for each token in trigger or not
    # type scoring: for each overlapping trigger: determine if type is good
