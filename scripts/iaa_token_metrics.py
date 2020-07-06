#!/usr/bin/env python3
"""
iaa_token_metrics.py
sentivent_webannoparser
10/10/18
Copyright (c) Gilles Jacobs. All rights reserved.

DEPRECATED: token span metrics are not a good indicator of annotator performance
"""

from nltk.metrics.agreement import AnnotationTask
from sklearn.preprocessing import LabelEncoder
from itertools import groupby, combinations
from sklearn.metrics import f1_score, precision_score, recall_score
import numpy as np
import pandas as pd
import types
from functools import partial
from parser import *
from parse_project import parse_project

pd.set_option("display.expand_frame_repr", False)


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


if __name__ == "__main__":

    # use nltk.agreement AnnotationTask and customize
    # annotation unit will be determined in several ways
    # nltk.metrics.agreement data format [(coder, item, labels)]

    iaa_project = parse_project(settings.IAA_XMI_DIRP)

    token_level_data = []
    token_level_label = {}
    attribs = [
        "event_extent",
        "participant_extent",
        "filler_extent",
        "argument_extent",  # previous two combined
        "event_type",  # remove these from report
        "event_subtype",  # remove this from report
    ]

    # identification of annotation unit
    for doc in iaa_project.annotation_documents:
        doc_id = doc.title.split("_")[0]
        for token in doc.tokens:
            token_level_data.append((doc.annotator_id, f"{doc_id}_{token.index}"))
            for attrib in attribs:
                if "extent" in attrib:
                    if "argument" in attrib:
                        label = (
                            1
                            if (
                                getattr(token, "participant_extent")
                                or getattr(token, "filler_extent")
                            )
                            else -1
                        )
                    else:
                        label = -1 if getattr(token, attrib) is None else 1
                elif attrib == "event_type":
                    events = getattr(token, "event_extent")
                    label = (
                        "".join([str(getattr(ev, attrib)) for ev in events])
                        if events
                        else -1
                    )
                elif attrib == "event_subtype":
                    events = getattr(token, "event_extent")
                    label = (
                        "".join(
                            [f"{ev.event_type}.{ev.event_subtype}" for ev in events]
                        )
                        if events
                        else -1
                    )
                # elif attrib == ""
                token_level_label.setdefault(attrib, []).append(label)

    # event type label

    anns = ["anno_01", "anno_02", "anno_03"]
    # combo = list(combinations(anns, 2)) # for scoring anns against each other separately
    # combo.append(tuple(anns))
    combo = [tuple(anns)]
    for anno_pair in combo:
        print(anno_pair)
        all_ann_results = {}
        for annotation, labels in token_level_label.items():
            # data = [(ann_id, tokidx, (l,i)) for i, ((ann_id, tokidx), l) in
            #         enumerate(zip(token_level_data, labels)) if ann_id in anno_pair]
            data = [
                (ann_id, tokidx, l)
                for (ann_id, tokidx), l in zip(token_level_data, labels)
                if ann_id in anno_pair
            ]

            # t = CustomAnnotationTask(data, distance=("windowed_label_distance", {"window": 2}))
            t = CustomAnnotationTask(data)
            # all_ann_results[annotation] = {"Weighted Cohen": t.weighted_kappa()}
            if "extent" in annotation:
                averaging = "binary"
            else:
                averaging = "micro"
            all_ann_results[annotation] = t.compute_all(average=averaging)

        iaa_df = pd.DataFrame(all_ann_results)
        print(iaa_df)
        print(iaa_df.to_latex())
