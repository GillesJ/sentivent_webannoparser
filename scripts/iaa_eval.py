#!/usr/bin/env python3
'''
iaa_eval.py
webannoparser 
10/23/18
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import numpy as np
import itertools
from pathlib import Path
import pickle
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import util
import operator
import pandas as pd
from parser import WebannoProject
from parser import Element
from iteration_utilities import duplicates
pd.set_option('display.max_columns', None)

# represent corpus as list of token ids:

class AgreementStudy:

    def __init__(self, project):

        self.project = project
        self.annotation_documents = None
        self.ere_matched = False

        self.discont_map = {"events": "discontiguous_triggers"}

        self._set_annotations()

    def dice_coefficient(self, a, b):

        a.sort()
        b.sort()
        if not len(a) or not len(b): return 0.0
        """ quick case for true duplicates """
        if a == b: return 1.0

        """ use python list comprehension, preferred over list.append() """

        # assignments to save function calls
        lena = len(a)
        lenb = len(b)

        # count match
        matches = len(set(a) & set(b))

        score = 2 * float(matches) / float(lena + lenb)
        return score



    def _match_partial_span(self, gold_anns, system_anns, system_annotator_id):
        '''
        Computes partial token span overlap scores using the dice coefficient.
        Token spans are list of unique token id's based on their position in the document.
        This is different from other matching that computes dice coefficient between types, i.e. not exact positions in the document.
        :param gold_anns: gold document annotation_documents
        :param system_anns: system annotation_documents
        :param system_annotator_id: id of system annotator
        :return: np array with dice token overlap scores
        '''

        gold_tokens = [ann.token_span for ann in gold_anns]  # testing
        system_tokens = [ann.token_span for ann in system_anns]
        token_combination = list(itertools.product(gold_tokens, system_tokens))
        token_id_combination = (([t.token_id for t in gold_tok], [t.token_id for t in sys_tok]) for gold_tok, sys_tok in token_combination)

        scores = np.fromiter((self.dice_coefficient(a, b) for (a, b) in token_id_combination), float)

        # for i, (gold_tok, sys_tok) in enumerate(token_combination): # for testing
        #     print(f"{i}: {scores[i]} {[t.text for t in gold_tok]} {[t.text for t in sys_tok]}")

        nz_idx = np.nonzero(scores)
        scores_nz = scores[nz_idx]

        annotation_combination = np.array(list(itertools.product(gold_anns, system_anns)))
        annotation_combination_nz = annotation_combination[nz_idx]

        for row in np.column_stack((annotation_combination_nz, scores_nz)):
            gold_ann = row[0]
            sys_ann = row[1]
            score= row[2]
            match = {"partial_token_sim": sys_ann, "position_dice_sim_score": score}
            if gold_ann.match_candidates[system_annotator_id] is not None:
                gold_ann.match_candidates[system_annotator_id].append(match)
            else:
                gold_ann.match_candidates[system_annotator_id] = [match]

    def _match_by_type(self, gold_anns, system_anns, system_annotator_id):

        def set_match(match_in_window, gold_ann, window=None):

            gold_ann_begin_token_pos = min(t.index for t in gold_ann.token_span)
            gold_ann_end_token_pos = max(t.index for t in gold_ann.token_span)
            gold_ann_token_text = [t.text for t in gold_ann.token_span]

            for m in match_in_window:
                match_ann_begin_token_pos = min(t.index for t in m.token_span)
                match_ann_end_token_pos = max(t.index for t in m.token_span)
                match_ann_token_text = [t.text for t in m.token_span]

                token_distance = max(gold_ann_begin_token_pos - match_ann_end_token_pos, match_ann_begin_token_pos - gold_ann_end_token_pos)
                dice_sim_score = self.dice_coefficient(gold_ann_token_text, match_ann_token_text)
                match = {
                    f"in_window_{window}": m,
                    "dice_sim_score": dice_sim_score,
                    "token_distance": token_distance,
                }
                if gold_ann.match_candidates[system_annotator_id] is not None:
                    gold_ann.match_candidates[system_annotator_id].append(match)
                else:
                    gold_ann.match_candidates[system_annotator_id] = [match]

        for gold_ann in gold_anns:

            # look for the same maintype in a set window
            selection_val = gold_ann.event_type  # TODO make the settings_attrib a keyword argument
            type_match = list(filter(lambda x: x.event_type == selection_val, system_anns))
            # limit to window criterium
            # window = same sentence
            gold_in_sentence_id = [s.element_id for s in gold_ann.in_sentence]
            match_in_sentence = list(
                filter(lambda x: [s.element_id for s in x.in_sentence] == gold_in_sentence_id, type_match))
            set_match(match_in_sentence, gold_ann, window="sentence")
            # window is document
            set_match(type_match, gold_ann, window="document")


    def match_candidate_annotations(self, annotation_name):
        '''
        Collect target annotation candidates that match by using several criteria.
        For each document one is considered gold and the rest system annotation_documents.
        This generates matches for each system on each gold annotation.
        Matching criteria are:
        1) Partial token span dice coefficient similarity scoring:
            Dice score for every pair of possible gold and system mentions.
        2) Mention mapping by type value.
        :param annotation_name: attribute name of document containing the relevant annotation_documents to match.
        :return:
        '''
        # annotation.tokens does not yet include discontiguous spans
        self.set_annotation_token_span(annotation_name)

        # set all possible match candidates on annotation
        for annotated_docs in zip(*self.annotation_documents.values()): # iterate over 3 docs

            for (gold_doc, system_doc) in itertools.permutations(annotated_docs, 2):

                gold_anns = getattr(gold_doc, annotation_name)

                # make an attrib for containing the system matches
                for gold_ann in gold_anns:
                    if hasattr(gold_ann, "match"):
                        gold_ann.match_candidates[system_doc.annotator_id] = None
                    else: gold_ann.match_candidates = {system_doc.annotator_id: None}

                system_anns = getattr(system_doc, annotation_name)

                # match partial span matches
                self._match_partial_span(gold_anns, system_anns, system_doc.annotator_id)

                # match annotation_documents by type that are not matched
                self._match_by_type(gold_anns, system_anns, system_doc.annotator_id)


    def set_annotation_token_span(self, annotation_attrib):

        def _set_token_id(doc_id):
            if not hasattr(token, "token_id"):
                tok_id = f"{doc_id}_{token.index}"  # todo eliminate this when using source docs
                token.token_id = tok_id

        if not hasattr(self.project.annotation_documents[0], annotation_attrib):
            raise ValueError(f"Attribute {annotation_attrib} is not on document.")
        if annotation_attrib in self.discont_map:
            print(
                f"Attribute {annotation_attrib} will be completed with discontiguous span "
                f"{self.discont_map[annotation_attrib]}.")
            if not hasattr(self.project.annotation_documents[0], self.discont_map[annotation_attrib]):
                raise ValueError(f"Attribute {self.discont_map[annotation_attrib]} is not on document.")

        for doc in self.project.annotation_documents:
            doc_id = doc.title.split("_")[0] # todo eliminate this when using source docs
            for ann in getattr(doc, annotation_attrib):
                ann.token_span = []
                # set doc for easier debugging
                ann.document = doc
                ann.matched_to_gold = {}
                for token in ann.tokens:
                    _set_token_id(doc_id)
                    ann.token_span.append(token)
                    # get the discont from the annotation
                if getattr(ann, self.discont_map[annotation_attrib]):
                    for discont_ann in getattr(ann, self.discont_map[annotation_attrib]):
                        for token in discont_ann.tokens:
                            _set_token_id(doc_id)
                            ann.token_span.append(token)

    def _set_annotations(self):
        '''
        Group the docs in the object by annotator.
        :return:
        '''
        annotation_documents = {}
        key = lambda x: x.annotator_id
        self.project.annotation_documents.sort(key=key)
        for ann_id, docs in itertools.groupby(event_project.annotation_documents, key):
            docs = list(docs)
            docs.sort(key = lambda x: x.title)
            annotation_documents[ann_id] = docs
        if annotation_documents: self.annotation_documents = annotation_documents
        # assert same len and assert same titles
        def check_equal(lst):
            return not lst or lst.count(lst[0]) == len(lst)

        if not check_equal([[d.title for d in x] for x in self.annotation_documents.values()]):
            raise ValueError("All titles of the documents are the same in IAA study.")

    def _get_annotation_from_match(self, match):
        return [x for x in match.values() if isinstance(x, Element)][0]

    def collect_annotations_by_annotator(self, annotation_name, attribute=None):
        '''
        Collect all gold annotations by annotation name and if attribute dict is set, with that attribute value on attribute.
        :param annotation_name: The name of the annotation unit as defined on a Document (cf. Document object attributes in parser.py).
        :param attribute: Attribute value pairs in tuple format to filter annotations of interest (e.g. events of certain types).
        :return: dict of annotations by annotator/system {system: [annotations]}
        '''
        all_annotations = {}
        for anno_id, gold_docs in self.annotation_documents.items():
            all_anns = []
            for doc in gold_docs:
                annotation = getattr(doc, annotation_name)
                if attribute is not None:
                    annotation = list(filter(lambda x: getattr(x, attribute[0]) == attribute[1], annotation))
                    pass
                all_anns.extend(annotation)
            all_annotations[anno_id] = all_anns

        return all_annotations

    def yield_annotations(self, annotation_name):
        for gold_docs in self.annotation_documents.values():
            for doc in gold_docs:
                annotations = getattr(doc, annotation_name)
                for ann in annotations:
                    yield ann

    def is_attribute_accurate(self, gold, match, attribute_name):

        if attribute_name == "event_subtype":
            gold_val = getattr(gold, "event_type") + str(getattr(gold, "event_subtype"))
            match_val = getattr(match, "event_type") + str(getattr(match, "event_subtype"))
        else:
            gold_val = getattr(gold, attribute_name)
            match_val = getattr(match, attribute_name)
        # gold_val = getattr(gold, attribute_name)
        # match_val = getattr(match, attribute_name)
        if gold_val == match_val:
            return True
        else:
            return False

    def score_ere_nugget(self, all_gold_annotations, attributes=["event_type", "event_subtype", "polarity_negation", "modality"]):

        if not self.ere_matched: raise ValueError("Types have not yet been matched.")
        # get the annotations which are matched
        scores_all = {}
        for gold_anno_id, gold_annotations in all_gold_annotations.items():
            scores_all[gold_anno_id] = {}
            # number of gold annotations needed for Recall and attribute accuracy calculation
            n_gold = len(gold_annotations)
            #  init TP, FP and attribute score dict = {other_anno_id: (TP, FP)}
            sys_ann_ids = [k for k in self.annotation_documents.keys() if k != gold_anno_id]
            metrics_template = {"tp": 0.0, "tp_attrib": 0.0, "fp": 0.0, "fp_attrib": 0.0, "n_matches": 0.0, "tp_relaxed": 0.0, "tp_attrib_relaxed": 0.0}
            attribute_acc = {attrib : 0.0 for attrib in attributes}
            metrics_template.update(attribute_acc)
            scores = {k: copy.deepcopy(metrics_template) for k in sys_ann_ids}

            # for every gold: get sys matches
            for gold_ann in gold_annotations:
                # for every match:
                for sys_anno_id, sys_matches in gold_ann.selected_match.items():
                    if sys_matches is not None:
                        # score attribute accuracy
                        scores[sys_anno_id]["n_matches"] += 1.0 # count the valid span matches to not penalize twice for span errors
                        for sys_match in sys_matches:
                            sys_match["attribute_match"] = {} # collect if the attribute matches True False: needed for computing tp_attrib
                            for attribute_name in attributes:
                                attribute_is_accurate = self.is_attribute_accurate(
                                        gold_ann,
                                        self._get_annotation_from_match(sys_match),
                                        attribute_name)
                                sys_match["attribute_match"][attribute_name] = attribute_is_accurate
                                if attribute_is_accurate:
                                    scores[sys_anno_id][attribute_name] += 1.0 / len(sys_matches) # attribute is accurate count
                        # compute mention level span score: compute true pos en false pos
                        # if two spans match in system: add the dice score of the maximum value: do not double count the coordinated / erroneously split one
                        max_span_dice_match = max(sys_matches, key=lambda x: x["position_dice_sim_score"])
                        scores[sys_anno_id]["tp"] += max_span_dice_match["position_dice_sim_score"]
                        scores[sys_anno_id]["tp_relaxed"] += 1.0
                        if all(max_span_dice_match["attribute_match"].values()): # all attributes are accurate is the condition for attribute based TP
                            scores[sys_anno_id]["tp_attrib"] += max_span_dice_match["position_dice_sim_score"]
                            scores[sys_anno_id]["tp_attrib_relaxed"] += 1.0
                        else:
                            scores[sys_anno_id]["fp_attrib"] += 1.0
                    else:
                        scores[sys_anno_id]["fp"] += 1.0
                        scores[sys_anno_id]["fp_attrib"] += 1.0

            #  compute P, R, F1, acc
            for sys_anno_id, metrics in scores.items():
                n_sys = len(all_gold_annotations[sys_anno_id])
                metrics["n_sys"] = n_sys
                metrics["n_gold"] = n_gold
                metrics["p"] = metrics["tp"] / (metrics["tp"] + metrics["fp"])
                metrics["p_alt"] = metrics["tp"] / n_sys
                metrics["p_relaxed"] = metrics["tp_relaxed"] / n_sys
                metrics["p_attrib"] = metrics["tp_attrib"] / (metrics["tp_attrib"] + metrics["fp_attrib"])
                metrics["p_attrib_alt"] = metrics["tp_attrib"] / n_sys
                metrics["p_attrib_relaxed"] = metrics["tp_attrib_relaxed"] / n_sys
                metrics["r"] = metrics["tp"] / n_gold
                metrics["r_attrib"] = metrics["tp_attrib"] / n_gold
                metrics["r_relaxed"] = metrics["tp_relaxed"] / n_gold
                metrics["r_attrib"] = metrics["tp_attrib"] / n_gold
                metrics["r_attrib_relaxed"] = metrics["tp_attrib_relaxed"] / n_gold
                metrics["f1_span"] = (2 * metrics["p"] * metrics["r"]) / (metrics["p"] + metrics["r"])
                metrics["f1_span_alt"] = (2 * metrics["p_alt"] * metrics["r"]) / (metrics["p_alt"] + metrics["r"])
                metrics["f1_attrib"] = (2 * metrics["p_attrib"] * metrics["r_attrib"]) / (
                        metrics["p_attrib"] + metrics["r_attrib"])
                metrics["f1_attrib_alt"] = (2 * metrics["p_attrib_alt"] * metrics["r_attrib"]) / (
                        metrics["p_attrib_alt"] + metrics["r_attrib"])
                metrics["f1_span_relaxed"] = (2*metrics["p_relaxed"] * metrics["r_relaxed"]) / (
                        metrics["p_relaxed"] + metrics["r_relaxed"])
                metrics["f1_attrib"] = (2 * metrics["p_attrib"] * metrics["r_attrib"]) / (
                            metrics["p_attrib"] + metrics["r_attrib"])
                metrics["f1_attrib_relaxed"] = (2 * metrics["p_attrib_relaxed"] * metrics["r_attrib_relaxed"]) / (
                            metrics["p_attrib_relaxed"] + metrics["r_attrib_relaxed"])
                # compute acc
                for attribute_name in attributes:
                    metrics[f"{attribute_name}_acc"] = metrics[attribute_name] / metrics["n_matches"]

            # collect scores
            scores_all[gold_anno_id].update(scores)

        results = self.average_ere_scores(scores_all)

        return results

    def average_ere_scores(self, scores_all):
        '''
        Return the average by annotator and average of all nugget scores.
        :param self:
        :param scores_all: all scores by gold and system in embedded dict.
        :return: all_results: dict containing all steps of averaging for full overview of performance.
        '''
        all_results = {}
        all_results["all_scores"] = scores_all
        all_results["avg_annotator_scores"] = {}
        for k_gold, system_scores in scores_all.items():
            sys_scores = list(system_scores.values())
            all_results["avg_annotator_scores"][k_gold] = { k : sum(t[k] for t in sys_scores)/len(sys_scores) for k in sys_scores[0] }

        avg_scores_by_anno = list(all_results["avg_annotator_scores"].values())
        all_results["avg_all_scores"] = { k : sum(t[k] for t in avg_scores_by_anno)/len(avg_scores_by_anno) for k in avg_scores_by_anno[0] }

        return all_results



    def resolve_match_coordination(self, gold_ann, selected_match):
        '''
        Resolve cases in which multiple identical matches are made on the same position.
        This only happens for coordinated events with participant coordination.
        Two cases are possible:
        a. Gold is coordinated: the matched coordination depends on the order in which the annotator annotated the events.
        This is implemented by the index of the gold coordinated element.
        We do not put checks on attributes to get a better match as this would unfairly skew matching.
        b. Gold is uncoordinated: all selected matches are kept.
        :param gold_ann: gold reference annotation element
        :param selected_match: selected candidate matches
        :return: selected_match: selected matches with wrongly coordinated element removed.
        '''

        selected_anns = [self._get_annotation_from_match(m) for m in selected_match]

        # resolve coordination issues: select the same coordinated element
        if len(selected_anns) > 1:
            # does the gold ann have coordination?
            if gold_ann.coordination:
                gold_ann_coordination_position = gold_ann.coordination.index(gold_ann)
                selected_match = [selected_match[gold_ann_coordination_position]]

        return selected_match

    def select_ere_matches(self, annotation_name):
        '''

        :param annotation_name:
        :return:
        '''
        all_gold_annotations = self.collect_annotations_by_annotator(annotation_name)
        for gold_anno_id, gold_annotations in all_gold_annotations.items():
            already_matched = {} # set for checking unique annotations

            for gold_ann in gold_annotations:
                gold_ann.selected_match = {} # dict for collected selected match

                for system_anno_id, candidate_system_matches in gold_ann.match_candidates.items():
                    gold_ann.selected_match[system_anno_id] = None
                    already_matched.setdefault(system_anno_id, set())

                    if candidate_system_matches is not None:
                        candidate_system_matches = list(filter(lambda x: "partial_token_sim" in x, candidate_system_matches))

                        if candidate_system_matches is not None:
                            if len(candidate_system_matches) > 1:
                                # multiple system annotations can be mapped to gold if they overlap with by same amount
                                max_sim = max(candidate_system_matches, key=lambda x: x["position_dice_sim_score"])[
                                    "position_dice_sim_score"]
                                selected_match = [x for x in candidate_system_matches if x["position_dice_sim_score"] == max_sim]
                            else:
                                selected_match = candidate_system_matches

                            selected_match = self.resolve_match_coordination(gold_ann, selected_match)

                            for sel_match in selected_match:
                                ann_match = self._get_annotation_from_match(sel_match)
                                if ann_match not in already_matched[system_anno_id]:
                                    if gold_ann.selected_match[system_anno_id] is not None:
                                        gold_ann.selected_match[system_anno_id].append(sel_match)
                                    else: gold_ann.selected_match[system_anno_id] = [sel_match]
                                    # add the gold that the system annotation is matched to
                                    ann_match.matched_to_gold.setdefault(gold_anno_id, []).append(gold_ann)

                                already_matched[system_anno_id].add(ann_match)

                        else: gold_ann.selected_match[system_anno_id] = None
                    else: gold_ann.selected_match[system_anno_id] = None

        self.print_selected_match(annotation_name)

        duplicates_found, _ = self.check_duplicate_selected_match("events")  # this is a sanity check
        assert not duplicates_found
        self.ere_matched = True


    def print_selected_match(self, annotation_name):
        all_gold_annotations = self.collect_annotations_by_annotator(annotation_name)
        for gold_anno_id, gold_annotations in all_gold_annotations.items():
            for gold_ann in gold_annotations:
                for other_anno_id, match in gold_ann.match_candidates.items():
                    print(
                        f"\n{'-'*160}"
                        f"\nMATCH FROM {gold_anno_id} TO {other_anno_id} IN {gold_ann.document.title}"
                        f"\n{gold_ann} (with selection strat Liu)\n"
                    )
                    if gold_ann.match_candidates[other_anno_id] is not None:
                        if gold_ann.selected_match[other_anno_id] is None:
                            print("No matched were selected from candidates.")
                        else:
                            for match in gold_ann.match_candidates[other_anno_id]:
                                matched_to = self._get_annotation_from_match(match).matched_to_gold
                                if match in gold_ann.selected_match[other_anno_id]:
                                    # print(f"\033[1m-> {match}\033[0m")
                                    print(f"->\t{match}\t{matched_to}")
                                else:
                                    print(f"{match}\t{matched_to}")
                    else:
                        print("No candidate matches were found for this annotation.")


    def check_duplicate_selected_match(self, annotation_name):
        '''
        Test to check
        :return:
        '''
        # check if there are double matches: there cannot be two or more system matches per annotation for one annotator
        all_gold_annotations = self.collect_annotations_by_annotator(annotation_name)
        flat_matches = {}
        for gold_anno_id, gold_annotations in all_gold_annotations.items():
            flat_matches[gold_anno_id] = {}
            for gold_ann in gold_annotations:
                for sys_anno_id, sys_match in gold_ann.selected_match.items():
                    if sys_match is not None:
                        sys_match = [self._get_annotation_from_match(m) for m in sys_match]
                        flat_matches[gold_anno_id].setdefault(sys_anno_id, []).extend(sys_match)

        duplicate_entries = {}
        duplicates_found = False
        for gold_anno_id, sys_matches in flat_matches.items():
            duplicate_entries[gold_anno_id] = {}
            for sys_anno_id, matches in sys_matches.items():
                dupes = list(duplicates(matches))
                if dupes:
                    duplicates_found = True
                duplicate_entries[gold_anno_id][sys_anno_id] = [match for match in matches if match in dupes]
        return duplicates_found, duplicate_entries

    def get_selected_match_counts(self, annotation_name):
        all_anns = iaa_study.collect_annotations_by_annotator(annotation_name)
        selected_matches = {k: {kk: [] for kk in [kkk for kkk in all_anns.keys() if kkk is not k]} for k in
                            all_anns.keys()}
        for anno_id, annotations in all_anns.items():
            for ann in annotations:
                for other_anno_id, matches in ann.selected_match.items():
                    if matches:
                        selected_matches[anno_id].setdefault(other_anno_id, []).append(matches)
        cnt = {g: {s: (len(match), round(100 * len(match) / len(all_anns[g]), 2)) for s, match in m.items()} for g, m in
               selected_matches.items()}
        return cnt

    def _select_partial_span_match(matches):
        # todo add double spn and matching
        selected_match = None
        partial_span_matches = list(filter(lambda x: "partial_token_sim" in x, matches))
        if partial_span_matches:
            if len(partial_span_matches) > 1:
                selected_match = max(partial_span_matches, key=lambda x: x["position_dice_sim_score"])
            else:
                selected_match = partial_span_matches[0]
        return selected_match

    def _select_match_by_key(matches, match_key=None):
        selected_match = None
        valid_match_keys = ["in_window_sentence", "in_window_document"]
        # if token window is set by "in_window_int" match_keys to set the window to match.
        # we use the candidate matches in the the document "in_window_document"
        if match_key not in valid_match_keys:
            window = int(match_key.split("in_window_")[-1])
            window_matches = list(filter(lambda x: "token_distance" in x and
                                                   x["token_distance"] <= window, matches))
        # else, use the valid match_key
        else:
            window_matches = list(filter(lambda x: match_key in x, matches))
        if window_matches:
            # if multiple potential matches were found by the key, select with the highest token type dice similarity and then shortest token distance
            if len(window_matches) > 1:
                selected_match = max(window_matches, key=lambda x: (x["dice_sim_score"], -x["token_distance"]))
            else:
                selected_match = window_matches[0]
        return selected_match

    def select_match(matches, strategy=["partial_token_sim", "in_window_sentence"]):
        selected_match = None
        if matches is not None:
            for match_key in strategy:
                # partial token_sim needs the max operator
                if match_key == "partial_token_sim":
                    selected_match = _select_partial_span_match(matches)
                else:
                    selected_match = _select_match_by_key(matches, match_key = match_key)
                if selected_match is not None: break
coordinated
            return selected_match

    def select_custom_match(self, annotation_name, strategy=["partial_token_sim", "in_window_sentence"]):
        all_gold_annotations = self.collect_annotations_by_annotator(annotation_name)

        for gold_anno_id, gold_annotations in all_gold_annotations.items():
            already_matched = {} # set for checking unique annotations

            for gold_ann in gold_annotations:
                gold_ann.selected_match = {} # dict for collected selected match

                for system_anno_id, candidate_system_matches in gold_ann.match_candidates.items():
                    gold_ann.selected_match[system_anno_id] = None
                    already_matched.setdefault(system_anno_id, set())

                    if candidate_system_matches is not None:
                        # filter the matches by strategy
                        candidate_system_matches = list(filter(lambda x: "partial_token_sim" in x, candidate_system_matches))

                        if candidate_system_matches is not None:
                            if len(candidate_system_matches) > 1:
                                # multiple system annotations can be mapped to gold if they overlap with by same amount
                                max_sim = max(candidate_system_matches, key=lambda x: x["position_dice_sim_score"])[
                                    "position_dice_sim_score"]
                                selected_match = [x for x in candidate_system_matches if x["position_dice_sim_score"] == max_sim]
                            else:
                                selected_match = candidate_system_matches

                            selected_match = self.resolve_match_coordination(gold_ann, selected_match)

                            # set selected match if not in already seen
                            for sel_match in selected_match:
                                ann_match = self._get_annotation_from_match(sel_match)
                                if ann_match not in already_matched[system_anno_id]:
                                    if gold_ann.selected_match[system_anno_id] is not None:
                                        gold_ann.selected_match[system_anno_id].append(sel_match)
                                    else: gold_ann.selected_match[system_anno_id] = [sel_match]
                                    # add the gold that the system annotation is matched to
                                    ann_match.matched_to_gold.setdefault(gold_anno_id, []).append(gold_ann)

                                already_matched[system_anno_id].add(ann_match)

                        else: gold_ann.selected_match[system_anno_id] = None
                    else: gold_ann.selected_match[system_anno_id] = None

        self.print_selected_match(annotation_name)

        duplicates_found, _ = self.check_duplicate_selected_match("events")  # this is a sanity check
        assert not duplicates_found
        self.ere_matched = True


    def plot_confusion_matrix(self, annotation_pairs, attrib="event_type", filepath="confusion_matrix.png"):

        def _plot_confusion_matrix(cm, classes,
                                   title='Confusion matrix',
                                   ylabel='True_label',
                                   xlabel='Predicted label',
                                   cmap=plt.cm.Greens):
            """
            This function prints and plots the confusion matrix.
            Normalization over class support is added.
            """
            cm_normalized = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
            print("Normalized confusion matrix")
            print(cm_normalized)

            im = plt.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
            plt.title(title)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=90)
            plt.yticks(tick_marks, classes)

            thresh = cm_normalized.max() / 2.
            for i, j in itertools.product(range(cm_normalized.shape[0]), range(cm_normalized.shape[1])):
                pct = f"{cm_normalized[i, j]:.1f}\n"
                n = f"\nn={int(round(cm[i, j])):d}"
                text_spec = {"horizontalalignment": "center", "verticalalignment": "center",
                             "color": "white" if cm_normalized[i, j] > thresh else "black"}
                if cm[i, j]:
                    plt.text(j, i, pct, **text_spec)
                    plt.text(j, i, n, **text_spec, fontsize="x-small")
                else:
                    plt.text(j, i, "0", **text_spec)

            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            plt.colorbar(im, fraction=0.046, pad=0.04)


        # make the pairings of attributes
        class_names = set()
        attrib_pairs = []
        id_pairs = []
        for gold_anno_id, gold_annotations in annotation_pairs.items():
            gold_attrib_vals = []
            match_attrib_vals = {k: [] for k in annotation_pairs.keys() if k != gold_anno_id}
            for gold_ann in gold_annotations:
                # for every match:
                class_names.add(getattr(gold_ann, attrib))
                gold_attrib_vals.append(getattr(gold_ann, attrib))
                for sys_anno_id, sys_matches in gold_ann.selected_match.items():
                    if sys_matches is not None:
                        match_attrib = getattr(self._get_annotation_from_match(sys_matches[0]), attrib)
                    else: match_attrib= "None"
                    match_attrib_vals.setdefault(sys_anno_id, []).append(match_attrib)
                    class_names.add(match_attrib)

            for match_id, matches in match_attrib_vals.items():
                attrib_pairs.append((gold_attrib_vals, matches))
                id_pairs.append((gold_anno_id, match_id))

        # plot
        class_names = sorted(list(class_names))
        class_names.append(class_names.pop(class_names.index('None'))) # move None to last position for plot aesthetics
        classes = np.array(class_names)
        all_cnf_matrix = []
        figsize = (10, 10)
        for (ref_id, match_id), (ref, match) in zip(id_pairs, attrib_pairs):
            cnf_matrix = confusion_matrix(ref, match, labels=classes)
            all_cnf_matrix.append(cnf_matrix)
            np.set_printoptions(precision=2)

            # Plot non-normalized confusion matrix
            fig = plt.figure(figsize=figsize)
            _plot_confusion_matrix(cnf_matrix, classes=classes, ylabel=ref_id, xlabel=match_id,
                                  title='Confusion matrix with normalization')
            plt.show()
            fig.savefig(f"{ref_id}_{match_id}_conf_matrix.svg")

        mean_cnf = np.mean(all_cnf_matrix, axis=0)
        cnf_df = pd.DataFrame(mean_cnf).set_index(classes)
        cnf_df.columns = classes
        print(cnf_df)
        fig = plt.figure(figsize=figsize)
        _plot_confusion_matrix(mean_cnf, classes=classes, ylabel="all_ref_avg", xlabel="all_sys_avg",
                               title='Confusion matrix with normalization all comparison average')
        plt.show()
        fig.savefig("all_average_conf_matrix.svg")

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

    iaa_study = AgreementStudy(event_project)
    iaa_study.match_candidate_annotations("events") # create all possible matches according to different criteria
    iaa_study.select_ere_matches("events") # select the partial span ere nugget match
    cnt = iaa_study.get_selected_match_counts("events")
    all_attributes = ["event_type", "event_subtype", "polarity_negation", "modality"] # TODO think about event participants

    # score ere nuggets
    all_gold_annotations = iaa_study.collect_annotations_by_annotator("events")
    ere_nugget_results = iaa_study.score_ere_nugget(all_gold_annotations, attributes=all_attributes)

    # plot confusion matrices
    iaa_study.plot_confusion_matrix(all_gold_annotations)

    # attribute_combinations = [] # combined attributes are not very informative, separate and all combined is more interpretable
    # for r in range(1, len(all_attributes)+1):
    #     attribute_combinations.extend(list(itertools.combinations(all_attributes, r)))
    # attribute_combinations_results = {"+".join(attrib_combo): iaa_study.score_ere_nugget("events", attributes=attrib_combo) for attrib_combo in attribute_combinations}
    separate_attributes_results = {attrib: iaa_study.score_ere_nugget(all_gold_annotations, attributes=[attrib]) for attrib in all_attributes}

    ia_span_f1 = ere_nugget_results["avg_all_scores"]["f1_span_alt"]
    ia_all_attrib_f1 = ere_nugget_results["avg_all_scores"]["f1_attrib_alt"]
    ia_type_f1 = separate_attributes_results["event_type"]["avg_all_scores"]["f1_attrib_alt"]
    ia_subtype_f1 = separate_attributes_results["event_subtype"]["avg_all_scores"]["f1_attrib_alt"]
    ia_polarity_f1 = separate_attributes_results["polarity_negation"]["avg_all_scores"]["f1_attrib_alt"]
    ia_modality_f1 = separate_attributes_results["modality"]["avg_all_scores"]["f1_attrib_alt"]
    ia_type_acc = ere_nugget_results["avg_all_scores"]["event_type_acc"]
    ia_subtype_acc = ere_nugget_results["avg_all_scores"]["event_subtype_acc"]
    ia_polarity_acc = ere_nugget_results["avg_all_scores"]["polarity_acc"]
    ia_modality_acc = ere_nugget_results["avg_all_scores"]["modality_acc"]

    # to get an idea of type difficulty: get f1 scores of gold annotations by event type
    all_event_types = []
    all_event_subtype = []
    for doc in event_project.annotation_documents:
        if doc.events:
            for ev in doc.events:
                all_event_types.append(ev.event_type)

    event_type_score = {}
    for event_type in set(all_event_types):
        type_gold_annotations = iaa_study.collect_annotations_by_annotator("events", attribute=("event_type", event_type))
        type_ere_nugget_results = iaa_study.score_ere_nugget(type_gold_annotations, attributes=all_attributes)
        event_type_score[event_type] = type_ere_nugget_results["avg_all_scores"]

    filter_scores = ["f1_span_alt", "f1_attrib_alt", "n_gold"]
    df = pd.DataFrame(event_type_score).transpose()
    df = df.filter(items=filter_scores)
    df['f1_span_alt_rank'] = df['f1_span_alt'].rank(ascending=False)
    df['f1_attrib_alt_rank'] = df['f1_attrib_alt'].rank(ascending=False)
    df = df.sort_values(by="f1_span_alt", ascending=False)
    print(df)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=[9,6])
    figax = fig.add_subplot(111)
    ax = df[["f1_span_alt", "f1_attrib_alt"]].plot(ax=figax)
    ax.set_xticks(list(range(18)))
    ax.set_xticklabels(list(df.index), rotation = 45, ha="right")
    ax.legend(["Span F1-score", "Span+attributes F1-score"]);
    plt.show()
    fig.savefig("event_type_f1_scores.png")
    pass

    # iaa_study.select_custom_match("events")
    # custom_nugget_results = iaa_study.score_ere_nugget("events", )

    # TODO Note: inform yourself by using NLTK agreement data format

# a. Direct match based on f1. Get a separate score for this:
    # collect all annotation_documents


    # get list per anno
    # # detection task: THIS IS CONCEPTUALLY WRONG
    # # (coder, item, label) if two items match: generate same item id and pos label,
    # detect_data = []
    # seen = []
    # for gold_anno_id, other_ann in all_matches.items():
    #     for other_anno_id, annotation_documents in other_ann.items():
    #         for (gold_ann, match_ann) in annotation_documents:
    #             if match_ann is not None:
    #                 match_ann = [x for x in match_ann.values() if isinstance(x, Element)][0]
    #                 ids = sorted([gold_ann.element_id, match_ann.element_id])
    #                 ids = sorted([hash((gold_anno_id, gold_ann.text, gold_ann.element_id)), hash((other_anno_id, match_ann.text, match_ann.element_id))])
    #                 match_id = gold_ann.document.title + "_" + '_'.join(map(lambda x: str(x), ids))
    #                 l = 1
    #             else:
    #                 # ids = sorted([gold_ann.element_id, 0])
    #                 ids = sorted([hash((gold_anno_id, gold_ann.text, gold_ann.element_id))])
    #                 match_id = gold_ann.document.title + "_" + '_'.join(map(lambda x: str(x), ids))
    #                 l = -1
    #             detect_data.append((gold_anno_id, match_id, 1))
    #             detect_data.append((other_anno_id, match_id, l))

    # type task
    # (coder, item, label) if two items match: generate same item id and pos label,
    # type_data = []
    # for gold_anno_id, other_ann in all_matches.items():
    #     for other_anno_id, annotation_documents in other_ann.items():
    #         for (gold_ann, match_ann) in annotation_documents:
    #             if match_ann is not None:
    #                 match_ann = [x for x in match_ann.values() if isinstance(x, Element)][0]
    #                 ids = sorted([hash((gold_anno_id, gold_ann.text, gold_ann.element_id)), hash((other_anno_id, match_ann.text, match_ann.element_id))])
    #                 match_id = gold_ann.document.title + "_" + '_'.join(map(lambda x: str(x), ids))
    #                 l = 1
    #
    # d = list(duplicates(detect_data))
    # t = custom(annotation)

# b. Direct match and type match with same type partial span overlap:


# c. Direct match and type match with same type partial span overlap and set windows:
#
#     dupe_cnt = 0
#     for ev in gold_doc.events:
#         match = ev.match[system_doc.annotator_id]
#         if match and len(match) >= 2:
#             print(match)
#             dupe_cnt += 1
#             pass
#     pass
#
#     # TODO resolve coordination: partial_tokenspan score = 1.0 for two in system document