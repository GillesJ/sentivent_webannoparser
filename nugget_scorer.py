#!/usr/bin/env python3
"""
nugget_scorer.py
webannoparser 
4/5/19
Copyright (c) Gilles Jacobs. All rights reserved.

This code is an implementation of Liu et al. (2015) event nugget scorer.
It currently scores at the document-level and scores are macro-avg there.

N.B: selecting match with maximum dice score is already done by Matcher:
i.e. the argmax steps are precomputed before the Scorer steps in Algorithm 3 and 4 in Liu et al. 2015).

[1] Liu, Z., Mitamura, T., & Hovy, E. (2015). Evaluation Algorithms for Event Nugget Detection : A Pilot Study. Proceedings of the 3rd Workshop on EVENTS at the NAACL-HLT, 53–57.

Implementation approach:
- Score a nugget in 2 steps: 1. matching and 2. scoring: different settings can be set for both phases.
- 1. Matching: two step process consists of matching token spans by:
- 1.1 Candidate search: match based on token span dice-score > 0.0 for ERE, other search criteria can be entered too for custom matching.
- 1.2. Selection of candidates: Using heuristics select final candidates: important for obtaining good matches.
-2. Scoring:

TODO:
- Check accuracy counts as in [1] Algorithm 3
- Change the matching (but not the scoring) to be less strict because of differences with more lose matching.
- Handle preprocessing of coordinated and "multiple option" gold annotations in Matching
    -
    -
-
"""
from parser import *
from parse_project import parse_project, parse_corpus
from functools import partial
import re
import util
from pathlib import Path
from itertools import groupby, product
from collections import Counter
import settings
import pandas as pd


class MentionMapper:
    """
    MentionMapper for matching annotations using several strategies.
    Generates candidate matches for an annotation unit in a system document using a matching strategy in the 'match' method.
    The select method selects the final match from all candidates for an annotation based on several criteria.
    """

    def match_candidates(self, gold, system, strategy):
        """

        :param system: systemerence document
        :param gold: Gold standard document
        :param strategy:
        :return:
        """
        matcher = get_matcher(strategy)
        return matcher(system, gold)

    def select(self, candidates, criteria):

        selector = get_selector(criteria)
        return selector(candidates)

    def match(self, gold, system, match_strategy="ere", select_criteria="ere"):
        """
        Make

        :param system: System document.
        :param gold: Gold standard document.
        :param match_strategy:
        :param select_criteria:
        :return:
        """

        matcher = get_matcher(match_strategy)
        candidates = matcher(gold, system)  # gold annotations with mapped attribute

        collect_match_metadata(candidates, len(gold.events), len(system.events))

        selector = get_selector(select_criteria)
        matches = selector(candidates)

        # set matched system event as attrib on gold event, simplifies calls
        for gold_ann in gold.events:
            gold_ann.ere_mapping = []

            for (gold_m, system, dice_score) in matches:
                if gold_ann == gold_m:
                    gold_ann.ere_mapping.append((system, dice_score))

        return gold


def collect_match_metadata(mappings, n_gold, n_system):
    # system annotation cannot be mapped to multiple gold, but 1 gold can be to multiple system (1_g-n_s)
    sys_l = [x[1] for x in mappings]
    gold_l = [x[0] for x in mappings]
    not_mapped_sys = n_system - len(sys_l)
    not_mapped_gold = n_gold - len(gold_l)

    print(
        f"{len(mappings)} mappings made between {n_gold} system and {n_system} gold annotations. "
        f"{not_mapped_sys} system and {not_mapped_gold} gold annotations not mapped."
    )


def get_matcher(strategy):
    """
    Factory method for fetching the matcher func.
    :param strategy:
    :return:
    """
    if strategy == "ere":
        return _match_ere
    elif strategy == "ere_sentivent":
        return _match_ere_sentivent
    else:
        raise ValueError(strategy)


def get_selector(criteria):

    if criteria == "ere":
        return _select_ere
    else:
        raise ValueError(criteria)


def _match_ere_sentivent(gold, system):
    """
    Variant of ERE nugget scoring matching strategy [1] accounting for exact overlap token spans.
    Exactly overlapping span annotations are allowed in SENTiVENT but not in TAC KBP.

    1. "Liu, Z., Mitamura, T., & Hovy, E. (2015).
    Evaluation Algorithms for Event Nugget Detection : A Pilot Study.
    Proceedings of the 3rd Workshop on EVENTS at the NAACL-HLT, 53–57."


    :param gold:
    :param system:
    :return:
    """

    mappings = []

    # score all events with dice coefficient
    for gold_event in gold.events:

        gold_token_ids = gold_event.get_extent_token_ids(
            extent=["discontiguous_triggers"]
        )

        for system_event in system.events:

            system_token_ids = system_event.get_extent_token_ids(
                extent=["discontiguous_triggers"]
            )
            dice_score = dice_coefficient(gold_token_ids, system_token_ids)

            if dice_score > 0:
                mappings.append((gold_event, system_event, dice_score))

    return mappings


def _match_ere(gold, system):
    """
    ERE nugget scoring matching strategy as explained in "Liu, Z., Mitamura, T., & Hovy, E. (2015).
    Evaluation Algorithms for Event Nugget Detection : A Pilot Study.
    Proceedings of the 3rd Workshop on EVENTS at the NAACL-HLT, 53–57."

    Note: Non-general search approach for map candidates
    This approach differs from our approaches because it considers all mention combinations at once in the full document.
    On the other hand, our own custom mapper uses constraint search for each gold event.

    :param gold:
    :param system:
    :return:
    """

    mappings = []
    seen_system = set()

    # score all events with dice coefficient
    for gold_event in gold.events:

        gold_token_ids = gold_event.get_extent_token_ids(
            extent=["discontiguous_triggers"]
        )

        for system_event in system.events:

            system_token_ids = system_event.get_extent_token_ids(
                extent=["discontiguous_triggers"]
            )
            dice_score = dice_coefficient(gold_token_ids, system_token_ids)

            # # for TESTING
            # print(" ".join(str(x) for x in gold_token_ids), "-",
            #       " ".join(str(x) for x in system_token_ids), "-",
            #       " ".join(str(t) for t in gold_token_span), "-",
            #       " ".join(str(t) for t in system_token_span), "|",
            #       dice_score)

            if not system_event in seen_system and dice_score > 0.0:  # disable
                #     mapping = { # for TESTING inspection
                #         "gold": gold_event,
                #         "system": system_event,
                #         "span_dice_score": dice_score,
                #         "gold_text": " ".join(str(t) for t in gold_token_span),
                #         "system_text": " ".join(str(t) for t in system_token_span),
                #         "gold_token_ids": " ".join(str(x) for x in gold_token_ids),
                #         "system_token_ids": " ".join(str(x) for x in system_token_ids),
                #     }
                #
                #     gold_event.ere_mapping.append(mapping)
                #     system_event.ere_mapped_to_gold.append(mapping)

                mappings.append((gold_event, system_event, dice_score))
                seen_system.add(system_event)

    # make sure each system is only mapped once, multiple
    sys_l = [x[1] for x in mappings]
    assert len(set(sys_l)) == len(sys_l)

    return mappings


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


def _argmax_dicescore(ann_score_list):
    """
    Selects all candidate matches with the highest dice span sim score.
    This is needed because same span can have multiple annotations in our implementation.
    This is only the case for Liu et al. 2015 when gold event nuggets are mistakenly split up into multiple other nuggets of same length.
    Default python max selects only the first item with max value, so cannot be used.
    :param ann_score_list: list of tuples (AnnotationObject, dice_similarity_score)
    :return: list of matches with the max sim score value
    """
    max_sim = max(m[2] for m in ann_score_list)
    matches = [x for x in ann_score_list if x[2] == max_sim]
    print(
        f"Selected {len(matches)} from {len(ann_score_list)} based on dice scor {max_sim}"
    )
    return matches


def _select_ere(candidates):
    def select_max(candidates):
        """
        Select candidates with max dice score. Can return multiple with same dice score.
        :param candidates: candidate matches
        :return:
        """

        selected_matches = []
        # if multiple system match on one gold: take the highest dice span overlap, but
        # if they overlap with by same amount keep them both
        for gold_event, matches in groupby(candidates, key=lambda x: x[0]):
            matches = list(matches)
            if len(matches) > 1:
                old_matches = matches
                matches = _argmax_dicescore(matches)
            selected_matches.extend(matches)

        return selected_matches

    selection = select_max(candidates)

    try:
        # check that each system event is only matched once
        if selection:
            c = Counter(m[1].element_id for m in selection)
            # print(c)
            assert c.most_common(1)[0][1] == 1
    except:
        print(c)
        raise ValueError("Something went wrong in selection")

    return selection


class NuggetScorer:
    def score(self, gold_doc, system_doc, score_criteria="ere_all_attributes"):

        scorer, attributes = get_scorer(score_criteria)

        return scorer(gold_doc, system_doc, attributes=attributes)


def get_scorer(criteria):

    if criteria == "all_attributes":
        return (
            _score_ere_nugget,
            ["event_type", "event_subtype", "modality", "polarity_negation", "realis"],
        )
    elif criteria == "ere_liu":
        return _score_ere_nugget, ["event_type", "realis"]
    else:
        raise "ValueError"


def compute_prf(metrics):

    metrics["r"] = metrics["tp"] / metrics["n_gold"]
    metrics["p"] = metrics["tp"] / (metrics["tp"] + metrics["fp"])

    try:
        metrics["f1"] = (2 * metrics["p"] * metrics["r"]) / (
            metrics["p"] + metrics["r"]
        )
    except ZeroDivisionError:
        metrics["f1"] = 0.0

    # group keys by type of metric denoted by _suffix (relaxed, attrib, etc)
    metric_suffixes = [
        re.split(r"(tp|fp)_", n)[-1] for n in metrics.keys() if "tp_" in n or "fp_" in n
    ]

    # compute alternative span precision P as proposed in Liu et al. 2015
    metrics["p_alt"] = (
        metrics["tp"] / metrics["n_system"] if metrics["n_system"] else 0.0
    )

    try:
        metrics["f1_alt"] = (2 * metrics["p_alt"] * metrics["r"]) / (
            metrics["p_alt"] + metrics["r"]
        )
    except ZeroDivisionError:
        metrics["f1_alt"] = 0.0

    # make alternative precision as in paper: tp / n_sys
    for m in metric_suffixes:
        tp = metrics[f"tp_{m}"]
        fp = metrics["fp"]

        # compute precision, recall and alternative precision
        metrics[f"p_{m}"] = tp / (tp + fp) if tp or fp else 0.0
        metrics[f"r_{m}"] = tp / metrics["n_gold"] if metrics["n_gold"] else 0.0
        metrics[f"p_{m}_alt"] = tp / metrics["n_system"] if metrics["n_system"] else 0.0

        try:
            metrics[f"f1_{m}"] = (2 * metrics[f"p_{m}"] * metrics[f"r_{m}"]) / (
                metrics[f"p_{m}"] + metrics[f"r_{m}"]
            )
        except ZeroDivisionError:
            metrics[f"f1_{m}"] = 0.0

        try:
            metrics[f"f1_{m}_alt"] = (2 * metrics[f"p_{m}_alt"] * metrics[f"r_{m}"]) / (
                metrics[f"p_{m}_alt"] + metrics[f"r_{m}"]
            )
        except ZeroDivisionError:
            metrics[f"f1_{m}_alt"] = 0.0

    return metrics


def compute_acc(scores):
    """
    Compute attribute accuracy based on a metrics dict with "_acc" keys and the total amount of system docs.
    In accordance
    :param scores:
    :return:
    """

    for k, v in scores.items():
        if k.startswith("acc_"):
            acc = v / scores["n_system"] if scores["n_system"] else 0.0
            scores[k] = acc

    return scores


def check_matched(ann):
    """
    Check if annotation is matched.
    :param ann:
    Returns True if ok else raises ValueError
    """
    if any("mapping" in attr_n for attr_n in dir(ann)) or any(
        "match" in attr_n for attr_n in dir(ann)
    ):
        return True
    else:
        raise ValueError("Gold-system matches need to be made before scoring.")


def is_attribute_match(gold, match, attribute_name):

    if attribute_name == "event_subtype":
        # need to specify maintype because some same subtype stem on different maintypes
        gold_val = (getattr(gold, "event_type"), getattr(gold, "event_subtype"))
        match_val = (getattr(match, "event_type"), getattr(match, "event_subtype"))

    else:
        gold_val = getattr(gold, attribute_name)
        match_val = getattr(match, attribute_name)

    if gold_val == match_val:
        return True
    else:
        return False


def _score_ere_nugget(
    gold_doc, system_doc, attributes=["event_type", "polarity_negation", "modality"]
):
    """
    ERE nugget scoring strategy as explained in [1].

    N.B: selecting match with maximum dice score is already done by Matcher
    (i.e. argmax in Algorithm 3 and 4 in Liu et al. 2015).
    Multiple matches are off the case described in [1] pp. where one gold annotation is erroneously split
    into multiple annotations.

    Computes all variants mentioned in the above paper:
    - f1: span (dice score) f1 of event triggers
    - f1_alt: Uses alternative Precision ("p_alt") as [1] pp.55 sec.2.4
    - f1_attrib_alt:


    :param gold_doc:
    :param system_doc:
    :param attributes:
    :return: dict of all metrics and scores
    """

    check_matched(
        gold_doc.events[-1]
    )  # sanity check if gold and system matching step has completed

    metrics = {
        "tp": 0.0,  # default span tp where of one match with argmax dice score is selected
        # when there are multiple system candidate matches for a gold
        "fp": 0.0,  # default span fp
        # alternative scoring that differs from "Evaluation algo. for Event Nugget Detection,  Liu et al. 2015"
        "tp_relaxed": 0.0,  # if token span overlaps count full score, instead of dice coeff score
        # attribute matched metrics which are count when all attributes match
        "fp_attrib": 0.0,  # fp when all attributes do not match (very strict)
        "tp_attrib": 0.0,  # only count tp if all attributes are accurate and add dice coeff score (very strict)
        "tp_attrib_relaxed": 0.0,  # as above, but +1, instead of dice coeff score
        # counts
        "n_gold": len(gold_doc.events),  # total number of gold annotations
        "n_system": len(system_doc.events),  # total number of reference annotations
        "n_match": 0.0,  # total number of gold to potentially multiple system (1-n) mappings for attribute accuracy
        "acc_allattrib": 0.0,
    }

    # add attribute accuracy, fp, tp keys
    metrics.update({"acc_" + attrib_n: 0.0 for attrib_n in attributes})
    metrics.update(  # tp for individual attributes
        {"tp_" + attrib_n: 0.0 for attrib_n in attributes}
    )
    metrics.update(  # fp for individual attributes
        {"fp_" + attrib_n: 0.0 for attrib_n in attributes}
    )

    # check if matches on gold
    for gold_ev in gold_doc.events:

        if gold_ev.ere_mapping:
            # [1] Algorithm 2 pp. 55: Compute span F1
            sys_ev_max, max_dice_score = max(gold_ev.ere_mapping, key=lambda x: x[1])
            metrics["tp"] += max_dice_score
            metrics[
                "tp_relaxed"
            ] += 1.0  # different from [1]: count every span overlap as tp: no penalty for long annotations.

            # [1] Algorithm 4 pp 55: Compute True Positive with all attributes
            if all(
                is_attribute_match(gold_ev, sys_ev_max, attrib_n)
                for attrib_n in attributes
            ):  # [1] Algorithm 4 pp 55 line 3: Check if all attributes match for the max hit.
                metrics["tp_attrib"] += max_dice_score  # increment dice score
                metrics["tp_attrib_relaxed"] += 1.0  # deviation from [1] own relaxation
            else:
                metrics["fp_attrib"] += 1.0

            for (sys_ev, dice_score) in gold_ev.ere_mapping:
                # [1] Algorithm 3 line 3: extract attributes and check accuracy individually
                attrib_match = []
                for attrib_n in attributes:  # check if each attrib matches
                    attribute_is_accurate = is_attribute_match(
                        gold_ev, sys_ev, attrib_n
                    )
                    attrib_match.append(attribute_is_accurate)
                    if attribute_is_accurate:  # add attributes to score dict
                        metrics["acc_" + attrib_n] += 1.0 / len(
                            gold_ev.ere_mapping
                        )  #  [1] Algorithm 3 line 3: increment acc score for INDIVIDUAL attributes. Different from Algorithm 3 which considers all at once.
                        metrics["tp_" + attrib_n] += dice_score  # algorithm 4
                    else:
                        metrics["fp_" + attrib_n] += 1.0
                # Attribute TP: all attributes match, add tp_attrib, tp_attrib_relaxed
                if all(
                    attrib_match
                ):  # [1] Algorithm 3 line 3: check if all attributes match
                    metrics["acc_allattrib"] += 1.0 / len(
                        gold_ev.ere_mapping
                    )  #  [1] Algorithm 3 line 3: increment acc score
        else:
            metrics["fp"] += 1.0
            metrics["fp_attrib"] += 1.0
            for attrib_n in attributes:
                metrics["fp_" + attrib_n] += 1.0

    # compute P, R and F1 and accs
    metrics = compute_prf(metrics)
    metrics = compute_acc(metrics)

    return metrics


def find_exact_overlap(events):
    """
    In a list of annotations return the ones that exactly overlap and have the exact same boundaries e.g.:
    "The [[report]] about oil prices comes in at."
    :param events: list of event annotation objects.
    :return: list of lists of exactly overlapping events.
    """
    token_span_ids = [
        tuple(ev.get_extent_token_ids(extent=["discontiguous_triggers"]))
        for ev in events
    ]
    # use counter to find overlapping spans: if token position ids are same, there is exact event trigger overlap.
    counter = Counter(token_span_ids)
    overlap_token_span = [k for k, v in counter.items() if v > 1]
    if overlap_token_span:
        overlap_events = [
            [
                ev
                for ev in events
                if tuple(ev.get_extent_token_ids(extent=["discontiguous_triggers"]))
                == d
            ]
            for d in overlap_token_span
        ]
        return overlap_events


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


if __name__ == "__main__":

    # load the final IAA study project and set gold standard
    iaa_proj = parse_project(settings.IAA_XMI_DIRP)

    # load the final clean project
    gold_proj = parse_project(settings.CLEAN_XMI_DIRP)
    gold_docs = [
        doc
        for doc in gold_proj.annotation_documents
        if doc.title in settings.IA_IDS and doc.annotator_id == settings.MOD_ID
    ]

    # get some stats on overlap annotations
    overlap_stats_iaa = compute_overlap_stats(iaa_proj)
    overlap_stats_gold = compute_overlap_stats(gold_proj)

    # set the matching, selection, and scoring criteria
    scorer_settings = {
        "match_strategy": "ere",
        "select_criteria": "ere",
        "score_criteria": "all_attributes",
    }

    # use moderator_id to determine gold file and system files
    system_docs = list(
        filter(
            lambda d: d.annotator_id != settings.MOD_ID, iaa_proj.annotation_documents
        )
    )

    system_docs.sort(key=lambda x: x.annotator_id)
    system_docs_dict = {
        anno_id: list(sys_docs)
        for anno_id, sys_docs in groupby(system_docs, key=lambda x: x.annotator_id)
    }

    mapper = MentionMapper()
    scorer = NuggetScorer()

    records = []

    for gold_doc in gold_docs:
        for system_anno_id, system_docs in system_docs_dict.items():
            system_doc = next(
                sys_doc for sys_doc in system_docs if sys_doc.title == gold_doc.title
            )

            print(f"Starting mention mapping for {system_anno_id} {system_doc.title}.")
            gold_matched = mapper.match(
                gold_doc,
                system_doc,
                match_strategy=scorer_settings["match_strategy"],
                select_criteria=scorer_settings["select_criteria"],
            )

            # preprocess for multiple gold annotations on same string

            ## find gold_anns on same exact token span (for gold and system)

            ## join them into one annotation with attribs

            ## TODO check how this can be compared

            doc_scores = scorer.score(
                gold_matched,
                system_doc,
                score_criteria=scorer_settings["score_criteria"],
            )

            record = {"doc_id": system_doc.title, "anno_id": system_doc.annotator_id}
            record.update(doc_scores)
            records.append(record)

    # remove non-annotated doc of anno_03, bac_05
    remove = [
        {
            "anno_id": "anno_03",
            "doc_id": "bac05_bofa-includes-bitcoin-trust-in-broader-ban-on-investments.txt",
        },
        {
            "anno_id": "anno_01",
            "doc_id": "duk06_like-many-of-its-peers-duk-is-trading-in-the-oversold-zone.txt",
        },
    ]

    records_filt = [
        r for r in records if not any(rm.items() <= r.items() for rm in remove)
    ]

    # MAKE PANDAS DATAFRAME AND EXPORT TO EXCEL
    all_score_df = pd.DataFrame.from_records(records_filt)
    # round to 4 decimals
    all_score_df = all_score_df.round(decimals=4)

    # mean over column to get final scores and sort
    all_mean_df = all_score_df.mean(axis=0)
    anno_mean_df = all_score_df.groupby(
        ["anno_id"]
    ).mean()  # scores by annotator, transpose for readab.
    docs_mean_df = all_score_df.groupby(
        ["doc_id"]
    ).mean()  # scores by doc, transpose for readab.

    # Write scores to Excel file
    xl_fn = (
        "nuggetscores-"
        + "-".join(f"{k}={v}" for k, v in scorer_settings.items())
        + "_test3.xlsx"
    )
    xl_fp = Path(settings.OPT_DIRP, xl_fn)
    xl_writer = pd.ExcelWriter(xl_fp, engine="xlsxwriter")
    all_mean_df.to_excel(
        xl_writer, sheet_name="All mean scores"
    )  # Write each dataframe to a different worksheet.
    anno_mean_df.to_excel(xl_writer, sheet_name="Annotator mean scorers")
    docs_mean_df.to_excel(xl_writer, sheet_name="Document mean scores")
    # column names to sort columns in Excel
    important_cols = [
        "doc_id",
        "anno_id",
        "f1_alt",
        "f1_relaxed_alt",
        "f1_attrib_alt",
        "f1_attrib_relaxed_alt",
    ]
    cols_sorted = important_cols + [
        cn for cn in all_score_df.columns.to_list() if cn not in important_cols
    ]
    # write the excel
    all_score_df = all_score_df.reindex(columns=cols_sorted)
    all_score_df.to_excel(xl_writer, index=False, sheet_name="All scores")
    xl_writer.save()  # Close the Pandas Excel writer and output the Excel file
