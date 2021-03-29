#!/usr/bin/env python3
"""
Parse and pre-process a project export to the GTS jsonl format for ABSA pair and triplet extraction.

[{
	"id": "id of sentence 1",
	"sentence": "context of sentence 1",
	"triples": [{
		"uid": "id of first aspect term of sentence 1",
		"target_tags": "the first aspect term with BIO scheme",
		"opinion_tags": "the corresponding multiple opinion terms of the aspect term with BIO scheme",
		"sentiment": "the corresponding sentiment polarity of the aspect term"
	},{
        the second aspect term of sentence 1
        ...
    }]
}, {
	sentence 2
    ...
}]

parse_to_gts.py
webannoparser
24/01/21
Copyright (c) Gilles Jacobs. All rights reserved.
"""
import json
import shutil
import random
import itertools
from pathlib import Path
from collections import Counter, deque
from itertools import islice
from parse_project import parse_process_project
from parser import Filler, Participant, Event
from collections import OrderedDict
import settings
import argparse

random.seed(42)

def split_train_dev_test(dataset, split_doc_ids):

    split_dataset = {}

    for inst in dataset:
        for splitname, docids in split_doc_ids.items():
            if inst["id"].split(":")[0] in docids:
                split_dataset.setdefault(splitname, []).append(inst)

    return split_dataset

def basic_stats(data):

    # count the B tags in triples to get opinion term count
    opi_tags = [[t.split("\\")[-1] for t in inst["opinion_tags"].split(" ")] for sen in data for inst in sen["triples"]]
    opi_c = sum(Counter(tags)["B"] for tags in opi_tags)

    counts = {
        "sentences_n": len(data),
        "target_n": sum(len(inst["triples"]) for inst in data),
        "opinion_n": opi_c,
    }
    # count polarities
    pol_c = Counter(triple["sentiment"] for inst in data for triple in inst["triples"])
    for k, v in pol_c.items():
        counts[f"{k}_n"] = v
        counts[f"{k}_pct"] = round(v / sum(pol_c.values()) * 100.0, 1)

    return counts

def remove_sentiment_expression(project, sentiment_expression):
    '''
    Utility to remove in-place on project a sentiment expression on doc and sentence level.
    :param project:
    :param sentiment_expression:
    :return:
    '''

    for doc in project.annotation_documents:
        try:
            doc.sentiment_expressions.remove(sentiment_expression)
        except ValueError as e:
            pass
        try:
            doc.events.remove(sentiment_expression)
        except ValueError as e:
            pass
        for sen in doc.sentences:
            try:
                sen.sentiment_expressions.remove(sentiment_expression)
            except ValueError as e:
                pass
            try:
                sen.events.remove(sentiment_expression)
            except ValueError as e:
                pass


def remove_cross_sentence(project, unit_iterator):

    sentiment_expressions = list(unit_iterator)
    sentiment_expressions_preproc = []
    sentiment_expression_total = len(sentiment_expressions)
    total_target_count = 0

    for se in sentiment_expressions:
        sentence_id_se = se.in_sentence[0].element_id
        for tgt in se.targets[:]:
            total_target_count += 1
            sentence_id_tgt = tgt.in_sentence[0].element_id
            max_sentence_window = 0 # 0 = same sentence, 1 is previous/next sentence, n-sentence window
            if abs(sentence_id_tgt - sentence_id_se) > max_sentence_window:
                print(f"Cross-sentence target removed out of sentence window (={max_sentence_window}):\n\t{se}")
                se.targets.remove(tgt)
                statistics["target_cross_sentence_removed"] += 1
        if se.targets: # sentiment expression still has targets and thus are valid
            sentiment_expressions_preproc.append(se)
        else: # no targets left because they were all removed in previous step, do not add to output list
            statistics["sentiment_expression_cross_sentence_removed"] += 1
            print(f"Sentiment expression removed, all targets out of sentence window (={max_sentence_window}):\n\t{se}")
            remove_sentiment_expression(project, se)
    print(f"{statistics['target_cross_sentence_removed']}/{total_target_count} ({round(100*statistics['target_cross_sentence_removed']/total_target_count,1)}%) targets removed because cross sentence.")
    print(f"{statistics['sentiment_expression_cross_sentence_removed']}/{sentiment_expression_total} ({round(100*statistics['sentiment_expression_cross_sentence_removed']/sentiment_expression_total,1)}%) sentiment expressions removed because cross sentence.")

    return sentiment_expressions_preproc


def replace_canonical_referents_sentiment(project, remove_cross_sentence=False):
    '''

    NB: this is reimplementation of project.replace_canonical_referents func which was buggy and too complicated.
    :param sentiment_expressions:
    :param remove_cross_sentence:
    :return: list of sentiment_expressions_preproc with new target_preproc attrib.
    '''

    sentiment_expressions_preproc = []

    for se in project.get_sentiment_expressions():
        targets_preproc = []
        for tgt in se.targets:
            tgt_is_arg_and_has_refs = (type(tgt).__name__ == "Participant" or type(tgt).__name__ == "Filler") and \
                (not tgt.canonical_referents in [[],None] and tgt.canonical_referents != "from_canonref")
                # sometimes there are multiple canonrefs tagged
                # this can be a) annotation mistake or
                # b) multiple reference to a group, e.g. "all" refers to three companies
            tgt_is_event_and_has_refs = (type(tgt).__name__ == "Event" and not tgt.coreferents in [[],None]) # collect coreferents for event too
            if tgt_is_arg_and_has_refs or tgt_is_event_and_has_refs:
                refs_attr = "canonical_referents" if tgt_is_arg_and_has_refs else "coreferents"
                for ref in getattr(tgt, refs_attr):
                    # check whether canonical referent is in same sentence
                    same_sentence = [s.element_id for s in tgt.in_sentence] == [
                        s.element_id for s in ref.in_sentence
                    ]
                    # if skip_cross_sentence is True only replace when canonref is in same sentence
                    if remove_cross_sentence and same_sentence:
                        targets_preproc.append(ref)
                        print(
                            f"Replaced {tgt.text} with {ref.text} in same sentence:\n\t{se}"
                        )
                        statistics["target_replaced_ref"] += 1
                    elif not remove_cross_sentence: # if skip_cross_sentence is False always replace
                        targets_preproc.append(ref)
                        print(
                            f"Replaced {tgt.txt} with {ref.txt}:\n\t{se}."
                        )
                        statistics["target_replaced_ref"] += 1
            else: # no canon/co-refs so keep it in
                targets_preproc.append(tgt)
        se.targets = targets_preproc
        sentiment_expressions_preproc.append(se)
    return sentiment_expressions_preproc


def remove_pronominal(project, unit_iterator):

    sentiment_expressions = list(unit_iterator)
    sentiment_expressions_preproc = []
    total_target_count = 0

    for se in sentiment_expressions:
        # NOT NEEDED THERE ARE NONE: check if sentiment expression itself is pronominal
        # if se.check_pronominal():
        #     print(f"Pronominal sentiment expression removed:\n\t{se}")
        #     statistics["sentiment_expression_pronominal_removed"] += 1
        #     continue
        # check targets
        for tgt in se.targets[:]:
            total_target_count += 1
            if tgt.check_pronominal():
                print(f"Pronominal target removed:\n\t{se}")
                se.targets.remove(tgt)
                statistics["target_pronominal_removed"] += 1
        if se.targets:
            sentiment_expressions_preproc.append(se)
        else:
            print(f"Sentiment expression removed because all targets pronominal:\n\t{se}")
            statistics["sentiment_expression_pronominal_removed"] += 1
            remove_sentiment_expression(project, se)

    print(f"{statistics['target_pronominal_removed']}/{total_target_count} ({round(100*statistics['target_pronominal_removed']/total_target_count,1)}%) targets removed because pronominal.")
    print(f"{statistics['sentiment_expression_pronominal_removed']}/{len(sentiment_expressions)} ({round(100*statistics['sentiment_expression_pronominal_removed']/len(sentiment_expressions),1)}%) sentiment expressions removed because it has all pronominal targets.")

    return sentiment_expressions_preproc

def filter_target_roles_events(project):
    '''
    Reads the json of {eventtype.subtype: [valid_target_role1, valid_target_role2]}
    to remove roles that cannot be a sentiment target entity from events.
    This removes non-entity arguments such as TIME, PLACE, etc and ensures a valid mapping from event annotation to
    :param project: populate valid targets in-lpace on events.
    :return: new events for
    '''

    with open(cmd_args.target_role, "rt") as target_role_in:
        target_role_map = json.load(target_role_in)

    # # debug sanity check
    valid_targets = {et_arg["name"] for et in target_role_map for et_arg in et["participant"] + et["filler"] if et_arg["is_target"]}
    # invalid_targets = {et_arg["name"] for et in target_role_map for et_arg in et["participant"] + et["filler"] if not et_arg["is_target"]}

    print(f"Setting valid roles for targets: {valid_targets}")

    for ev in project.get_events():
        if ev.event_type is None:
            print(f"No type label on event {ev.friendly_id()}")
            continue
        type_fn = ev.event_type
        if ev.event_subtype:
            type_fn += f"_{ev.event_subtype.split('_')[0]}"
        arguments = set(itertools.chain.from_iterable((ev.participants or [], ev.fillers or [])))
        target_role_map_event_match = [et for et in target_role_map if et["full_name"] == type_fn]
        if not target_role_map_event_match:
            print(f"No match for {type_fn.upper()} {ev.friendly_id()} ")
        valid_roles = set()
        for et in target_role_map_event_match:
            for et_arg in et["participant"] + et["filler"]:
                if et_arg["is_target"]:
                    valid_roles.add(et_arg["name"])

        for arg in arguments:
            if arg.role in valid_roles:
                ev.targets.append(arg)
            # else:
            #     print(f"{arg.role.upper()} not a valid target in {valid_roles}")
            #     pass

def replace_canonical_referents(unit_iterator, remove_cross_sentence=False):

    sentiment_expressions_preproc = []

    for se in unit_iterator:
        targets_preproc = []
        for tgt in se.targets:
            tgt_is_arg_and_has_refs = (type(tgt).__name__ == "Participant" or type(tgt).__name__ == "Filler") and \
                (not tgt.canonical_referents in [[],None] and tgt.canonical_referents != "from_canonref")
                # sometimes there are multiple canonrefs tagged
                # this can be a) annotation mistake or
                # b) multiple reference to a group, e.g. "all" refers to three companies
            tgt_is_event_and_has_refs = (type(tgt).__name__ == "Event" and not tgt.coreferents in [[],None]) # collect coreferents for event too
            if tgt_is_arg_and_has_refs or tgt_is_event_and_has_refs:
                refs_attr = "canonical_referents" if tgt_is_arg_and_has_refs else "coreferents"
                for ref in getattr(tgt, refs_attr):
                    # check whether canonical referent is in same sentence
                    same_sentence = [s.element_id for s in tgt.in_sentence] == [
                        s.element_id for s in ref.in_sentence
                    ]
                    # if skip_cross_sentence is True only replace when canonref is in same sentence
                    if remove_cross_sentence and same_sentence:
                        targets_preproc.append(ref)
                        print(
                            f"Replaced {tgt.text} with {ref.text} in same sentence:\n\t{se}"
                        )
                        statistics["target_replaced_ref"] += 1
                    elif not remove_cross_sentence: # if skip_cross_sentence is False always replace
                        targets_preproc.append(ref)
                        print(
                            f"Replaced {tgt.txt} with {ref.txt}:\n\t{se}."
                        )
                        statistics["target_replaced_ref"] += 1
            else: # no canon/co-refs so keep it in
                targets_preproc.append(tgt)
        se.targets = targets_preproc
        sentiment_expressions_preproc.append(se)
    return sentiment_expressions_preproc

def preprocess_events(project):
    '''
    Preprocess events for mapping 2 sentiment - target annotations.
    Removes in-place sentiment expressions from project.annotation_documents[i].sentiment_expressions and
     project.annotation_documents[i].sentences[j].sentiment_expressions.
    Removes in-place targets from SE.targets.
    :param project:
    :return:
    '''
    # focus first on sentiment annotations
    ev_orig_n = sum(1 for se in project.get_events())
    tgt_orig_n = sum(len(list(itertools.chain.from_iterable((ev.participants or [], ev.fillers or [])))) for ev in project.get_events())

    # set target accumulator on events
    for ev in project.get_events():
        ev.preprocess_trigger(fix_false_discont=True,
                              make_continuous_max_dist=6,
                              truncate_to_len=False,)
        ev.targets = []

    if cmd_args.target_role:
        filter_target_roles_events(project)
    
    # make cross-sentence annotations in-sentence and count stats of cut relations
    if cmd_args.replace_canonical_referents:
        evs = replace_canonical_referents(project.get_events(), remove_cross_sentence=cmd_args.remove_cross_sentence)
    if cmd_args.remove_cross_sentence:
        evs = remove_cross_sentence(project, project.get_events())
    if cmd_args.remove_pronominal:
        evs = remove_pronominal(project, project.get_events())

    tgt_removed_n = statistics["target_cross_sentence_removed"] + statistics["target_pronominal_removed"]
    tgt_removed_pct = round(100*tgt_removed_n/tgt_orig_n, 1)
    ev_removed_n = ev_orig_n-len(evs)
    ev_removed_pct = round(100*ev_removed_n/ev_orig_n, 1)
    print(f"{tgt_removed_n}/{tgt_orig_n} ({tgt_removed_pct}%) targets removed in preprocessing.")
    print(f"{ev_removed_n}/{ev_orig_n} ({ev_removed_pct}%) sentiment expressions removed in preprocessing.")

    return evs

def preprocess_sentiment(project):
    '''
    Preprocess sentiment expression - target annotations.
    Removes in-place sentiment expressions from project.annotation_documents[i].sentiment_expressions and
     project.annotation_documents[i].sentences[j].sentiment_expressions.
    Removes in-place targets from SE.targets.
    :param project:
    :return:
    '''
    # focus first on sentiment annotations
    ses_orig_n = sum(1 for se in project.get_sentiment_expressions())
    tgt_orig_n = sum(len(se.targets) for se in project.get_sentiment_expressions())

    # make cross-sentence annotations in-sentence and count stats of cut relations
    if cmd_args.replace_canonical_referents:
        ses = replace_canonical_referents(project.get_sentiment_expressions(), remove_cross_sentence=cmd_args.remove_cross_sentence)
    if cmd_args.remove_cross_sentence:
        ses = remove_cross_sentence(project, project.get_sentiment_expressions())
    if cmd_args.remove_pronominal:
        ses = remove_pronominal(project, project.get_sentiment_expressions())

    tgt_removed_n = statistics["target_cross_sentence_removed"] + statistics["target_pronominal_removed"]
    tgt_removed_pct = round(100*tgt_removed_n/tgt_orig_n, 1)
    se_removed_n = ses_orig_n-len(ses)
    se_removed_pct = round(100*se_removed_n/ses_orig_n, 1)
    print(f"{tgt_removed_n}/{tgt_orig_n} ({tgt_removed_pct}%) targets removed in preprocessing.")
    print(f"{se_removed_n}/{ses_orig_n} ({se_removed_pct}%) sentiment expressions removed in preprocessing.")


def get_bio_tags(context_tokens, annotation_tokens):

    token_bio_seq = ["O"] * len(context_tokens)
    for unit in annotation_tokens:
        unit_tags = ["B"] + ["I"]*(len(unit)-1)
        for i, token in enumerate(unit):
            token_bio_seq[token.index_sentence] = unit_tags[i]

    return token_bio_seq

def get_bio_text(context_tokens, annotation_tokens):
    '''

    :param context_tokens: List of tokens of wider context (document- or sentence-level).
    :param annotation_tokens: Nested list of tokens corresponding to the annotations units for whic hto generate BIO labels.
    :return:
    '''
    token_bio_seq = get_bio_tags(context_tokens, annotation_tokens)
    token_bio_text = " ".join(f"{tok.text}\\{tag}" for tok, tag in zip(context_tokens, token_bio_seq))
    return token_bio_text

def get_opinion_tokens(sentiment_expressions):

    opinion_tokens = []
    for se in sentiment_expressions:
        if "SentimentExpression" == type(se).__name__:
            # sometimes units are parsed with duplicate and out-of-order tokens, so dedupe and sort the tokens.
            se_tokens = sorted(list(set(se.tokens)), key=lambda x: x.begin)
        elif "Event" == type(se).__name__:
            # TODO implement contentful token selection for discontinuous triggers, right now use same as SE above
            se_tokens = sorted(list(set(se.tokens)), key=lambda x: x.begin)
        opinion_tokens.append(se_tokens)

    opinion_tokens = sorted(opinion_tokens, key=lambda x: x[0].begin) # sort the units by begin token

    return opinion_tokens

def parse_to_gts(project):

    data = []
    # sentence level
    for doc in project.annotation_documents:
        for i, sen in enumerate(doc.sentences):

            sen_id = f"{doc.document_id}:{i}"

            triples = []
            # make a dict from one target->many sentiment expression OF SAME POLARITY in sentence
            target_opinions = {}
            for u in sen.sentiment_expressions + sen.events:
                polarity = u.polarity_sentiment
                for tgt in u.targets:
                    target_opinions.setdefault((tgt, polarity), set()).add(u)
            for j, ((tgt, polarity), ses) in enumerate(target_opinions.items()):
                # we need to get the opinion SE term that links to the aspect term > do with dict {AT: [OT1, OT2]} above
                target_tags = get_bio_text(sen.tokens, [sorted(list(set(tgt.tokens)), key=lambda x: x.begin)])
                opinion_tokens = get_opinion_tokens(ses)
                opinion_tags = get_bio_text(sen.tokens, opinion_tokens)

                triplet_json = OrderedDict([
                    ("uid", f"{sen_id}-{j}"),
                    ("target_tags", target_tags),
                    ("opinion_tags", opinion_tags),
                    ("sentiment", polarity),
                ])
                triples.append(triplet_json)

            inst_json = {
                "id": f"{sen_id}",
                "sentence": " ".join(t.text for t in sen.tokens),
                "triples": triples,
            }
            data.append(inst_json)

    return data


def write_dataset(dataset_split, opt_dir):
    opt_dir = Path(opt_dir)
    opt_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in dataset_split.items():
        fp = opt_dir / f"{split_name}.json"
        with open(fp, "wt") as split_out:
            json.dump(split_data, split_out, indent=2)
            print(f"Wrote {split_name} GTS data to {fp}.")

def remove_empty(dataset):

    return [inst for inst in dataset if inst["triples"]]


def main():
    global cmd_args, statistics
    # arguments
    parser = argparse.ArgumentParser(
        description="Preprocess SENTiVENT WebAnno event data."
    )
    parser.add_argument(
        "input_dir", help="Name for input unzipped WebAnno XMI export directory."
    )
    parser.add_argument("output_dir", help="Name for output directory.")
    parser.add_argument(
        "--annotations",
        default="sentiment",
        const="sentiment",
        nargs="?",
        choices=["none", "sentiment", "event2sentiment", "sentiment+event2sentiment"],
        help="Parse no annotations (only text) or sentiment expressions only (sentiment (default)). \
             Map event annotations to sentiment anotations with 'event2sentiment'. \
             Combine both with sentiment+event2sentiment.",
    )
    parser.add_argument(
        "--target_role",
        nargs="?",
        help="JSON file containing argument roles to exclude from event to sentiment transformations.",
    )
    parser.add_argument(
        "--span_width",
        default=0,
        nargs='?',
        type=int,
        help="Max. span width. Truncate other spans (args, targets, sents) annotations to this width.",
    )
    parser.add_argument('--replace_canonical_referents',
                        action='store_true',
                        help="Replace argument and target annotations with their linked CanonicalReferent. \
                             Cross-sentence CanonicalReferent links are skipped by default if --allow_cross_sentence is not used."
                        )
    parser.add_argument('--skip_cross_sentence',
                    action='store_true',
                    help="Remove cross-sentence relations (sentiment-target (enabled by default)."
                    )
    parser.add_argument('--remove_pronominal',
                    action='store_true',
                    help="Skip pronominal annotations such as events, arguments and sentiment targets that are the result of coreference. \
                         If 'replace_canonical_referents' is enabled, targets will be replaced with CanonicalReferent annotation first, \
                         if no CanonRef is available pronominal annotations will be removed."
                    )
    cmd_args = parser.parse_args()

    # init stats
    statistics = {
        "target_replaced_ref": 0,
        "target_cross_sentence_removed": 0,
        "sentiment_expression_cross_sentence_removed": 0,
        "target_pronominal_removed": 0,
        "sentiment_expression_pronominal_removed": 0,
    }

    # parse from raw Webanno
    # data_dir = cmd_args.input_dir
    data_dir = settings.MASTER_DIRP # set with settings.py for experiments because of documentation
    print(f"Parsing raw WebAnno data in {data_dir}")
    project = parse_process_project(data_dir, from_scratch=False)

    # preprocess events
    if "event" in cmd_args.annotations:
        print("Preprocessing event annotations and map to sentiment-target.")
        preprocess_events(project)

    # preprocess annotations
    if "sentiment" in cmd_args.annotations:
        print("Preprocessing sentiment annotations.")
        preprocess_sentiment(project)

    # parse to GTS format
    print(f"Converting to GTS jsonl format.")
    data_json = parse_to_gts(project)

    # split data
    print(f"Creating data splits.")
    splits = settings.SPLITS_DOC_EXPERIMENTS # TODO add argument parse option
    dataset_splits = split_train_dev_test(data_json, splits)

    # print some basic stats
    for split_n, split_data in dataset_splits.items():
        print(f"{split_n}: {basic_stats(split_data)}")

    # write jsonl
    # opt_dir = Path("/home/gilles/repos/dygiepp/data/ace-event/processed-data/sentivent/json")
    write_dataset(dataset_splits, Path(cmd_args.output_dir))

    # remove negative sentence instances without aspect term.
    dataset_splits_no_empty = {split_n: remove_empty(data) for split_n, data in dataset_splits.items()}
    for split_n, split_data in dataset_splits_no_empty.items():
        print(f"{split_n}: {basic_stats(split_data)}")
    write_dataset(dataset_splits_no_empty, f"{cmd_args.output_dir}-no-empty/")

    # # backup role map to opt
    # if cmd_args.role_map:
    #     shutil.copy(cmd_args.role_map, str(opt_dir / "role-map.json"))


if __name__ == "__main__":
    main()
