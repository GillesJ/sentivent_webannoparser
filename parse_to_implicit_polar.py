#!/usr/bin/env python3
"""
Parse and pre-process a project export to the GTS jsonl format for ABSA pair and triplet extraction.

Preprocessing steps applied for implicit sentiment pilot:
- Make discontinuous parts continuous if they add less than 24 tokens to the core annotations.
- Add in annotated modifiers (diminishers, negation, intensifiers, specifiers, etc.) to the polar expression span if they add less than 24 tokens.
Polarity shifting by these modifiers was taken into account when annotating polarity (e.g. negation changes pos to neg) so they should be added to the span.
- For experiments where targets are included: Coreference annotations are linked for targets. Cross-sentence targets are removed.
Discontinuous target entity spans have been made continuous if they add less than 24 tokens to the core annotation.

parse_to_implicit_polar.py
webannoparser
09/03/21
Copyright (c) Gilles Jacobs. All rights reserved.
"""
import json
import shutil
import random
import itertools
from pathlib import Path
import pandas as pd
from collections import Counter, deque
from itertools import islice
from parse_project import parse_process_project
from parser import Filler, Participant, Event
import settings
import argparse

random.seed(42)

def _dist(s1, s2):
    # sort the two ranges such that the range with smaller first element
    # is assigned to x and the bigger one is assigned to y
    x, y = sorted((s1, s2))

     #now if x[1] lies between x[0] and y[0](x[1] != y[0] but can be equal to x[0])
     #then the ranges are not overlapping and return the difference of y[0] and x[1]
     #otherwise return 0
    if x[0] <= x[1] < y[0] and all( y[0] <= y[1] for y in (s1,s2)):
        return y[0] - x[1]
    return 0

def split_train_dev_test(dataset, split_doc_ids):

    split_dataset = {}

    for inst in dataset:
        for splitname, docids in split_doc_ids.items():
            if inst["id"].split(":")[0] in docids:
                split_dataset.setdefault(splitname, []).append(inst)

    return split_dataset

def remove_cross_sentence_targets(unit_iterator):

    sentiment_expressions = list(unit_iterator)
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
        se.targets = list(set(targets_preproc))
        sentiment_expressions_preproc.append(se)
    return sentiment_expressions_preproc

def remove_pronominal_targets(unit_iterator):

    sentiment_expressions = list(unit_iterator)

    for se in sentiment_expressions:
        targets_preproc = []
        for tgt in se.targets:
            if tgt.check_pronominal():
                print(f"Removed pronominal target \"{tgt.text}\" on \"{se.friendly_id()}\".")
            else:
                targets_preproc.append(tgt)
        se.targets = targets_preproc


def filter_target_roles_events(project):
    '''
    Reads the json of {eventtype.subtype: [valid_target_role1, valid_target_role2]}
    to remove roles that cannot be a sentiment target entity from events.
    This removes non-entity arguments such as TIME, PLACE, etc and ensures a valid mapping from event annotation to
    :param project: populate valid targets in-lpace on events.
    :return: new events for
    '''

    with open('/home/gilles/repos/sentivent_webannoparser/sentivent_en_event_typology.json', "rt") as target_role_in:
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
            print(f"No match for {type_fn.upper()} {ev.friendly_id()}.")
        valid_roles = set()
        for et in target_role_map_event_match:
            for et_arg in et["participant"] + et["filler"]:
                if et_arg["is_target"]:
                    valid_roles.add(et_arg["name"])

        for arg in arguments:
            if arg.role in valid_roles:
                ev.targets.append(arg)
            else:
                print(f"{arg.role.upper()} not a valid target in {valid_roles}")

def replace_canonical_referents(unit_iterator, skip_cross_sentence=False, max_extended=24):

    for se in unit_iterator:
        targets_preproc = []
        for tgt in se.targets:
            tgt_idc = [tgt.tokens[0].index_sentence, tgt.tokens[-1].index_sentence]
            tgt_is_arg_and_has_refs = (type(tgt).__name__ == "Participant" or type(tgt).__name__ == "Filler") and \
                (not tgt.canonical_referents in [[],None] and tgt.canonical_referents != "from_canonref")
                # sometimes there are multiple canonrefs tagged
                # this can be a) annotation mistake or
                # b) multiple reference to a group, e.g. "all" refers to three companies
            tgt_is_event_and_has_refs = (type(tgt).__name__ == "Event" and not tgt.coreferents in [[],None]) # collect coreferents for event too
            if tgt_is_arg_and_has_refs or tgt_is_event_and_has_refs:
                refs_attr = "canonical_referents" if tgt_is_arg_and_has_refs else "coreferents"
                print(f'Merging \"{tgt.text}\" with canonrefs {[t.text for t in getattr(tgt, refs_attr)]}.')
                for ref in getattr(tgt, refs_attr):
                    # check whether canonical referent is in same sentence
                    same_sentence = [s.element_id for s in tgt.in_sentence] == [
                        s.element_id for s in ref.in_sentence
                    ]
                    # if skip_cross_sentence is True only replace when canonref is in same sentence
                    ref_idc = [ref.tokens[0].index_sentence, ref.tokens[-1].index_sentence]
                    dist = _dist(tgt_idc, ref_idc)
                    extend_dist = len(ref.tokens) + dist
                    if extend_dist < max_extended:
                        if skip_cross_sentence and same_sentence:
                            targets_preproc.append(ref)
                            print(
                                f"\tAdd coref of {tgt.text} -> {ref.text} in same sentence:\n\t\t{se}"
                            )
                        elif not skip_cross_sentence: # if skip_cross_sentence is False always replace
                            targets_preproc.append(ref)
                            print(
                                f"\tAdd coref of {tgt.txt} -> {ref.text}:\n\t\t{se}."
                            )
                        elif skip_cross_sentence and not same_sentence:
                            print(f'\t!Skipped {ref.text} because not-in-same sentence skipping enabled.')
                            targets_preproc.append(tgt)
                    else: print(f'\t!Skip add {ref.text} for {tgt.text} because exceed max extend.')
            else: # no canon/co-refs so keep it in
                targets_preproc.append(tgt)

        se.targets = list(set(se.targets + targets_preproc))

def preprocess_trigger(ev,
                       max_extended=24,
                       ):

    core = ev.tokens
    print(f'Discont. merging \"{ev.text}\" with parts {[d.text for d in ev.discontiguous_triggers]}')
    core_sent = core[0].in_sentence
    discont_parts = []
    for d in ev.discontiguous_triggers:
        d_sent = d.tokens[0].in_sentence
        dc_toks = sorted(list(set(d.tokens)), key=lambda x: x.index_sentence)
        if core_sent != d_sent:
            print(f'\t!Skip discont. merge: {d.text} cross-sentence.')
        else: discont_parts.append(dc_toks)
    discont_parts.sort(key=lambda x: x[0].index_sentence)

    # now merge
    for dc in discont_parts:
        core = ev.tokens
        core_idc = [core[0].index_sentence, core[-1].index_sentence]
        dc_idc = [dc[0].index_sentence, dc[-1].index_sentence]
        dist = _dist(core_idc, dc_idc)
        if (dist + len(dc)) < max_extended:
            merged_tokens = sorted(list(set(core + dc)), key=lambda x: x.index_sentence)
            merged_idc = [merged_tokens[0].index_sentence, merged_tokens[-1].index_sentence]
            merged_tokens_continuous = core[0].in_sentence.tokens[merged_idc[0]:merged_idc[-1]+1]
            ev.tokens = merged_tokens_continuous
            ev.text = ' '.join(t.text for t in ev.tokens)
        else:
            print(f'\t!Skip discont merge {core} and discont. {dc} because exceeds max extend {max_extended}.')
    return ev

def preprocess_events(project):
    '''
    Preprocess events for mapping 2 sentiment - target annotations.
    Removes in-place sentiment expressions from project.annotation_documents[i].sentiment_expressions and
     project.annotation_documents[i].sentences[j].sentiment_expressions.
    Removes in-place targets from SE.targets.
    :param project:
    :return:
    '''
    # set target accumulator on events
    for ev in project.get_events():
        if ev.discontiguous_triggers:
            ev = preprocess_trigger(ev, max_extended=24)
        ev.targets = []

    filter_target_roles_events(project)
    
    # make cross-sentence annotations in-sentence and count stats of cut relations
    replace_canonical_referents(project.get_events(), skip_cross_sentence=True, max_extended=1024) # no need to restrict by exceeding dist
    remove_cross_sentence_targets(project.get_events())
    remove_pronominal_targets(project.get_events())
    # remove pronominal events
    evs = []
    for ev in project.get_events():
        if ev.check_pronominal():
            print(f'Removed pronominal ev {ev.friendly_id()}.')
        else:
            evs.append(ev)
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
    tgts_orig = [s.targets[:] for s in project.get_sentiment_expressions()]

    # make cross-sentence annotations in-sentence and count stats of cut relations
    replace_canonical_referents(project.get_sentiment_expressions(), skip_cross_sentence=True, max_extended=1024)
    remove_cross_sentence_targets(project.get_sentiment_expressions())
    remove_pronominal_targets(project.get_sentiment_expressions())

    # # compare
    # tgts_proc = [s.targets for s in project.get_sentiment_expressions()]
    # for orig, proc in zip(tgts_orig, tgts_proc):
    #     orig.sort(key=lambda x: (x.begin, x.end))
    #     proc.sort(key=lambda x: (x.begin, x.end))
    #     if orig != proc:
    #         print(orig, proc)

    # remove pronominal ses (there are none in SENTiVENT)
    ses = []
    for se in project.get_sentiment_expressions():
        if se.check_pronominal():
            print(f'Removed pronominal se {se}.')
        else:
            ses.append(se)

    return ses

def to_instances(polex):

    instances = []
    id_cnt = Counter()

    for p in polex:
        p.tokens = sorted(list(set(p.tokens)), key=lambda x: x.index_sentence)
        polex_span = (p.tokens[0].index_sentence, p.tokens[-1].index_sentence)
        polex_text = ' '.join(t.text for t in p.tokens)
        sent_text = ' '.join(t.text for t in p.tokens[0].in_sentence.tokens)
        id_sent = f'{p.document_title.split("_")[0]} {p.in_sentence[0].element_id:02d}'
        id_cnt.update([id_sent])
        id_inst = f'{id_sent} {id_cnt[id_sent]-1:02d}'
        pol = p.polarity_sentiment # scoped is with non-annotated mods so we need unscoped
        target_spans = []
        for tgt in p.targets:
            tgt.tokens = sorted(list(set(tgt.tokens)), key=lambda x: x.index_sentence)
            tgt_span = (tgt.tokens[0].index_sentence, tgt.tokens[-1].index_sentence)
            target_spans.append(tgt_span)
        # merge to make continuous
        merged_spans = sorted([polex_span] + target_spans)
        polex_with_targets_span = (merged_spans[0][0], merged_spans[-1][-1])
        polex_with_targets_text = ' '.join(
            t.text for t in p.tokens[0].in_sentence.tokens[polex_with_targets_span[0]: polex_with_targets_span[-1]+1])

        instance = [id_inst, pol, pol, sent_text,
                    polex_text, polex_with_targets_text,
                    polex_span, polex_with_targets_span, target_spans, ""]
        instances.append(instance)

    return pd.DataFrame(instances, columns=['id', 'polarity', 'polarity_orig', 'sentence',
                                    'polex', 'polex+targets',
                                    'polex_span', 'polex+targets_span', 'target_spans', 'split',])

def main():

    # parse from raw Webanno
    # data_dir = cmd_args.input_dir
    data_dir = settings.MASTER_DIRP # set with settings.py for experiments because of documentation
    print(f"Parsing raw WebAnno data in {data_dir}")
    project = parse_process_project(data_dir, from_scratch=False)

    # preprocess events
    print("Preprocessing event annotations and map to sentiment-target.")
    evs = preprocess_events(project)

    # preprocess annotations
    print("Preprocessing sentiment annotations.")
    ses = preprocess_sentiment(project)

    # parse to GTS format
    print(f"Converting to csv format.")
    df_main = to_instances(evs + ses)
    df_main = df_main.sort_values(['id'])
    # dupes
    dupes_text = df_main[df_main.duplicated(['polarity', 'polex+targets'], keep=False)]
    df_main = df_main.drop_duplicates(['polarity', 'polex+targets'], keep='first')

    print(f"Creating data splits.")
    splits = settings.SPLITS_DOC_EXPERIMENTS
    splits_worker = {v: k for k, vs in splits.items() for v in vs}
    df_main["split"] = df_main.id.apply(lambda x: splits_worker[x.split()[0]])

    # write
    opt_fp = 'sentivent_implicit.csv'
    df_main = df_main.sort_values(['id'])
    df_main.to_csv(opt_fp, sep='\t', index=False)
    print(f'Wrote preprocessed data to {opt_fp}')

    # print some basic stats
    df_polstats = pd.DataFrame()
    df_polstats['polarity_n_total'] = df_main.polarity.value_counts()
    df_polstats['polarity_pct_total'] = df_main.polarity.value_counts(normalize=True)
    for split in ['train', 'dev', 'test']:
        df_polstats[f'polarity_n_{split}'] = df_main[df_main['split'] == split].polarity.value_counts()
        df_polstats[f'polarity_pct_{split}'] = df_main[df_main['split'] == split].polarity.value_counts(normalize=True)
    print(df_polstats[[c for c in df_polstats.columns if 'pct' in c]].transpose())


if __name__ == "__main__":
    main()
