#!/usr/bin/env python3
"""
Parse a project to the DYGIEPP jsonl format.

Format: 1 doc per line

{
"doc_key": "document_id",
"sentences: [["tokens", "in", "first", "sentence"], ["second", "sentence"]],
"events": [ # sublist per sentence
    [   # sen 1
        [ # event 1
            [trigger_tok, event_type], # trigger
            [arg_start_tok, arg_end_tok, arg_type], # args
        ]
    ],
    [], # sen2
],
"clusters": [ # NOT EVENT COREF

]
}

parse_project.py
webannoparser
5/17/19
Copyright (c) Gilles Jacobs. All rights reserved.

Calling script to parse a webanno project.
"""
import json
import random
import itertools
from pathlib import Path
from collections import Counter
from parse_project import parse_project
from parser import Filler
import settings
import spacy
from spacy.tokens import Doc
import argparse
from spacy.symbols import nsubj, VERB, NOUN
from spacy import displacy

random.seed(42)

def parse_typology(typology_fp):
    '''
    Parse the typology json file and return dict
    :param typology_fp:
    :return:
    '''
    with open(typology_fp, "rt") as f_in:
        typology = json.load(f_in)
    return typology

def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc

nlp = spacy.load("en_core_web_lg")
parser = nlp.get_pipe('parser')
nlp.add_pipe(prevent_sentence_boundary_detection, name='prevent-sbd', before='parser')

tagger = nlp.get_pipe('tagger')
prevent_sbd = nlp.get_pipe('prevent-sbd')
parser = nlp.get_pipe('parser')
ner = nlp.get_pipe('ner')


def process_spacy(doc_tokens):
    doc = Doc(nlp.vocab, words=doc_tokens)
    tagger(doc)
    prevent_sbd(doc)
    ner(doc)
    parser(doc)
    return doc

def get_trigger_head(ev, trigger_selection):
    '''
    Parse event sentence with Spacy and return trigger head token.
    '''
    trigger_toks = ev.get_extent_tokens(extent=["discontiguous_triggers"], source_order=True)
    if len(trigger_toks) == 1: # single-token triggers do not need to be parsed
        return trigger_toks[0]
    else: # multi-token triggers
        sen_tokens = [t.text for t in ev.in_sentence[0].tokens]
        doc = process_spacy(sen_tokens)

        trigger_toks_pos = sorted([i for i, t in enumerate(ev.in_sentence[0].tokens) if t in trigger_toks])
        trigger_span = doc[trigger_toks_pos[0]:trigger_toks_pos[-1]+1] # sadly has to be continuous span
        trigger_head = trigger_span.root
        if trigger_selection == "head_noun":
            if trigger_head.pos != NOUN:
                children_noun = [c for c in trigger_head.children if c.pos == NOUN and c.text in ev.get_extent_text(extent=['discontiguous_triggers'])]
                if children_noun:
                    trigger_head = children_noun[0] # pick the closest head noun
        trigger_head_token = ev.in_sentence[0].tokens[trigger_head.i]

        # print(f"{trigger_head}: {trigger_head_token}: {trigger_span}: {ev.text} (main tag)")
        # displacy.serve(doc, style='dep')
        # Trigger options: Head token (full), head token (first), head noun
        # print(f"{ev.event_type}) Head: {trigger_head_token} | Anno_discont: {ev.get_extent_text()} | Anno_first: {ev.text} ")
        return trigger_head_token

def split_train_dev_test(project, dataset):

    # split pre-defined holdin - holdout test
    splits = {"train": [], "dev": [], "test": []}
    holdin_docs = [doc for doc in dataset if doc["doc_key"] in [d.document_id for d in project.dev]]
    splits["test"] = [doc for doc in dataset if doc["doc_key"] in [d.document_id for d in project.test]]

    # split holdin into train - dev by random selection of 1 document belonging to each of the 30 companies
    # 30 documents is also the holdout test set size so works out perfectly
    comp_k = lambda x: ''.join([i for i in x["doc_key"] if not i.isdigit()])
    for company, comp_g in itertools.groupby(sorted(holdin_docs, key=comp_k), comp_k):
        comp_g = list(comp_g)
        dev_choice = random.choice(comp_g)
        splits["dev"].append(dev_choice)
        comp_g.remove(dev_choice)
        splits["train"].extend(comp_g)

    # show event type counts in splits to detect skew
    try:
        trigger_splits = {}
        for split_n, split_data in splits.items():
            for doc in split_data:
                triggers = [ev[0][1] for sen in doc["events"] for ev in sen]
                trigger_splits.setdefault(split_n, Counter()).update(triggers)
        print("Event type distribution in splits:")
        for split_n, c in trigger_splits.items():
            print(f"{split_n}:\t{sorted([(i, round(c[i] / sum(c.values()) * 100.0, 2)) for i in c])}")
    except KeyError as e:
        pass

    return splits

def check_arg_typology(ev, arg, typology):
    '''
    Check if the argument belongs to maintype or subtype.
    If the event is only parsed at the main type level ->
    :param typology:
    :param ev:
    :param arg:
    :return:
    '''

    if type(arg).__name__ == "Filler": # Fillers are not specialized for subtypes and are always valid
        # isinstance doesnt work here (pickling???)
        return True
    else: # only participant should be checked
        if "Historic" in arg.role:
            x = 1 + 1
            pass
        maintype_ff = lambda x: x["full_name"] == ev.event_type and x["maintype_name"] == ev.event_type
        valid_type = next(filter(maintype_ff, typology))

        arg_ff = lambda x: x["name"].split("_")[0] == arg.role.split("_")[0]
        valid_arg = list(filter(arg_ff, valid_type["participant"]))
        if len(valid_arg) > 0:
            return True
        else:
            return False

def parse_to_dygiepp(project, trigger_selection, annotations, typology):

    dataset = []

    for doc in project.annotation_documents:
        d = {}
        d["doc_key"] = doc.document_id
        for sen in doc.sentences:
            d.setdefault("sentences", []).append([t.text for t in sen.tokens])
            # get events
            if annotations == "event":
                events_in_sen = []
                for ev in sen.events:
                    if not ev.event_type: # error with sentiment annotations in which Jef made events by mistake
                        continue
                    event_dygiep = []
                    # add event trigger + type
                    trigger_token = get_trigger_head(ev, trigger_selection) # select single token from span
                    trigger_token_pos = trigger_token.index
                    #TODO parse full sentence with spacy + get head from extent
                    # print("event:", doc.tokens[trigger_token_pos], ev.event_type)
                    event_dygiep.append([trigger_token_pos, ev.event_type])
                    # add event args
                    arguments = set(itertools.chain.from_iterable((ev.participants or [], ev.fillers or [])))
                    for arg in arguments:
                        if check_arg_typology(ev, arg, typology): # check if argument is maintype and not subtype
                            # print(f"{arg} on {ev} valid.")
                            token_start_pos = arg.tokens[0].index
                            token_end_pos = arg.tokens[-1].index
                            arg_type = arg.role.split("_")[0]
                            event_dygiep.append([token_start_pos, token_end_pos, arg_type])
                            # print("Arg:", [t.text for t in doc.tokens[token_start_pos:token_end_pos+1]], arg_type)
                        # else: print(f"INVALID: {arg} on {ev}.")
                    events_in_sen.append(event_dygiep)
                d.setdefault("events", []).append(events_in_sen)

                # parse ner
                d.setdefault("ner", []).append([])

                # parse coref
                d.setdefault("clusters", []).append([])

                # parse relations
                d.setdefault("relations", []).append([])
        dataset.append(d)

    return dataset

def main():

    # arguments
    parser = argparse.ArgumentParser(description="Preprocess SENTiVENT WebAnno event data.")
    parser.add_argument("input_dir", help="Name for input unzipped WebAnno XMI export directory.")
    parser.add_argument("output_dir", help="Name for output directory.")
    parser.add_argument("--trigger_selection",
                        default="head_noun",
                        const="head_noun",
                        nargs="?",
                        choices=["head", "head_noun"],
                        help="Single token selection method for multiple token trigger spans.\
                             Select syntactic head or head noun (default).")
    parser.add_argument("--annotations",
                        default="event",
                        const="event",
                        nargs="?",
                        choices=["none", "event"],
                        help="Parse no annotations (only text) or event (default).")
    args = parser.parse_args()

    # parse from raw Webanno
    print(f"Parsing raw WebAnno data in {settings.MASTER_DIRP}")
    project = parse_project(settings.MASTER_DIRP, from_scratch=False)

    # parse the typology
    typology = parse_typology(settings.TYPOLOGY_FP)

    # parse to dygiepp format
    print(f"Converting to DYGIE++ jsonl format.")
    dataset = parse_to_dygiepp(project, args.trigger_selection, args.annotations, typology)

    # split data
    print(f"Creating data splits.")
    splits = split_train_dev_test(project, dataset)

    # write jsonl
    # opt_dir = Path("/home/gilles/repos/dygiepp/data/ace-event/processed-data/sentivent/json")
    opt_dir = Path(args.output_dir)
    opt_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in splits.items():
        fp = opt_dir / f"{split_name}.jsonl"
        with open(fp, "wt") as d_out:
            for doc in split_data:
                d_out.write(json.dumps(doc) + "\n")
            print(f"Wrote {split_name} DYGIE++ data to {fp}.")

if __name__ == "__main__":
    main()