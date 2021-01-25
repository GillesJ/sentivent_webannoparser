#!/usr/bin/env python3
"""
Parse a project to the DYGIEPP jsonl format.

Format: 1 doc per line

{
"doc_key": "document_id",
"sentences: [["tokens", "in", "first", "sentence"], ["second", "sentence"]],
"events": [ # sublist per sentence
    [   # sen 1
        [ # unit 1
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
import shutil
import random
import itertools
from pathlib import Path
from collections import Counter, deque
from itertools import islice
from parse_project import parse_process_project
from parser import Filler
import settings
import spacy
from spacy.tokens import Doc
import argparse
from spacy.symbols import nsubj, VERB, NOUN
from spacy import displacy

random.seed(42)


def prevent_sentence_boundary_detection(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc


def process_spacy(doc_tokens):
    doc = Doc(nlp.vocab, words=doc_tokens)
    tagger(doc)
    prevent_sbd(doc)
    ner(doc)
    parser(doc)
    return doc


# setup spacy
nlp = spacy.load("en_core_web_lg")
parser = nlp.get_pipe("parser")
nlp.add_pipe(prevent_sentence_boundary_detection, name="prevent-sbd", before="parser")

tagger = nlp.get_pipe("tagger")
prevent_sbd = nlp.get_pipe("prevent-sbd")
parser = nlp.get_pipe("parser")
ner = nlp.get_pipe("ner")

def get_trigger_span(ev):
    if cmd_args.discont_distance and ev.discontiguous_triggers:
        core_tokens = ev.get_extent_tokens(extent=[])
        tokens = core_tokens.copy()
        discont_tokens = [dt.tokens for dt in ev.discontiguous_triggers]
        for dtokens in discont_tokens:
            if core_tokens[-1].index < dtokens[0].index: # discont after core
                dist = dtokens[0].index - core_tokens[-1].index - 1
            elif core_tokens[0].index > dtokens[-1].index: # discont before core
                dist = core_tokens[0].index - dtokens[-1].index - 1
            else: # overlap due to annotation error
                dist = 0
            if dist <= cmd_args.discont_distance:
                tokens.extend(dtokens)
                print(f"Parsed discont. span from core {[t.text for t in core_tokens]} -> {[t.text for t in tokens]}.")
            else:
                print(f"Skipped discont. span {[t.text for t in dtokens]} to trigger {[t.text for t in tokens]} because distance exceeded.")
        tokens = sorted(list(set(tokens)), key=lambda x: x.begin)
    else:
        tokens = ev.get_extent_tokens(extent=["discontiguous_triggers"])
    # make continuous
    tokens_sentence = tokens[0].in_sentence.tokens
    tokens_cont = [t for t in tokens_sentence if tokens[0].index_sentence <= t.index_sentence <= tokens[-1].index_sentence]

    return tokens_cont

def parse_typology(typology_fp):
    """
    Parse the typology json file and return dict
    :param typology_fp:
    :return:
    """
    with open(typology_fp, "rt") as f_in:
        typology = json.load(f_in)
    return typology

def truncate_contentful(toks, anno, width):

    def sliding_window_iter(iterable, size):
        iterable = iter(iterable)
        window = deque(islice(iterable, size), maxlen=size)
        for item in iterable:
            yield tuple(window)
            window.append(item)
        if window:
            # needed because if iterable was already empty before the `for`,
            # then the window would be yielded twice.
            yield tuple(window)

    # get processed spacy span
    doc = anno.get_processed_sentence()[0]

    span_orig = doc[
        toks[0].index_sentence : toks[-1].index_sentence + 1
    ]
    try:
        label = "TRIG " + anno.event_type
    except:
        label = "ARG " + anno.role

    # generate and score candidates
    winner = (0, None)
    for cand in sliding_window_iter(span_orig, width):
        score = 0
        for t in cand:
            if t == span_orig.root:
                score += 1.0
            if t.pos_ in ["ADJ", "VERB", "NUM", "PROPN"]:
                score += 1.0
            if t.pos_ in ["NOUN"]:
                score += 1.5
            if t.dep_ in ["nsubj", "ROOT"]:
                score += 0.5
            if t.dep_ in ["amod", "quantmod"]:
                score += 0.5
        extremes_penalty = ["PUNCT", "ADP", "CCONJ"]
        # prefer numbers at start for args
        if cand[0].pos_ == "NUM" and "ARG" in label:
            score += 1.5
        # penalize constructs ending or starting with PUNCT or PREPOSITION
        if cand[0].pos_ in ["DET"]:
            score -= 0.5
        if cand[0].pos_ in extremes_penalty:
            score -= 1.25
        if cand[-1].pos_ in extremes_penalty:
            score -= 0.5
        # set winner
        if score > winner[0]:
            winner = (score, cand)
        # print("\t",score, cand, [t.pos_ for t in cand], [t.dep_ for t in cand])
    # print(f"\tWinner: {winner}")

    winner_idx = [t.i for t in winner[1]]
    toks_trunc = [t for t in toks if t.index_sentence in winner_idx]
    assert len(toks_trunc) == width
    print(f"Truncated {label} \"{span_orig}\" ({len(span_orig)}) to \"{' '.join(t.text for t in toks_trunc)}\" ({width}).")
    return toks_trunc


def select_trigger_head(trigger_toks, ev):
    """
    Parse event sentence with Spacy and return trigger head token.
    """
    doc = ev.get_processed_sentence()[0]

    trigger_toks_pos = [t.index_sentence for t in trigger_toks]
    trigger_span = doc[
        trigger_toks_pos[0] : trigger_toks_pos[-1] + 1
    ]  # sadly has to be continuous span
    if cmd_args.trigger_selection == "head_noun":
        trigger_head = trigger_span.root
        if trigger_head.pos != NOUN:
            trigger_toks_text = [t.text for t in trigger_toks]
            children_noun = [
                c
                for c in trigger_head.children
                if c.pos == NOUN
                and c.text in trigger_toks_text
            ]
            if children_noun:
                trigger_head = children_noun[0]  # pick the closest head noun
    elif cmd_args.trigger_selection == "head":
        trigger_head = trigger_span.root
    else:
        raise ValueError(f"\"{cmd_args.trigger_selection}\" is not a valid trigger selection strategy.")
    trigger_head_token = ev.in_sentence[0].tokens[trigger_head.i]

    # print(f"{trigger_head}: {trigger_head_token}: {trigger_span}: {ev.text} (main tag)")
    # displacy.serve(doc, style='dep')
    # Trigger options: Head token (full), head token (first), head noun
    # print(f"{ev.event_type}) Head: {trigger_head_token} | Anno_discont: {ev.get_extent_text()} | Anno_first: {ev.text} ")
    return [trigger_head_token]


def split_train_dev_test(project, dataset):

    # split pre-defined holdin - holdout test
    splits = {"train": [], "dev": [], "test": []}
    holdin_docs = [
        doc for doc in dataset if doc["doc_key"] in [d.document_id for d in project.dev]
    ]
    splits["test"] = [
        doc
        for doc in dataset
        if doc["doc_key"] in [d.document_id for d in project.test]
    ]

    # split holdin into train - dev by random selection of 1 document belonging to each of the 30 companies
    # 30 documents is also the holdout test set size so works out perfectly
    comp_k = lambda x: "".join([i for i in x["doc_key"] if not i.isdigit()])
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
                triggers = [ev[0][-1] for sen in doc["events"] for ev in sen]
                trigger_splits.setdefault(split_n, Counter()).update(triggers)
        print("Event type distribution in splits:")
        for split_n, c in trigger_splits.items():
            print(
                f"{split_n}:\t{sorted([(i, round(c[i] / sum(c.values()) * 100.0, 2)) for i in c])}"
            )
    except KeyError as e:
        pass

    return splits


def check_arg_typology(ev, arg, typology):
    """
    Check if the argument belongs to maintype or subtype.
    If the event is only parsed at the main type level ->
    :param typology:
    :param ev:
    :param arg:
    :return:
    """

    if (
        type(arg).__name__ == "Filler"
    ):  # Fillers are not specialized for subtypes and are always valid
        # isinstance doesnt work here (pickling???)
        return True
    else:  # only participant should be checked
        maintype_ff = (
            lambda x: x["full_name"] == ev.event_type
            and x["maintype_name"] == ev.event_type
        )
        valid_type = next(filter(maintype_ff, typology))

        arg_ff = lambda x: x["name"].split("_")[0] == arg.role.split("_")[0]
        valid_arg = list(filter(arg_ff, valid_type["participant"]))
        if len(valid_arg) > 0:
            return True
        else:
            return False


def parse_sentiment_relations(sentiment_expression, doc_offset):
    """
    :param sentiment_expression:
    :return:
    """
    sen_rels = []
    se_start = sentiment_expression.tokens[0].index
    se_end = sentiment_expression.tokens[-1].index
    label = "SentiPol." + sentiment_expression.polarity_sentiment_scoped
    se_sentence_ids = [s.element_id for s in sentiment_expression.in_sentence]
    for t in sentiment_expression.targets:
        # same sentence check TODO make optional cmdline argument
        # TODO add event full extent getting for token span
        same_sentence = [s.element_id for s in t.in_sentence] == se_sentence_ids
        if not same_sentence:
            print(
                f"Sentence-crossing sentiment-target relation skipped for {sentiment_expression.friendly_id()}\n->{t.friendly_id()}"
            )
            continue
        else:
            target_start = t.tokens[0].index + doc_offset
            target_end = t.tokens[-1].index + doc_offset
            sen_rels.append([se_start, se_end, target_start, target_end, label])
    return sen_rels


def get_trigger(ev):

    # first get full token span with discont. Made continuous for DYGIE format!
    trigger_tokens = get_trigger_span(ev)
    # If trigger selection option set and trigger span is multitoken: make single-trigger selection
    if cmd_args.trigger_selection != "none" and len(trigger_tokens) > 1:
        trigger_tokens = select_trigger_head(trigger_tokens, ev)

    # truncate if trigger_span_width option is set
    if cmd_args.trigger_span_width > 0 and len(trigger_tokens) > cmd_args.trigger_span_width:
        trigger_tokens = truncate_contentful(trigger_tokens, ev, cmd_args.trigger_span_width)

    return trigger_tokens

def parse_to_dygiepp(project, typology):

    dataset = []

    # load role mapping
    role_map_flag = False
    if cmd_args.role_map:
        with open(cmd_args.role_map, "rt") as jsonin:
            role_map = json.load(jsonin)
        # unroll onetarget-to-many json map into one-one dict
        role_map = {vv: k for k, v in role_map.items() for vv in v}
        role_map_flag = True

    # replace arguments by canonical referent only if they are in same sentence
    if cmd_args.replace_canonical_referents:
        project.replace_canonical_referents(cross_sentence=cmd_args.allow_cross_sentence)

    ev_skipped_n = 0
    ev_total_n = 0
    args_skipped_n = 0
    args_total_n = 0
    serel_total_n = 0
    serel_parsed_n = 0

    for doc in project.annotation_documents:
        d = {}
        d["doc_key"] = doc.document_id
        d["dataset"] = cmd_args.dataset
        doc_offset = 0

        for sen in doc.sentences:
            # sentence as list of tokens
            sen_tokens = [t.text for t in sen.tokens]
            if sen_tokens == ["."]: # clean wrong single-token "." sentence.
                doc_offset += -1
                continue
            if len(sen_tokens) < 2: # for DYGIE++ single-token sentences break modeling code. In our corpus these are single-token headings.
                sen_tokens += "." # Add punctuation to solve.
                doc_offset += 1
            d.setdefault("sentences", []).append(sen_tokens)
            # get events
            events_in_sen = []
            if cmd_args.annotations == "event" or cmd_args.annotations == "event+sentiment":
                for ev in sen.events:
                    ev_total_n += 1
                    if (
                        cmd_args.discard_pronominal and ev.check_pronominal()
                    ):  # PRONOMINAL EVENT TRIGGER > do not include TODO make option
                        ev_skipped_n += 1
                        continue
                    if (
                        not ev.event_type
                    ):  # error with sentiment annotations in which Jef made events by mistake
                        ev_total_n -= 1
                        continue
                    event_dygiep = []
                    # add event trigger + type
                    trigger_span = get_trigger(ev)
                    ev_parsed = [trigger_span[0].index + doc_offset, trigger_span[-1].index + doc_offset, ev.event_type]
                    event_dygiep.append(ev_parsed)
                    # add event args
                    arguments = set(
                        itertools.chain.from_iterable(
                            (ev.participants or [], ev.fillers or [])
                        )
                    )
                    # if arguments is None:
                    #     print("No args", ev.friendly())
                    #     continue
                    for arg in arguments:
                        args_total_n += 1
                        if cmd_args.discard_pronominal and arg.check_pronominal():  # PRONOMINAL PARTICIPANT SKIP
                            print(
                                f"{arg.friendly_id()} not included because pronominal."
                            )
                            args_skipped_n += 1
                            continue
                        if not cmd_args.allow_cross_sentence and ([s.element_id for s in arg.in_sentence] != [
                            s.element_id for s in ev.in_sentence
                        ]): # check arg-event cross sentence
                            print(
                                f"{arg.text}.{arg.role} [{arg.in_sentence[0].element_id}] not included because crosses sentence [{ev.in_sentence[0].element_id}]."
                            )
                            args_skipped_n += 1
                            continue
                        # truncate to configured size
                        arg_tokens = sorted(list(set(arg.tokens)), key=lambda x: x.index_sentence) # make set because some tokens are duplicated
                        if len(arg_tokens) > cmd_args.span_width > 0:
                            arg_tokens = truncate_contentful(arg_tokens, arg, cmd_args.span_width)
                        token_start_pos = arg_tokens[0].index + doc_offset
                        token_end_pos = arg_tokens[-1].index + doc_offset
                        arg_type = arg.role.split("_")[0]
                        # print(arg_type)
                        # map roles if option enabled
                        if role_map_flag and arg_type in role_map:
                            arg_type = role_map[arg_type]
                        event_dygiep.append([token_start_pos, token_end_pos, arg_type])
                        # print("Arg:", [t.text for t in doc.tokens[token_start_pos:token_end_pos+1]], arg_type)
                    events_in_sen.append(event_dygiep)
            d.setdefault("events", []).append(events_in_sen)

            # parse ner
            d.setdefault("ner", []).append([])

            # parse coref
            d.setdefault("clusters", []).append([])

            # parse sentiment as relations
            if "sentiment" in cmd_args.annotations:
                # parse sentiment
                sentiment_relations = []
                for se in sen.sentiment_expressions:
                    sentiment_relations.extend(parse_sentiment_relations(se, doc_offset))
                    serel_total_n += len(se.targets)
                serel_parsed_n += len(sentiment_relations)
                d.setdefault("relations", []).append(sentiment_relations)
            else:
                d.setdefault("relations", []).append([])
        dataset.append(d)

    if "event" in cmd_args.annotations:
        print(
            f"Skipped {round(100 * ev_skipped_n/ev_total_n, 2)}% ({ev_skipped_n}/{ev_total_n}) pronominal events."
        )
        print(
            f"Skipped {round(100 * args_skipped_n/args_total_n, 2)}% ({args_skipped_n}/{args_total_n}) pronominal and/or sentence-crossing arguments."
        )
    if "sentiment" in cmd_args.annotations:
        print(
            f"Skipped {round(100 * (serel_total_n-serel_parsed_n)/serel_total_n, 2)}% ({serel_total_n-serel_parsed_n}/{serel_total_n}) sentiment-target relations due to sentence-crossing."
        )

    if cmd_args.ner_predictions: # merge NER predictions
        with open(cmd_args.ner_predictions, "rt") as pred_in:
            preds = [json.loads(l) for l in pred_in.readlines()]
        doc_key = lambda x: x["doc_key"]
        for (parse, pred) in zip(sorted(dataset, key=doc_key), sorted(preds, key=doc_key)):
            assert parse["doc_key"] == pred["doc_key"]
            parse["ner"] = pred["predicted_ner"]

    return dataset

class PolyMap:
    def __init__(self):
        self._map = {}

    def __setitem__(self, key, value):
        self._map[key] = value

    def __getitem__(self, item):
        for keys in self._map:
            if item == keys:
                return self._map[item]
            if item in keys:
                return self._map[keys]
        return None

def main():
    global cmd_args
    # arguments
    parser = argparse.ArgumentParser(
        description="Preprocess SENTiVENT WebAnno event data."
    )
    parser.add_argument(
        "input_dir", help="Name for input unzipped WebAnno XMI export directory."
    )
    parser.add_argument("output_dir", help="Name for output directory.")
    parser.add_argument(
        "--role_map",
        nargs="?",
        help="JSON file containing mapping for roles. \
                              Used for combining roles or removing rules when setting key to None.",
    )
    parser.add_argument(
        "--ner_predictions",
        nargs="?",
        help="DYGIEPP output file containing NER predictions. \
            Used for combining merging silver-standard NER labels.",
    )
    parser.add_argument(
        "--dataset",
        default="sentivent",
        const="sentivent",
        nargs="?",
        help="Dataset name key for format in update DYGIE++ code.",
    )
    parser.add_argument(
        "--trigger_selection",
        default="none",
        const="none",
        nargs="?",
        choices=["none", "head", "head_noun"],
        help="Single token selection method for multiple token trigger spans. \
            Select syntactic head, head noun, or no selection (keep multi-token triggers) (default).",
    )
    parser.add_argument(
        "--trigger_span_width",
        default=0,
        nargs='?',
        type=int,
        help="Max. trigger span width. Truncate trigger spans that are longer to this width.",
    )
    parser.add_argument(
        "--span_width",
        default=0,
        nargs='?',
        type=int,
        help="Max. span width. Truncate other spans (args, targets, sents) annotations to this width.",
    )
    parser.add_argument(
        "--annotations",
        default="event",
        const="event",
        nargs="?",
        choices=["none", "event", "event+sentiment"],
        help="Parse no annotations (only text) or event (default).",
    )
    parser.add_argument("--discont_distance",
                        nargs='?',
                        type=int,
                        help="Max. token distance to follow when parsing discontiguous spans. \
                             Do not follow discontiguous if tokens between main and discont. exceeds n. DO NOT USE."
    )
    parser.add_argument('--replace_canonical_referents',
                        action='store_true',
                        help="Replace argument and target annotations with their linked CanonicalReferent. \
                             Cross-sentence CanonicalReferent links are skipped by default if --allow_cross_sentence is not used."
                        )
    parser.add_argument('--allow_cross_sentence',
                    action='store_true',
                    help="Allow cross-sentence relations (event-argument, sentiment-target (disabled by default)."
                    )
    parser.add_argument('--discard_pronominal',
                    action='store_true',
                    help="Skip pronominal annotations such as events, arguments and sentiment targets that are the result of coreference."
                    )
    cmd_args = parser.parse_args()

    # parse from raw Webanno
    print(f"Parsing raw WebAnno data in {settings.MASTER_DIRP}")
    project = parse_process_project(settings.MASTER_DIRP, from_scratch=False)

    # parse the typology
    typology = parse_typology(settings.TYPOLOGY_FP)

    # parse to dygiepp format
    print(f"Converting to DYGIE++ jsonl format.")
    dataset = parse_to_dygiepp(project, typology)

    # split data
    print(f"Creating data splits.")
    splits = split_train_dev_test(project, dataset)

    # write jsonl
    # opt_dir = Path("/home/gilles/repos/dygiepp/data/ace-event/processed-data/sentivent/json")
    opt_dir = Path(cmd_args.output_dir)
    opt_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_data in splits.items():
        fp = opt_dir / f"{split_name}.jsonl"
        with open(fp, "wt") as d_out:
            for doc in split_data:
                d_out.write(json.dumps(doc) + "\n")
            print(f"Wrote {split_name} DYGIE++ data to {fp}.")

    # backup role map to opt
    if cmd_args.role_map:
        shutil.copy(cmd_args.role_map, str(opt_dir / "role-map.json"))


if __name__ == "__main__":
    main()
