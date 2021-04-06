#!/usr/bin/env python3
"""
TODO: create a LinkFeature superclass for parsing the linked features to avoid code duplication and increase efficiency
TODO: refactor current Document class into a Document class and an Annotation class:
        Document contains the text, sentences, tokens, metadata and annotation_documents.
        Annotation contains all annotation_documents and the parsing code.
    This allows for less duplication (text, tokens, sentences and metadata are all the same for the same doc).
TODO: write generic parser for webanno features (Link, Slot, etc):
        cassis is slower than this prototype but more general
parser.py
sentivent_webannoparser
10/4/18
Copyright (c) Gilles Jacobs. All rights reserved.
"""
from __future__ import annotations

import sys

sys.path.append("/home/gilles/repos/")

import settings
from dataclasses import dataclass, field
from zipfile import ZipFile
import xml.dom.minidom as md

# import cassis
import fnmatch
import spacy
from pathlib import Path
from typing import List, Any
from sentivent_webannoparser.util import flatten, count_avg, pickle_webanno_project
import os
import multiprocessing
import json
from itertools import groupby, combinations_with_replacement
from copy import copy
from math import log


@dataclass
class Element:
    text: str
    begin: int = field(repr=False)
    end: int = field(repr=False)
    element_id: int = field(repr=False)
    annotator_id: str
    document_title: str
    in_document: Any = field(repr=False)

    def __hash__(self):
        return hash((self.element_id, self.text, self.begin, self.end))

    def friendly_id(self):
        """
        Make a representation of object that is easy to find in corpus.
        :return:
        """
        id = f"{self.annotator_id}_{self.document_title.split('_')[0]}"

        try:  # try making an sentence identifier if there is an in_sentence attrib
            sen_id = ",".join(str(se.element_id + 1) for se in self.in_sentence)
            id += f"_s{sen_id}"
        except Exception as e:
            print(e)
            pass

        if isinstance(self, Event):
            id += f"_{self.event_fulltype}"
        elif isinstance(self, Participant) or isinstance(self, Filler):
            id += f"_{self.role}"

        text_ellips = (
            (self.text[:15] + ".." + self.text[-15:])
            if len(self.text) > 32
            else self.text
        )
        id += f"-{text_ellips}"
        return id

    def get_processed_sentence(self):
        """
        Return the Spacy processed sentences in which the element is positioned.
        :return:
        """

        if not hasattr(self, "in_sentence"):
            raise AttributeError(f"{self} does not have attribute 'in_sentence'.")

        try:
            sen_ixs = [sen.element_id for sen in self.in_sentence]
        except TypeError as e:
            sen_ixs = [self.in_sentence.element_id]

        sen_procs = [
            s
            for i, s in enumerate(self.in_document.sentences_processed)
            if i in sen_ixs
        ]
        return sen_procs

    def get_processed_tokens(self, **kwargs):

        if hasattr(
            self, "get_extent_tokens"
        ):  # for events and sentimentexpr which have custom token getters
            token_origs = self.get_extent_tokens(**kwargs)
        else:
            token_origs = self.tokens
        sen_procs = self.get_processed_sentence()

        # some annotation can exceptionally span multiple sentences (actually not the case in practice)
        # groupby sentence to retrieve token corresponding to sentence
        sen_key = lambda x: x.in_sentence.element_id
        token_origs_by_sen = [
            list(g) for k, g in groupby(sorted(token_origs, key=sen_key), sen_key)
        ]

        token_procs = []
        for sen_tok_orig, sen_proc in zip(token_origs_by_sen, sen_procs):
            sen_token_orig_ixs = [t.index_sentence for t in sen_tok_orig]
            sen_token_procs = [
                t_proc for i, t_proc in enumerate(sen_proc) if i in sen_token_orig_ixs
            ]
            token_procs.extend(sen_token_procs)

        return token_procs

    def check_pronominal(self):
        """
        Check if the annotation is pronominal by full parsed PoS tag.
        All tokens should be pronominal (works best for anaphoric pronominal mentions as intended).
        :return:
        """
        pronom_tags = ["PRP", "PRP$", "WDT", "WP", "WP$"]
        token_procs = self.get_processed_tokens()
        all_pronom = all(
            t.tag_ in pronom_tags for t in token_procs
        )  # True if all tokens are pronom_tags
        # print(f"{' '.join(t.text + '.' + t.tag_ for t in token_procs)}: Pronominal = {all_pronom}")
        return all_pronom

    def replace_by_canonical_referent(self, cross_sentence=True):
        """
        Replace annotation unit by its Canonical referent.
        :param cross_sentence: Set to False: Do not replace by CanonicalReferent if the link crosses sentence boundary.
        :return:
        """
        replaced_count = 0

        # replace the participant in their containers
        def replace_in_containers(myobj, replacing):
            containers = {
                "event": "participants",
                "sentiment_expression": "targets",
                "sentence": "participants",
            }
            for c_name, attrib_n in containers.items():
                c_name = f"in_{c_name}"
                if hasattr(myobj, c_name):
                    for c in getattr(myobj, c_name):
                        to_replace_in = getattr(c, attrib_n)
                        to_replace_in.append(replacing)
                        if myobj in to_replace_in:
                            to_replace_in.remove(myobj)
                        to_replace_in.sort(key=lambda x: x.begin)

        if self.canonical_referents and self.canonical_referents != "from_canonref":
            # sometimes there are multiple canonrefs tagged
            # this can be a) annotation mistake or
            # b) multiple reference to a group, e.g. "all" refers to three companies.
            for canonref in self.canonical_referents:
                # check whether canonical referent is in same sentence
                same_sentence = [s.element_id for s in self.in_sentence] == [
                    s.element_id for s in canonref.in_sentence
                ]
                # always replace when cross_sentence is true
                # if cross_sentence is False (disallowed) only replace when canonref is in same sentence
                if cross_sentence or same_sentence:
                    # replace element
                    replacing_participant = Participant(
                        canonref.text,
                        canonref.begin,
                        canonref.end,
                        canonref.element_id,
                        canonref.annotator_id,
                        canonref.document_title,
                        canonref.in_document,
                        self.role,
                        "from_canonref",
                        self.link_id,
                        canonref.tokens,
                    )
                    replacing_participant.in_sentence = self.in_sentence
                    replacing_participant.in_document = self.in_document
                    if hasattr(self, "in_sentiment_expression"):
                        replacing_participant.in_sentiment_expression = (
                            self.in_sentiment_expression
                        )
                    replacing_participant.from_original_participant = copy(self)
                    print(
                        f"Replaced {self} with {canonref}. (in same sentence:{same_sentence})"
                    )

                    replace_in_containers(self, replacing_participant)

                    # replace on document
                    self.in_document.participants.append(replacing_participant)
                    if self in self.in_document.participants:
                        self.in_document.participants.remove(self)

                    replaced_count += 1
        return replaced_count

@dataclass
class Filler(Element):
    role: str
    link_id: int = field(repr=False)
    tokens: List = field(default=None, repr=False)

    def __hash__(self):
        return hash((self.element_id, self.text, self.begin, self.end))


@dataclass
class DiscontiguousTrigger(Element):
    link_id: int = field(repr=False)
    tokens: List = field(default=None, repr=False)

    def __hash__(self):
        return hash((self.element_id, self.text, self.begin, self.end))


@dataclass
class CanonicalReferent(Element):
    pronom_id: int = field(repr=False)
    referent_id: int
    tokens: List = field(default=None, repr=False)

    def __hash__(self):
        return hash((self.element_id, self.text, self.begin, self.end))


@dataclass
class Participant(Element):
    role: str
    canonical_referents: List[CanonicalReferent] = field(repr=False)
    link_id: int = field(repr=False)
    tokens: List = field(default=None, repr=False)

    def get_extent_text(self):
        return " ".join(t.text for t in sorted(list(set(self.tokens)), key=lambda x: x.index_sentence))

    def __hash__(self):
        return hash((self.element_id, self.text, self.begin, self.end))


class Sentence:
    """
    Class to represent a sentence

    Variables:
        id - number of this sentence in the text
        begin - position of the sentence start in the original text
        end - position of the sentecne end in the original text
        frames - list of frames contained in this sentence
    """

    def __init__(
        self,
        element_id,
        begin,
        end,
        tokens,
        events,
        sentiment_expressions,
        participants,
        fillers,
        canonical_referents,
        targets,
    ):
        self.element_id = element_id
        self.begin = begin
        self.end = end
        self.tokens = tokens
        self.events = events
        self.sentiment_expressions = sentiment_expressions
        self.participants = participants
        self.fillers = fillers
        self.canonical_referents = canonical_referents
        self.targets = targets

    def __eq__(self, other):
        if self.begin == other.begin and self.end == other.end:
            if [x for x in self.events if not x in other.events] == [] and [
                x for x in other.events if not x in self.events
            ] == []:
                return True
        return False

    def __str__(self):
        return " ".join(str(t) for t in self.tokens)

    def __repr__(self):
        """
        Custom repr for easier debugging.
        :return: repr string
        """
        text = str(self)
        text_ellips = (text[:31] + ".." + text[-31:]) if len(text) > 64 else text
        return f"{self.element_id}. {text_ellips}..."


@dataclass
class Event(Element):
    event_type: str
    event_subtype: str
    event_fulltype: str
    discontiguous_triggers: List[DiscontiguousTrigger] = field(repr=False)
    participants: List[Participant] = field(repr=False)
    fillers: List[Filler] = field(repr=False)
    polarity_negation: str = field(repr=False)
    modality: str = field(repr=False)
    realis: str = field(repr=False)
    polarity_sentiment: str = field(repr=True)
    polarity_sentiment_scoped: str = field(repr=False)
    tokens: List = field(default=None, repr=False)
    in_sentence: List[Sentence] = field(default=None, repr=False)
    coordination: List[Event] = field(default=None, repr=False)
    coreferents: List[Event] = field(default=None)

    def __hash__(self):
        return hash((self.element_id, self.text, self.begin, self.end))

    def get_coref_attrib(self, attrib):
        if self.coreferents is not None:
            coref_attribs = []
            for coref in self.coreferents:
                walk_attrib = coref.get_coref_attrib(attrib)
                if walk_attrib is not None:
                    coref_attribs.append(walk_attrib)
                attrib_val = getattr(coref, attrib)
                if attrib_val is not None:
                    coref_attribs.append(attrib_val)

            if coref_attribs:
                return flatten(coref_attribs)
            else:
                return None
        else:
            return None

    def get_sentence_text(self):
        return " ".join(t.text for sen in self.in_sentence for t in sen.tokens)

    def get_extent_tokens(self, extent=["discontiguous_triggers"], source_order=True):
        """
        Get a list of token objects for the event extent.
        The extent can be set to include discontiguous_triggers, participants, and/or fillers.
        Setting this to an empty list will only return the original
        In the definition of an event nugget we include all of these.
        :param extent: a list of elements with which the event annotation span can be extended. Default: discontigious_triggers, allowed: ["discontiguous_triggers", "participants", "fillers"]
        :param source_order: Maintain the word order of tokens in the source AnnotationDocument.
        Setting to False preserves the order of the annotation process by the annotators.
        This is an unreliable approximation of token span salience and cannot be recommended. Default: True
        :return: List of tokens
        """
        all_tokens = self.tokens.copy()
        core_sen_idx = all_tokens[0].in_sentence.element_id # to ensure discont is in same sentence

        for ext in extent:
            if getattr(self, ext):
                all_tokens.extend(t for x in getattr(self, ext) for t in x.tokens if t.in_sentence.element_id == core_sen_idx)

        all_tokens = list(
            set(all_tokens)
        )  # this is necessary because trigger and participant spans can overlap
        if source_order:  # return tokens in the order they appear in source text
            all_tokens.sort(key=lambda x: x.begin)

        return all_tokens

    def get_extent_token_ids(self, **kwargs):
        """
        Get a list of token ids for the event extent.
        Relies on Event.get_extent_tokens() for fetching the tokens.
        :param kwargs:
        :return: a list of token ids of the event extent that are unique in the document.
        """
        token_span = self.get_extent_tokens(**kwargs)
        return [t.index for t in token_span]

    def get_extent_text(
        self,
        extent=["discontiguous_triggers", "participants", "fillers"],
        source_order=True,
    ):
        return " ".join(
            t.text
            for t in self.get_extent_tokens(extent=extent, source_order=source_order)
        )

    def _fix_false_discont(self):

        fixed = [self.tokens]
        new_discont = self.discontiguous_triggers[:]
        idc = self.trigger_parts_idc
        parts = self.trigger_parts

        for i in range(len(idc)-1):
            if idc[i][1] + 1 >= idc[i+1][0]: # check if adjacent and merge
                if parts[i] not in fixed:
                    fixed.append(parts[i])
                if parts[i] in [ndc.tokens for ndc in new_discont]:
                    new_discont = [dc for dc in new_discont if dc.tokens != parts[i]]
                if parts[i+1] not in fixed:
                    fixed.append(parts[i+1])
                if parts[i+1] in [ndc.tokens for ndc in new_discont]:
                    new_discont = [dc for dc in new_discont if dc.tokens != parts[i+1]]

        self.tokens = sorted(list(set(i for l in fixed for i in l)), key=lambda x: x.index_sentence) # flatten, dedupe, and sort
        self.discontiguous_triggers = new_discont
        self.trigger_parts = sorted([self.tokens] + [d.tokens for d in self.discontiguous_triggers], key=lambda x: x[0].index_sentence)
        self.trigger_parts_idc = [(part[0].index_sentence, part[-1].index_sentence) for part in self.trigger_parts]

    def _generate_conti_combos(self, max_dist):

        conti_combo = [self.trigger_parts[i:j+1] for i,j in combinations_with_replacement(range(len(self.trigger_parts)),2)]
        conti_combo = [c for c in conti_combo if c[-1][-1].index_sentence - c[0][0].index_sentence < max_dist]
        conti_combo_idc = [(c[0][0].index_sentence, c[-1][-1].index_sentence) for c in conti_combo]
        conti_combo_tokens = [self.in_sentence[0].tokens[i[0]:i[1]+1] for i in conti_combo_idc]
        return conti_combo_tokens


    def _score_content_tokens(self, tokens):

        # extract preproc token span
        doc = tokens[0].get_processed_sentence()[0]

        span_orig = doc[
            tokens[0].index_sentence : tokens[-1].index_sentence + 1
        ]

        # generate and score candidates
        score = 0
        for t in span_orig:
            parts_idc = set(t.index_sentence for p in self.trigger_parts for t in p)
            if t.i in parts_idc:
                if t.pos_ in ["ADJ", "VERB", "ADV", "NUM", "NOUN", "ADP"]:
                    score += 1.0
                elif t.pos_ in ["AUX", "PROPN"]:
                    score += 0.5
            # boost core annotation
            # if t.i in set(t.index_sentence for t in self.tokens):
            #     score += 0.5

        return score

    def preprocess_trigger(self,
                           fix_false_discont=True,
                           make_continuous_max_dist=0,
                           truncate_to_len=False,
                           ):

        self.tokens_orig = self.tokens[:]
        self.discontiguous_triggers_orig = self.discontiguous_triggers[:] if self.discontiguous_triggers else None
        if self.discontiguous_triggers:
            self.trigger_parts = sorted([self.tokens] + [d.tokens for d in self.discontiguous_triggers], key=lambda x: x[0].index_sentence)
        else:
            self.trigger_parts = [self.tokens]
        self.trigger_parts_idc = [(part[0].index_sentence, part[-1].index_sentence) for part in self.trigger_parts]

        if self.discontiguous_triggers and fix_false_discont: # no discontinuous preproc needed
            # first fix annotation artifacts with adjacent discontinuous parts to make them continuous
            self._fix_false_discont()

        if self.discontiguous_triggers:  # generate all combinations of continual discont parts
            print(f"-----Making discont parts. continuous: [{' ... '.join(' '.join(t.text for t in p) for p in self.trigger_parts)}]")
            conti_combos = self._generate_conti_combos(make_continuous_max_dist)
            conti_combos_scores = [self._score_content_tokens(tokens) for tokens in conti_combos]
            # score by contentfullness scorer and get contentfullness/length ratio
            conti_combos_scores_scaled = [(log(score))/(1+log(len(combo))) if score else 0 for combo, score in zip(conti_combos, conti_combos_scores)]
            max_i = conti_combos_scores_scaled.index(max(conti_combos_scores_scaled))
            for i, (combo, score, s_score) in enumerate(zip(conti_combos, conti_combos_scores, conti_combos_scores_scaled)):
                prefix = " -> " if i == max_i else "    "
                print(f"{prefix}{' '.join(t.text for t in combo)} |\tscore: {score}\tlen ratio: {s_score}")
            self.tokens = conti_combos[max_i]


@dataclass
class SentimentExpression(Element):
    polarity_sentiment: str = field(repr=True)
    polarity_sentiment_scoped: str = field(repr=False)
    uncertain: str = field(repr=False)
    negated: str = field(repr=True)
    targets: List = field(repr=False)
    tokens: List[Token] = field(default=None, repr=False)
    in_sentence: List[Sentence] = field(default=None, repr=False)

    def __str__(self):
        id = f"|{self.annotator_id[:3]}|{self.in_document.document_id}|s{self.in_sentence[0].element_id:02d}|"
        se_text = f"{id} <{self.polarity_sentiment_scoped.upper()[:3]}> {self.get_extent_text()}"
        spacing = len(se_text) * " "
        tgt_text = [f"|s{t.in_sentence[0].element_id:02d}| {t.get_extent_text()}" for t in self.targets]
        return f"{se_text} --> " + f"\n{spacing} └-> ".join(tgt_text)

    def __hash__(self):
        return hash((self.document_title, self.element_id, self.text, self.begin, self.end))

    def get_extent_tokens(self, extent=[], source_order=True):
        """
        Get a list of token objects for the sentiment expression extent.
        The extent can be set to include targets.
        Setting this to an empty list will only return the original tokens.
        :param extent: a list of elements with which the event annotation span can be extended. Default: []
        :param source_order: Maintain the word order of tokens in the source AnnotationDocument.
        Setting to False preserves the order of the annotation process by the annotators.
        This is an unreliable approximation of token span salience and cannot be recommended. Default: True
        :return: List of tokens
        """
        all_tokens = self.tokens.copy()

        for ext in extent:
            if getattr(self, ext):
                all_tokens.extend(t for x in getattr(self, ext) for t in x.tokens)

        all_tokens = list(
            set(all_tokens)
        )  # this is necessary because trigger and participant spans can overlap
        if source_order:  # return tokens in the order they appear in source text
            all_tokens.sort(key=lambda x: x.begin)

        return all_tokens

    def get_extent_token_ids(self, **kwargs):
        tokens = self.get_extent_tokens(**kwargs)
        return [f"{self.document_title.split('_')[0]}_{t.index}" for t in tokens]

    def get_extent_text(self, **kwargs):
        return " ".join(t.text for t in self.get_extent_tokens(**kwargs))


@dataclass
class Token(Element):
    index: int
    event_extent: List[Event] = field(repr=False)
    participant_extent: List[Participant] = field(repr=False)
    filler_extent: List[Filler] = field(repr=False)
    canonical_referent_extent: List[CanonicalReferent] = field(repr=False)
    discontiguous_trigger_extent: List[DiscontiguousTrigger] = field(repr=False)
    sentiment_expression_extent: List[SentimentExpression] = field(repr=False)
    target_extent: List[Participant] = field(repr=False)

    def get_token_id(self):
        """
        Set token id based on document id + token position in text
        :return:
        """
        return f"{self.document_title}_{self.index}"

    def __hash__(self):
        return hash((self.element_id, self.text, self.begin, self.end))

    def __str__(self):
        return self.text


class SourceDocument:
    # TODO Finish this so NLP processing is done on only this shared object and annotation_documents are held in AnnotionDocument
    def __init__(self, annotation_documents):
        self.title = next(annotation_documents).title
        self.text = next(annotation_documents).text
        self.annotations = List[AnnotationDocument]


class AnnotationDocument:
    """
    Class to represent a WebAnno Annotation in the XMI format

    Arguments:
        file_name - name of the XMI file which contains the annotation

    Variables:
        text - textual representation of the annotated text
        tagset - attributes used to describe a frame
        sentences - list of sentences this document consists of
    """

    def __init__(self, xmi_content, path="", *args, **kwargs):

        self.path = path
        self.annotator_id = self.path.split("/")[-1].replace(".xmi", "")
        self.file_id = self.path.split("/")[-2]

        print(f"Parsing doc {self.path}")
        dom = md.parseString(xmi_content)
        self.text = dom.getElementsByTagName("cas:Sofa")[0].getAttribute("sofaString")
        self.title = dom.getElementsByTagName("type2:DocumentMetaData")[0].getAttribute(
            "documentTitle"
        )
        self.document_id = self.title.split("_")[0]

        self.events = None
        self.fillers = None
        self.discontiguous_triggers = None
        self.canonical_referents = None
        self.participants = None
        self.sentences = None
        self.tokens = None

        # stationary views of the XMI items
        sentences_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("type4:Sentence")
        ]
        tokens_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("type4:Token")
        ]
        events_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:A_Event")
        ]
        coref_event_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:CorefEvent")
        ]
        participant_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:B_Participant")
        ]
        participantlink_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:A_EventC_ParticipantLink")
        ]
        pronomcanonref_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:PronomCanonRef")
        ]
        filler_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:C_FILLER")
        ]
        fillerlink_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:A_EventD_FILLERLink")
        ]
        discontiguous_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:D_Discontiguous")
        ]
        discontiguouslink_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:A_EventF_DiscontiguousLink")
        ]
        sentiment_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:E_Sentiment")
        ]
        targetentity_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:E_SentimentA_TargetLink")
        ]
        targeteventlink_xmidata = [
            self.__convertAttributes__(node)
            for node in dom.getElementsByTagName("custom:E_SentimentB_TargetEventLink")
        ]

        # Create canonical referents objects
        canonical_referents = []
        for pcr in pronomcanonref_xmidata:
            canonical_referents.append(
                CanonicalReferent(
                    *self.__extract_default_xmi(pcr),
                    self.annotator_id,
                    self.title,
                    self,
                    pcr["Governor"],
                    pcr["Dependent"],
                )
            )
        if canonical_referents:
            self.canonical_referents = canonical_referents

        # Create participant objects from links, get element
        participants = []
        for part in participant_xmidata:
            text, begin, end, element_id = self.__extract_default_xmi(part)
            # parse all links that refer to the participants
            # parse participant_links which link to roles in events
            participant_links = list(
                filter(
                    lambda part_link: int(part_link["target"]) == element_id,
                    participantlink_xmidata,
                )
            )

            # parse canon referent
            if self.canonical_referents:
                canonref = list(
                    filter(
                        lambda canonref: str(canonref.pronom_id) == str(element_id),
                        self.canonical_referents,
                    )
                )
            else:
                canonref = None
            # participant spans can have multiple roles of different events but if canonical referent the participant can have no role
            for link in participant_links:
                role = link["role"] if link else None
                link_id = int(link["xmi:id"]) if link else None

                participants.append(
                    Participant(
                        text,
                        begin,
                        end,
                        element_id,
                        self.annotator_id,
                        self.title,
                        self,
                        role,
                        canonref,
                        link_id,
                    )
                )
        if participants:
            self.participants = participants

        # Create Filler objects
        fillers = []
        for fill in filler_xmidata:
            text, begin, end, element_id = self.__extract_default_xmi(fill)
            participant_links = list(
                filter(
                    lambda fill_link: int(fill_link["target"]) == element_id,
                    fillerlink_xmidata,
                )
            )
            # fill have 1 role each but if canonical referent the participant can have no role
            for link in participant_links:
                role = link["role"] if link else None
                link_id = int(link["xmi:id"]) if link else None

                fillers.append(
                    Filler(
                        text,
                        begin,
                        end,
                        element_id,
                        self.annotator_id,
                        self.title,
                        self,
                        role,
                        link_id,
                    )
                )
        if fillers:
            self.fillers = fillers

        # parse discontiguous trigger objects
        discontiguous_triggers = []
        for discont in discontiguous_xmidata:
            text, begin, end, element_id = self.__extract_default_xmi(discont)
            link = list(
                filter(
                    lambda discont_link: int(discont_link["target"]) == element_id,
                    discontiguouslink_xmidata,
                )
            )
            link_id = int(link[0]["xmi:id"]) if link else None

            discontiguous_triggers.append(
                DiscontiguousTrigger(
                    text,
                    begin,
                    end,
                    element_id,
                    self.annotator_id,
                    self.title,
                    self,
                    link_id,
                )
            )
        if discontiguous_triggers:
            self.discontiguous_triggers = discontiguous_triggers

        # Create events objects
        events = []
        for event in events_xmidata:

            text, begin, end, element_id = self.__extract_default_xmi(event)
            event_type = event.get("a_Type", None)
            event_subtype = event.get("b_Subtype", None)
            event_fulltype = f"{event_type}.{event_subtype}"

            # match participant by the link_id in the c_Participant feature
            participant_link_id = event.get("c_Participant", None)
            if participant_link_id:  # some events have no participants
                participant_link_id = [int(x) for x in participant_link_id.split()]
                participants = list(
                    filter(
                        lambda p: p.link_id in participant_link_id, self.participants
                    )
                )
            else:
                participants = None

            # match fillers
            filler_link_id = event.get("d_FILLER", None)
            if filler_link_id:  # some events have no fillers
                filler_link_id = [int(x) for x in filler_link_id.split()]
                fillers = list(
                    filter(lambda f: f.link_id in filler_link_id, self.fillers)
                )
            else:
                fillers = None

            # match discontiguous triggers
            discont_link_id = event.get("f_Discontiguous", None)
            if discont_link_id:  # most events have no disconts
                discont_link_id = [int(x) for x in discont_link_id.split()]
                discontiguous_triggers = list(
                    filter(
                        lambda d: d.link_id in discont_link_id,
                        self.discontiguous_triggers,
                    )
                )
            else:
                discontiguous_triggers = None

            # parse factuality
            othermodality = event.get("e_OtherModalityFactuality", None)
            if othermodality == "false":
                modality = "certain"
            elif othermodality == "true":
                modality = "other"
            else:
                modality = othermodality

            negativepolarity = event.get("f_NegativePolarityFactuality", None)
            if negativepolarity == "false":
                polarity_negation = "positive"
            elif negativepolarity == "true":
                polarity_negation = "negative"
            else:
                polarity_negation = negativepolarity

            if (
                polarity_negation == "positive" and modality == "certain"
            ):  # not the same as Liu
                realis = "asserted"
            else:
                realis = "other"

            # parse sentiment polarity
            polarity_sentiment = event.get("i_Polarity", None)
            if polarity_sentiment == "=Neutral":
                polarity_sentiment = "neutral"
            elif polarity_sentiment == "+Positive":
                polarity_sentiment = "positive"
            elif polarity_sentiment == "-Negative":
                polarity_sentiment = "negative"
            # set polarity when negated
            if polarity_sentiment == "positive" and polarity_negation == "negative":
                polarity_sentiment_scoped = "negative"
            elif polarity_sentiment == "negative" and polarity_negation == "negative":
                polarity_sentiment_scoped = "positive"
            else:  # for neutral or not negated
                polarity_sentiment_scoped = polarity_sentiment

            events.append(
                Event(
                    text,
                    begin,
                    end,
                    element_id,
                    self.annotator_id,
                    self.title,
                    self,
                    event_type,
                    event_subtype,
                    event_fulltype,
                    discontiguous_triggers,
                    participants,
                    fillers,
                    polarity_negation,
                    modality,
                    realis,
                    polarity_sentiment,
                    polarity_sentiment_scoped,
                )
            )
        if events:
            self.events = events

        # match coreferent events
        self.coreferent_event_xmidata = coref_event_xmidata
        for coref in coref_event_xmidata:
            coref_from_id = int(coref["Governor"])
            coref_to_id = int(coref["Dependent"])
            from_event = next(
                filter(lambda ev: ev.element_id == coref_from_id, self.events)
            )
            to_event = next(
                filter(lambda ev: ev.element_id == coref_to_id, self.events)
            )
            if from_event.coreferents is not None:
                if to_event not in from_event.coreferents:
                    from_event.coreferents.append(to_event)
            else:
                from_event.coreferents = [to_event]
            if to_event.coreferents is not None:
                if from_event not in to_event.coreferents:
                    to_event.coreferents.append(from_event)
            else:
                to_event.coreferents = [from_event]

        # resolve coordinated events: coordinated events are in the exact same begin and end position
        from operator import attrgetter

        if self.events:
            position_key = attrgetter("begin", "end")
            evs = sorted(self.events, key=position_key)
            coordinated = [
                g
                for g in [list(g) for k, g in groupby(evs, position_key)]
                if len(g) > 1
            ]
            for coord_event_group in coordinated:
                for event in coord_event_group:
                    event.coordination = coord_event_group

        # create Sentiment Expression objects
        self.sentiment_expressions = []
        self.targets = []
        for sent in sentiment_xmidata:

            (
                sent_text,
                sent_begin,
                sent_end,
                sent_element_id,
            ) = self.__extract_default_xmi(sent)
            sent_polarity = sent.get("e_Polarity", None)
            sent_uncertain = sent.get("c_Uncertain", None)
            sent_negation = sent.get("d_Negated", None)

            # parse sentiment polarity
            if sent_polarity == "=Neutral":
                sent_polarity = "neutral"
            elif sent_polarity == "+Positive":
                sent_polarity = "positive"
            elif sent_polarity == "-Negative":
                sent_polarity = "negative"
            # flip polarity when negated
            if sent_polarity == "positive" and sent_negation == "negative":
                sent_polarity_scoped = "negative"
            elif sent_polarity == "negative" and sent_negation == "negative":
                sent_polarity_scoped = "positive"
            else:  # keep same for neutral or not negated
                sent_polarity_scoped = sent_polarity

            # create Target Links both Entity and Event targets
            targets = []
            # parse sentiment target links which link participants to sentiment as a target
            targetentity_link_id = sent.get(
                "a_Target"
            ).split()  # split is needed because multiple target ids are space seperated
            for tentlink_id in targetentity_link_id:
                targetentity_link = next(
                    (
                        tlink
                        for tlink in targetentity_xmidata
                        if tlink["xmi:id"] == tentlink_id
                    ),
                    None,
                )
                target_entity_id = targetentity_link["target"]
                try:
                    target_participant = next(
                        p
                        for p in self.participants
                        if str(p.element_id) == target_entity_id
                    )
                except StopIteration:
                    # target is not found in already parsed Event Participants, it is a new Participant annotation
                    part_xmi = next(
                        pxmi
                        for pxmi in participant_xmidata
                        if pxmi["xmi:id"] == target_entity_id
                    )
                    text, begin, end, element_id = self.__extract_default_xmi(part_xmi)
                    target_participant = Participant(
                        text,
                        begin,
                        end,
                        element_id,
                        self.annotator_id,
                        self.title,
                        self,
                        "sentiment_target",
                        None,
                        None,
                    )
                    pass
                targets.append(target_participant)
            # parse event targets
            targetevent_link_id = sent.get("b_TargetEvent").split()
            for tevlink_id in targetevent_link_id:
                targetevent_link = next(
                    (
                        tlink
                        for tlink in targeteventlink_xmidata
                        if tlink["xmi:id"] == tevlink_id
                    ),
                    None,
                )
                target_event_id = targetevent_link["target"]
                target_event = next(
                    ev for ev in self.events if str(ev.element_id) == target_event_id
                )
                targets.append(target_event)

            # make Sentiment Expression
            se = SentimentExpression(
                sent_text,
                sent_begin,
                sent_end,
                sent_element_id,
                self.annotator_id,
                self.title,
                self,
                sent_polarity,
                sent_polarity_scoped,
                sent_uncertain,
                sent_negation,
                targets,
            )
            self.sentiment_expressions.append(se)
            self.targets.extend(targets)

        # create token object
        self.tokens = []
        for i, tok in enumerate(tokens_xmidata):
            text, begin, end, element_id = self.__extract_default_xmi(tok)

            extent_args = []
            extent_attribs = [
                "events",
                "participants",
                "fillers",
                "canonical_referents",
                "discontiguous_triggers",
                "sentiment_expressions",
                "targets",
            ]

            for attrib in extent_attribs:
                attrib_val = getattr(self, attrib)
                if attrib_val:
                    extent = [
                        x for x in attrib_val if begin >= x.begin and end <= x.end
                    ]
                    if not extent:
                        extent = None  # set empty result to None
                else:
                    extent = None
                extent_args.append(extent)

            token = Token(
                text,
                begin,
                end,
                element_id,
                self.annotator_id,
                self.title,
                self,
                i,
                *extent_args,
            )
            self.tokens.append(token)
            for ann_unit in extent_args:  # match tokens to annotation units
                if ann_unit is not None:
                    for au in ann_unit:
                        if au.tokens is not None:
                            au.tokens.append(token)
                        else:
                            au.tokens = [token]

        # Create sentence objects
        self.sentences = []
        for i, sentence in enumerate(sentences_xmidata):
            begin = int(sentence["begin"])
            end = int(sentence["end"])

            units_in_sentence = {
                "tokens": [],
                "events": [],
                "sentiment_expressions": [],
                "participants": [],
                "fillers": [],
                "canonical_referents": [],
                "targets": [],
            }

            for k in units_in_sentence:
                if getattr(self, k) is not None:  # skip if None
                    units_in_sentence[k] = [
                        u for u in getattr(self, k) if u.begin >= begin and u.end <= end
                    ]

            sentence = Sentence(i, begin, end, *units_in_sentence.values())

            # append the sentence on in_sentence attrib of events, sentiment_expressions, fillers, arguments in doc
            for k, units in units_in_sentence.items():
                if k != "tokens":
                    for u in units:
                        if not hasattr(u, "in_sentence"):
                            u.in_sentence = [sentence]
                        else:
                            if (
                                u.in_sentence is not None
                                and sentence not in u.in_sentence
                            ):
                                u.in_sentence.append(sentence)
                            else:
                                u.in_sentence = [sentence]
                else:
                    # set sentence_index on tokens for position in sentence and set in_sentence
                    for i, token in enumerate(units):
                        token.index_sentence = i
                        token.in_sentence = sentence

            self.sentences.append(sentence)  # add sentence to document

        if self.events and self.sentences and self.tokens:
            print(
                f"Parsed {len(self.events)} events in {len(self.sentences)} sentences ({len(self.tokens)} tokens)."
            )
        else:
            print(
                f"Warning: Empty document: Either no events or sentences or tokens in document."
            )

        # add in_event and in_sentiment_expression on participants:
        def append_attribute(myobj, attrib_k, val):
            """
            Append value to attribute list, create the attribute list if does not exist.
            """
            vals = getattr(myobj, attrib_k, [])
            if val not in vals:
                vals.append(val)
            setattr(myobj, attrib_k, vals)

        for ev in self.events:
            if ev.participants:
                for p in ev.participants:
                    append_attribute(p, "in_event", ev)
            if ev.fillers:
                for f in ev.fillers:
                    append_attribute(f, "in_event", ev)

        for se in self.sentiment_expressions:
            if se.targets:
                for t in se.targets:
                    append_attribute(t, "in_sentiment_expression", se)

    @staticmethod
    def __convertAttributes__(xml_source):
        """
        Function to converts XML attributes into a dictionary

        Arguments:
            xmlSource - XML attributes which are to be converted

        Returns:
            - dictionary view of the XML attributes
        """
        attributes = {}
        for attrName, attrValue in xml_source.attributes.items():
            attributes[attrName] = attrValue
        return attributes

    def __extract_default_xmi(self, xmidata):
        """
        Parses
        :param xmidata: A parsed xmi element as a python dict.
        :type xmidata: dict
        :return: text, begin, end, element_id
        :rtype: str, int, int, int
        """
        begin = int(xmidata["begin"])
        end = int(xmidata["end"])
        return self.text[begin:end], begin, end, int(xmidata["xmi:id"])

    def __str__(self):
        return f"Document {self.title} by {self.annotator_id}"

    def __repr__(self):
        """
        Custom repr for easier debugging.
        :return: repr string
        """
        return f"{self.document_id} {self.annotator_id}"


class WebannoProject:
    def __init__(self, project_dir, format="xmi"):

        # check format TODO add tsv3 support and calls
        if format not in ["xmi", "from_documents", "zip"]:
            raise ValueError(
                'The only project format is zipped or unzipped (UIMA XMI CAS) "xmi".'
            )
        else:
            self.format = format

        # check if project_dir is a valid WebAnno XMI export project
        # # check if it is str or Path object and resolve relative to absolute
        if isinstance(project_dir, os.PathLike) and project_dir.is_dir():
            dir_path = os.fspath(project_dir)
        elif isinstance(project_dir, str):
            # No, it's a directory name
            dir_path = Path(project_dir)
            if not dir_path.is_dir():
                raise TypeError(f"{project_dir} is not a directory.")
            dir_path = os.fspath(dir_path)
        else:
            raise TypeError(f"{project_dir} is not a directory.")

        self.project_dir = dir_path

        # set annotation dir TODO add Curation output support
        annotation_dir = Path(self.project_dir) / "annotation"
        if annotation_dir.is_dir():
            self.annotation_dir = annotation_dir
        else:
            self.annotation_dir = None

        with open(
            next(Path(self.project_dir).glob("./exportedproject*json")), "r"
        ) as project_meta_in:
            self.project_metadata = json.load(project_meta_in)

        self.annotation_document_fps = self._get_annotation_document_fps()
        self.annotation_documents = None
        self.source_documents = None
        self.typesystem = self._get_typesystem()

    def _get_typesystem(self):

        if not self._walked_documents:
            self._get_annotation_document_fps()

        # get first doc because typesystem is identical for all docs
        doc_fp = Path(self.annotation_document_fps[0])
        zip_fp = doc_fp.parent

        zfile = ZipFile(zip_fp, "r")
        for fn in zfile.namelist():
            if fn == "typesystem.xml":
                self.typesystem_fp = str(zip_fp / fn)
                self.typesystem = zfile.read(fn)

    def _get_annotation_document_fps(self):

        annotation_document_titles = [
            fp.name for fp in Path(self.annotation_dir).iterdir()
        ]
        annotation_document_titles.sort()

        zipfile_fps = flatten(
            [
                (Path(self.annotation_dir) / Path(docn)).glob("./*zip")
                for docn in annotation_document_titles
            ]
        )
        annotation_document_fps = []
        for zip_fp in zipfile_fps:
            for fn in ZipFile(zip_fp, "r").namelist():
                if fnmatch.fnmatch(fn, "*xmi"):
                    annotation_document_fps.append(str(zip_fp / fn))

        self._walked_documents = True

        return annotation_document_fps

    def _unzip_content(self, fp):
        fp = Path(fp)
        name = fp.name
        zip_fp = fp.parent

        with ZipFile(zip_fp, "r") as z:
            content = z.read(name)

        return content

    def parse_annotation_project(self, multiproc=True):

        # unzip and parse the documents
        print(f"Parsing {len(self.annotation_document_fps)} documents.")

        if multiproc:
            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                annotation_documents = pool.map(
                    self._parse_doc_from_zip, self.annotation_document_fps
                )
        else:
            annotation_documents = []
            for ann_doc_fp in self.annotation_document_fps:
                annotation_documents.append(self._parse_doc_from_zip(ann_doc_fp))
        if annotation_documents:
            self.annotation_documents = annotation_documents
            # source_documents = [] # todo create source documents
            # for title, docs in itertools.groupby(self.annotation_documents, lambda x: x.title):
            #     source_documents.append(SourceDocument(docs))

    def _parse_doc_from_zip(self, ann_doc_fp):
        xmicontent = self._unzip_content(ann_doc_fp)
        return AnnotationDocument(xmicontent, path=ann_doc_fp)

    # def _parse_cas_from_zip(self, ann_doc_fp):
    #     # this is much slower
    #
    #     typesystem_fp = Path(self.typesystem_fp)
    #     typesystem_filename = typesystem_fp.name
    #     typesystem_zip_fp = typesystem_fp.parent
    #
    #     with ZipFile(typesystem_zip_fp) as z:
    #         with z.open(typesystem_filename) as f:
    #             typesystem = cassis.load_typesystem(f)
    #
    #     ann_doc_fp = Path(ann_doc_fp)
    #     ann_doc_filename = ann_doc_fp.name
    #     ann_doc_zip_fp = ann_doc_fp.parent
    #
    #     with ZipFile(ann_doc_zip_fp) as z:
    #         with z.open(ann_doc_filename) as f:
    #             cas = cassis.load_cas_from_xmi(f, typesystem=typesystem)
    #
    #     events = [e for e in cas.select('webanno.custom.A_Event')]
    #     print(len(events))

    # def dump_pickle(self, fp):
    #     '''
    #     We use dill as a pickling/deserialization library as a more robust alternative to pickle.
    #     :param fp: output path
    #     '''
    #     with open(fp, "wb") as self_out:
    #         dill.dump(self, self_out)

    def get_events(self, subset=""):
        '''
        Return all events in project or in subset of project.
        :param subset: "dev" for events in devset "test" for test set events.
        :return:
        '''
        if subset.lower() == "dev":
            docs = self.dev
        elif subset.lower() == "test":
            docs = self.test
        else:
            docs = self.annotation_documents
        for doc in docs:
            for ev in doc.events:
                yield ev

    def get_arguments(self, subset=None):
        if subset.lower() == "dev":
            docs = self.dev
        elif subset.lower() == "test":
            docs = self.test
        else:
            docs = self.annotation_documents
        for doc in docs:
            for arg in doc.participants + doc.fillers:
                yield arg

    def get_sentiment_expressions(self):
        for doc in self.annotation_documents:
            for se in doc.sentiment_expressions:
                yield se

    def get_annotation_from_documents(self, anno_attrib_name):
        for doc in self.annotation_documents:
            for val in getattr(doc, anno_attrib_name):
                yield val

    def clean_duplicate_documents(self):
        """
        Clean duplicate docs that are a consequence of opening them in WebAnno.
        :param docs:
        :return:
        """
        title_k = lambda x: x.title
        for k, g in groupby(sorted(self.annotation_documents, key=title_k), title_k):
            g = list(g)
            if len(g) > 1:
                # check first if one is in test set
                to_remove = [x for x in g if x not in self.test]
                if (
                    len(to_remove) > 1
                ):  # if test is not matched, make subselection based on annotation unit count
                    select_k = lambda x: (
                        len(x.events) + len(x.sentiment_expressions),
                        x.annotator_id != "gilles",
                    )
                    to_remove.sort(key=select_k, reverse=True)
                    to_remove = to_remove[1:]
                for docrm in to_remove:
                    self.annotation_documents.remove(docrm)
                    if docrm in self.dev:
                        self.dev.remove(docrm)
                    elif docrm in self.test:
                        self.test.remove(docrm)
                    print(f"Duplicate doc removed: {docrm}")

    def process_spacy(self):
        """
        Run the Spacy processing pipeline on the annotation documents.
        Add processed docs so they can be accessed.
        :return: adds .nlp to each doc
        """

        def prevent_sentence_boundary_detection(doc):
            for token in doc:
                # This will entirely disable spaCy's sentence detection
                token.is_sent_start = False
            return doc

        def process_sentence(sen_tokens):
            doc = spacy.tokens.Doc(nlp.vocab, words=sen_tokens)
            tagger(doc)
            prevent_sbd(doc)
            ner(doc)
            parser(doc)
            return doc

        # setup spacy nlp pipeline
        nlp = spacy.load("en_core_web_lg")
        parser = nlp.get_pipe("parser")
        nlp.add_pipe(
            prevent_sentence_boundary_detection, name="prevent-sbd", before="parser"
        )

        tagger = nlp.get_pipe("tagger")
        prevent_sbd = nlp.get_pipe("prevent-sbd")
        parser = nlp.get_pipe("parser")
        ner = nlp.get_pipe("ner")

        for doc in self.annotation_documents:
            doc.sentences_processed = []
            for sen in doc.sentences:
                sen_tokens = [t.text for t in sen.tokens]
                sen_proc = process_sentence(sen_tokens)
                # add processed sentence to doc
                doc.sentences_processed.append(sen_proc)

            print(f"Processed with Spacy: {doc.document_id}")

    def replace_canonical_referents(self, **kwargs):
        """
        Replaces all participant arguments in project with canonical referent links.
        :return:
        """
        replaced_count = 0
        for doc in self.annotation_documents:
            for sen in doc.sentences:
                for part in sen.participants:
                    replaced_count += part.replace_by_canonical_referent(**kwargs) #this will replace in-place in project, but returns a count too.

        return replaced_count

def parse_main_iaa(main_dirp, iaa_dirp, opt_fp):
    """
    Parses the main corpus and IAA gold standard files and joins them in one WebAnnoProject.
    :return:
    """

    # moderator_id = "gilles"
    # exclude_moderator = lambda x: moderator_id not in Path(x.path).stem
    # include_moderator = lambda x: moderator_id in Path(x.path).stem

    main_project = WebannoProject(main_dirp)
    # # exclude moderator and trial files which start with two digits
    # main_anndocs_final = [p for p in main_project.annotation_document_fps
    #                       if moderator_id not in Path(p).stem and not Path(p).parents[1].stem[0:1].isdigit()]
    #
    # iaa_project = WebannoProject(iaa_dirp)
    # # exclude all annotators except moderator and trial files which start with two digits
    # iaa_anndocs_final = [p for p in iaa_project.annotation_document_fps
    #                      if moderator_id in Path(p).stem and not Path(p).parents[1].stem[0:1].isdigit()]

    main_project.annotation_document_fps.extend(main_project.annotation_document_fps)

    main_project.parse_annotation_project()

    main_project.dump_pickle(opt_fp)
    print(f"Written project object pickle to {opt_fp}")


if __name__ == "__main__":

    parse_and_pickle(settings.IAA_XMI_DIRP, settings.IAA_PARSER_OPT)
    # parse_and_pickle(settings.MAIN_XMI_DIRP, settings.MAIN_PARSER_OPT)

    # # ANNOTATION_DIRP = "../example_data"
    # project_dirp = "/home/gilles/00-sentivent-fwosb-phd-2017-2020/00-event-annotation/webanno-project-export/XMI-corrected-SENTiVENT-event-english-1_2019-01-28_1341"
    # opt_dirp = "sentivent_en_webanno_correction.pickle"
    # exclude_gilles = lambda x: "anno" in Path(x.path).stem
    #
    # iaa_project = WebannoProject(project_dirp)
    # iaa_project.parse_annotation_project()
    # # iaa_project.process_spacy()
    # iaa_project.dump_pickle(opt_dirp)
