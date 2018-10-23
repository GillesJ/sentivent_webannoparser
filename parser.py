#!/usr/bin/env python3
"""
TODO: create a LinkFeature superclass for parsing the linked features to avoid code duplication and increase efficiency
TODO: refactor current Document class into a Document class and an Annotation class:
        Document contains the text, sentences, tokens, metadata and annotations.
        Annotation contains all annotations and the parsing code.
    This allows for less duplication (text, tokens, sentences and metadata are all the same for the same doc).
TODO: write generic parser for webanno features (Link, Slot, etc):
        cassis is slower than this prototype but more general
parser.py
sentivent-webannoparser
10/4/18
Copyright (c) Gilles Jacobs. All rights reserved.
"""
from dataclasses import dataclass, field
from zipfile import ZipFile
import xml.dom.minidom as md
import cassis
import fnmatch
import pickle
import spacy
from pathlib import Path
from typing import List
from util import flatten, count_avg
import os
import json

@dataclass
class Element:
    text: str
    begin: int = field(repr=False)
    end: int = field(repr=False)
    element_id: int = field(repr=False)

@dataclass
class Filler(Element):
    role: str
    link_id: int = field(repr=False)

@dataclass
class DiscontiguousTrigger(Element):
    link_id: int = field(repr=False)

@dataclass
class CanonicalReferent(Element):
    pronom_id: int = field(repr=False)
    referent_id: int

@dataclass
class Participant(Element):
    role: str
    canonical_referents: List[CanonicalReferent] = field(repr=False)
    link_id: int = field(repr=False)

@dataclass
class Event(Element):
    event_type: str
    event_subtype: str
    discontiguous_triggers: List[DiscontiguousTrigger] = field(repr=False)
    participants: List[Participant] = field(repr=False)
    fillers: List[Filler] = field(repr=False)
    polarity: str = field(repr=False)
    modality: str = field(repr=False)

@dataclass
class Token(Element):
    index: int
    event_extent: List[Event] = field(repr=False)
    participant_extent: List[Participant] = field(repr=False)
    filler_extent: List[Filler] = field(repr=False)
    canonical_referent_extent: List[CanonicalReferent] = field(repr=False)
    discontiguous_trigger_extent: List[DiscontiguousTrigger] = field(repr=False)

class Document:
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
        self.text = dom.getElementsByTagName('cas:Sofa')[0].getAttribute('sofaString')
        self.title = dom.getElementsByTagName("type2:DocumentMetaData")[0].getAttribute("documentTitle")

        self.events = None
        self.fillers = None
        self.discontiguous_triggers = None
        self.canonical_referents = None
        self.participants = None
        self.sentences = None
        self.tokens = None

        # stationary views of the XMI items
        sentences_xmidata = [self.__convertAttributes__(node) for node in dom.getElementsByTagName('type4:Sentence')]
        tokens_xmidata = [self.__convertAttributes__(node) for node in dom.getElementsByTagName('type4:Token')]
        events_xmidata = [self.__convertAttributes__(node) for node in dom.getElementsByTagName('custom:A_Event')]
        coref_event_xmidata = [self.__convertAttributes__(node) for node in
                               dom.getElementsByTagName('custom:CorefEvent')]
        participant_xmidata = [self.__convertAttributes__(node) for node in
                               dom.getElementsByTagName('custom:B_Participant')]
        participantlink_xmidata = [self.__convertAttributes__(node) for node in
                                   dom.getElementsByTagName('custom:A_EventC_ParticipantLink')]
        pronomcanonref_xmidata = [self.__convertAttributes__(node) for node in
                                  dom.getElementsByTagName('custom:PronomCanonRef')]
        filler_xmidata = [self.__convertAttributes__(node) for node in dom.getElementsByTagName('custom:C_FILLER')]
        fillerlink_xmidata = [self.__convertAttributes__(node) for node in
                              dom.getElementsByTagName('custom:A_EventD_FILLERLink')]
        discontiguous_xmidata = [self.__convertAttributes__(node) for node in
                                 dom.getElementsByTagName('custom:D_Discontiguous')]
        discontiguouslink_xmidata = [self.__convertAttributes__(node) for node in
                                     dom.getElementsByTagName('custom:A_EventF_DiscontiguousLink')]

        # Create canonical referents objects
        canonical_referents = []
        for pcr in pronomcanonref_xmidata:
            canonical_referents.append(
                CanonicalReferent(*self.__extract_default_xmi(pcr), pcr["Governor"], pcr["Dependent"])
            )
        if canonical_referents: self.canonical_referents = canonical_referents

        # Create participant objects
        participants = []
        for part in participant_xmidata:
            text, begin, end, element_id = self.__extract_default_xmi(part)
            link = list(
                filter(lambda part_link: int(part_link['target']) == element_id, participantlink_xmidata))
            if self.canonical_referents:
                canonref = list(filter(lambda canonref: canonref.pronom_id == element_id, self.canonical_referents))
            else: canonref = None
            # participants have 1 role each but if canonical referent the participant can have no role
            role = link[0]["role"] if link else None
            link_id = int(link[0]["xmi:id"]) if link else None

            participants.append(
                Participant(text, begin, end, element_id, role, canonref, link_id)
            )
        if participants: self.participants = participants

        # Create Filler objects
        fillers = []
        for fill in filler_xmidata:
            text, begin, end, element_id = self.__extract_default_xmi(fill)
            link = list(filter(lambda fill_link: int(fill_link['target']) == element_id, fillerlink_xmidata))
            # fill have 1 role each but if canonical referent the participant can have no role
            role = link[0]["role"] if link else None
            link_id = int(link[0]["xmi:id"]) if link else None

            fillers.append(
                Filler(text, begin, end, element_id, role, link_id)
            )
        if fillers: self.fillers = fillers

        # parse discontiguous trigger objects
        discontiguous_triggers = []
        for discont in discontiguous_xmidata:
            text, begin, end, element_id = self.__extract_default_xmi(discont)
            link = list(filter(lambda discont_link: int(discont_link['target']) == element_id, discontiguouslink_xmidata))
            link_id = int(link[0]["xmi:id"]) if link else None

            discontiguous_triggers.append(
                DiscontiguousTrigger(text, begin, end, element_id, link_id)
            )
        if discontiguous_triggers: self.discontiguous_triggers = discontiguous_triggers

        # Create events objects
        events = []
        for event in events_xmidata:

                text, begin, end, element_id = self.__extract_default_xmi(event)
                event_type = event.get("a_Type", None)
                event_subtype = event.get("b_Subtype", None)

                # match participant by the link_id in the c_Participant feature
                participant_link_id = event.get("c_Participant", None)
                if participant_link_id:  # some events have no participants
                    participant_link_id = [int(x) for x in participant_link_id.split()]
                    participants = list(filter(lambda p: p.link_id in participant_link_id, self.participants))
                else:
                    participants = None

                # match fillers
                filler_link_id = event.get("d_FILLER", None)
                if filler_link_id:  # some events have no fillers
                    filler_link_id = [int(x) for x in filler_link_id.split()]
                    fillers = list(filter(lambda f: f.link_id in filler_link_id, self.fillers))
                else:
                    fillers = None

                # match discontiguous triggers
                discont_link_id = event.get("f_Discontiguous", None)
                if discont_link_id:  # most events have no disconts
                    discont_link_id = [int(x) for x in discont_link_id.split()]
                    discontiguous_triggers = list(
                        filter(lambda d: d.link_id in discont_link_id, self.discontiguous_triggers))
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
                    polarity = "positive"
                elif negativepolarity == "true":
                    polarity = "negative"
                else:
                    polarity = negativepolarity

                events.append(
                    Event(text, begin, end, element_id, event_type, event_subtype, discontiguous_triggers, participants,
                          fillers, polarity, modality)
                )
        if events: self.events = events

        # match coreferent events
        for coref in coref_event_xmidata:
            coref_from_id = int(coref["Governor"])
            coref_to_id = int(coref["Dependent"])
            from_event = list(filter(lambda ev: ev.element_id == coref_from_id, self.events))[0]
            to_event = list(filter(lambda ev: ev.element_id == coref_to_id, self.events))[0]
            if hasattr(from_event, "coreferents"):
                from_event.coreferents.append(to_event)
            else:
                from_event.coreferents = [to_event]

        # create token object
        self.tokens = []
        for i, tok in enumerate(tokens_xmidata):
            text, begin, end, element_id = self.__extract_default_xmi(tok)

            extent_args = []
            extent_attribs = ["events", "participants", "fillers", "canonical_referents", "discontiguous_triggers"]

            for attrib in extent_attribs:
                attrib_val = getattr(self, attrib)
                if attrib_val:
                    extent = [x for x in attrib_val if begin >= x.begin and end <= x.end]
                    if not extent: extent = None # set empty result to None
                else:
                    extent = None
                extent_args.append(extent)

            self.tokens.append(
                Token(text, begin, end, element_id, i, *extent_args)
            )

        # Create sentence objects
        self.sentences = []
        for i, sentence in enumerate(sentences_xmidata):
            begin = int(sentence["begin"])
            end = int(sentence["end"])
            if self.tokens:
                tokens = [token for token in self.tokens if token.begin >= begin and token.end <= end]
            if self.events:
                sentence_events = [event for event in self.events if event.begin >= begin and event.end <= end]
                self.sentences.append(Sentence(i, begin, end, tokens, sentence_events))

        if self.events and self.sentences and self.tokens:
            print(f"Parsed {len(self.events)} events in {len(self.sentences)} sentences ({len(self.tokens)} tokens).")
        else: print(f"Warning: Empty document: No events or sentences or tokens in document.")

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

class Sentence:
    """
    Class to represent a sentence

    Variables:
        id - number of this sentence in the text
        begin - position of the sentence start in the original text
        end - position of the sentecne end in the original text
        frames - list of frames contained in this sentence
    """

    def __init__(self, element_id, begin, end, tokens, events):
        self.element_id = element_id
        self.begin = begin
        self.end = end
        self.tokens = tokens
        self.events = events

    def __eq__(self, other):
        if self.begin == other.begin and self.end == other.end:
            if [x for x in self.events if not x in other.events] == [] and [x for x in other.events if
                                                                            not x in self.events] == []:
                return True
        return False

class WebannoProject:

    def __init__(self, project_dir, format="xmi"):

        # check format TODO add tsv3 support and calls
        if format not in ["xmi", "tsv3"]:
            raise ValueError("arg format can only \"xmi\" and \"tsv3\"")
        else: self.format = format

        # check if project_dir is a valid WebAnno XMI export project
        # # check if it is str or Path object and resolve relative to absolute
        if isinstance(project_dir, os.PathLike) and project_dir.is_dir():
            dir_path = os.fspath(project_dir)
        elif isinstance(project_dir, str):
            # No, it's a directory name
            dir_path = Path(project_dir)
            if not dir_path.is_dir(): raise TypeError(f"{project_dir} is not a directory.")
            dir_path = os.fspath(dir_path)
        else: raise TypeError(f"{project_dir} is not a directory.")

        self.project_dir = dir_path

        # set annotation dir TODO add Curation output support
        annotation_dir = Path(self.project_dir) / "annotation"
        if annotation_dir.is_dir():
            self.annotation_dir = annotation_dir
        else:
            self.annotation_dir = None

        # TODO use the metadata for parsing and checking/enriching of document status
        with open(next(Path(self.project_dir).glob("./exportedproject*json")), 'r') as project_meta_in:
            self.project_metadata = json.load(project_meta_in)

        self.annotation_document_fps = self._get_annotation_document_fps()
        self.typesystem = self._get_typesystem()

    def _get_typesystem(self):

        if not self._walked_documents:
            self._get_annotation_document_fps()

        # get first doc because typesystem is identical for all docs
        doc_fp = Path(self.annotation_document_fps[0])
        zip_fp = doc_fp.parent

        zfile = ZipFile(zip_fp, 'r')
        for fn in zfile.namelist():
            if fn == "typesystem.xml":
                self.typesystem_fp = str(zip_fp / fn)
                self.typesystem = zfile.read(fn)

    def _get_annotation_document_fps(self):

        annotation_document_titles = [fp.name for fp in Path(self.annotation_dir).iterdir()]

        zipfile_fps = flatten([(Path(self.annotation_dir) / Path(docn)).glob("./*zip") for docn in annotation_document_titles])
        annotation_document_fps = []
        for zip_fp in zipfile_fps:
            for fn in ZipFile(zip_fp, 'r').namelist():
                if fnmatch.fnmatch(fn, "*xmi"):
                    annotation_document_fps.append(str(zip_fp / fn))

        self._walked_documents = True

        return annotation_document_fps

    def _unzip_content(self, fp):
        fp = Path(fp)
        name = fp.name
        zip_fp = fp.parent

        with ZipFile(zip_fp, 'r') as z:
            content = z.read(name)

        return content

    def parse_annotation_project(self):
        if not self._walked_documents:
            self._get_annotation_document_fps()

        # unzip and parse the documents
        print(f"Parsing {len(self.annotation_document_fps)} documents.")
        self.documents = []
        for ann_doc_fp in self.annotation_document_fps:
            self.documents.append(self._parse_doc_from_zip(ann_doc_fp))

    def _parse_doc_from_zip(self, ann_doc_fp):
        xmicontent = self._unzip_content(ann_doc_fp)
        return Document(xmicontent, path=ann_doc_fp)

    def _parse_cas_from_zip(self, ann_doc_fp):
        # this is much slower

        typesystem_fp = Path(self.typesystem_fp)
        typesystem_filename = typesystem_fp.name
        typesystem_zip_fp = typesystem_fp.parent

        with ZipFile(typesystem_zip_fp) as z:
            with z.open(typesystem_filename) as f:
                typesystem = cassis.load_typesystem(f)

        ann_doc_fp = Path(ann_doc_fp)
        ann_doc_filename = ann_doc_fp.name
        ann_doc_zip_fp = ann_doc_fp.parent

        with ZipFile(ann_doc_zip_fp) as z:
            with z.open(ann_doc_filename) as f:
                cas = cassis.load_cas_from_xmi(f, typesystem=typesystem)

        events = [e for e in cas.select('webanno.custom.A_Event')]
        print(len(events))

    def dump_pickle(self, fp):
        with open(fp, "wb") as self_out:
            pickle.dump(self, self_out)

    def process_spacy(self):


        nlp = spacy.load('en_core_web_lg')

        self.spacy_documents = []
        for doc in self.documents:
            print(f"{doc.title}: dep parsing, NER tagging, word vectorizing with Spacy.")
            self.spacy_documents.append(nlp(doc.text))
            # self.spacy_documents.append(spacy.tokens.Doc(nlp.vocab, words=[[t.text for t in doc.sentences]))
            # self.spacy_documents.append(spacy.tokens.Doc(nlp.vocab, doc.text))


if __name__ == "__main__":

    # ANNOTATION_DIRP = "../example_data"
    project_dirp = "exports/XMI_SENTiVENT-event-english-1_2018-10-04_1236"
    opt_fp = "sentivent_en_webanno_project_my_obj.pickle"
    exclude_gilles = lambda x: "anno" in Path(x.path).stem

    event_project = WebannoProject(project_dirp)
    event_project.parse_annotation_project()
    event_project.documents = list(filter(exclude_gilles, event_project.documents)) # filter empty invalid gilles docs
    event_project.process_spacy()
    event_project.dump_pickle(opt_fp)

    avg_attribs = ["events", "sentences", "tokens", "participants", "fillers"]
    avg = {avg_attrib: count_avg(event_project.documents, avg_attrib, return_counts=True) for avg_attrib in avg_attribs}
    print(avg)

    all_event_types = []
    all_event_subtypes = []
    for doc in event_project.documents:
        if doc.events:
            for ev in doc.events:
                all_event_types.append(ev.event_type)
                all_event_subtypes.append(f"{ev.event_type}.{ev.event_subtype}")

    from collections import Counter
    print("Event types: ", Counter(all_event_types))
    print("Event subtypes", Counter(all_event_subtypes))
