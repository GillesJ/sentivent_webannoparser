#!/usr/bin/env python3
'''
Parse a project to a WebAnno project object or to a clean corpus dict.
parse_project.py
webannoparser 
5/17/19
Copyright (c) Gilles Jacobs. All rights reserved.

Calling script to parse a webanno project.
'''
import sys
sys.path.append("/home/gilles/repos/")

from sentivent_webannoparser.parser import WebannoProject
import sentivent_webannoparser.util as util
import sentivent_webannoparser.settings as settings

def parse_and_pickle(project_dirp, opt_fp):

    project = WebannoProject(project_dirp)
    project.parse_annotation_project()
    util.pickle_webanno_project(project, opt_fp)

def combine_inspect_corpus(ia_dirp, main_dirp):
    '''
    Load both the silver standard (non-IA adjudicated) main project and gold standard adjudicated IA project.
    Clean the main project from empty and redundant docs to obtain the final corpus.
    Joins the final docs into a corpus dict
     {"gold": list(Documents), "silver": list(Documents)}.
     Gold-standard: adjudicated from IA study based off three annotators.
     Silver-standard: annotated by 1 annotator and corrected by judge.
    Corpus creation procedure is as follows:
    1. Set aside gold standard IAA documents.
    2. Remove IA docs from silver standard.
    3. Remove documents without annotations: these are an artifact of WebAnno monitoring.
    4. TODO check multiple annotated in silver standard
    :param ia_dirp: Directory path containing export of Interannotator Project.
    :param main_dirp: Directory path containing export of Interannotator Project.
    :param opt_fp: Output dirpath in which intermediate WebannoProject pickles will be put and final output.
    :return: {"gold": list(Documents), "silver": list(Documents)}.
    '''
    # load the gold standard IA project and silver standard
    ia_project = WebannoProject(ia_dirp)
    ia_project.parse_annotation_project(multiproc=settings.PARSER_MULTIPROC)

    main_project = WebannoProject(main_dirp)
    main_project.parse_annotation_project(multiproc=settings.PARSER_MULTIPROC)

    # Start cleaning corpus
    # 1. Set aside gold standard adjudicated docs
    corpus = {
        "gold": [doc for doc in ia_project.annotation_documents if is_gold_standard(doc)],
    }
    silver_unclean = [doc for doc in main_project.annotation_documents if doc.title not in settings.IA_IDS]

    # 2. Remove documents with no event annotations
    unclean_len = len(main_project.annotation_documents)
    silver_unclean = [d for d in silver_unclean if d.events]
    print(f"Removed {unclean_len - len(silver_unclean)} silver-standard docs without event annotations. {unclean_len} remaining.")

    from collections import Counter
    title_cnt = Counter(d.title for d in silver_unclean)
    title_cnt = {k: v for k, v in title_cnt.items() if v > 1} # remove singles because not problematic

    single_annot_docs = []
    silver_clean = []
    for title, docgroup in groupby(silver_unclean, key=lambda x: x.title):
        docs = list(docgroup)
        if len(docs) > 1:
            selected_doc = select_redundant_annotated_doc(docs, method="best_tuple_score")
            keep_docs.append(selected_doc)
        else:
            silver_unclean.append(docs[0])
            single_annot_docs.append(docs[0])

    print(f"Removed {clean_len - len(keep_docs)} duplicate annotated docs by keeping docs with most events.")
    proj.annotation_documents_clean = keep_docs
    proj.single_annotator_documents = single_annot_docs

    corpus = {
        "gold": [doc for doc in ia_project.annotation_documents if is_gold_standard(doc)],
    }

    return corpus

def parse_project(xmi_export_dirp):
    """
    Parse a project and return a corpus of annotation docs in dict
    {silver: [list of AnnotationDocument], gold: [list of AnnotationDocument]}
    :param xmi_export_dirp:
    """
    # Load project and parse docs
    project = WebannoProject(xmi_export_dirp)
    project.parse_annotation_project(multiproc=settings.PARSER_MULTIPROC)

    # Split into silver and gold-standard

    gold = []
    silver = []
    for doc in docs:
        if doc.title in settings.IA_IDS:
            if doc.annotator_id == settings.MOD_ID: # throw away any IAA study that is not gilles
                gold.append(doc)
        else:
            silver.append(doc)

    return {"silver": silver, "gold": gold}


# if __name__ == "__main__":

    # Create full clean corpus


    # corpus = combine_inspect_corpus(settings.IAA_XMI_DIRP, settings.MAIN_XMI_DIRP)
    # parse_and_pickle(settings.IAA_XMI_DIRP, settings.IAA_PARSER_OPT)