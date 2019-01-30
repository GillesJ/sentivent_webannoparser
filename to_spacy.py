#!/usr/bin/env python3
'''
to_spacy.py
webannoparser 
10/17/18
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import spacy
import pickle
from parser import *
from itertools import zip_longest


if __name__ == "__main__":

    FROM_SCRATCH = True
    nlp = spacy.load('en_core_web_lg')
    project_fp = "sentivent_en_webanno_project_my_obj.pickle"
    # opt_fp = "spacy_" + project_fp

    # if not Path(opt_fp).is_file() or FROM_SCRATCH:
    #
    #     with open(project_fp, "rb") as project_in:
    #         project = pickle.load(project_in)
    #
    #     project.spacy_documents = []
    #     for doc in project.documents:
    #         sp_doc = nlp(doc.text)
    #         setattr(sp_doc, "title", doc.title)
    #         project.spacy_documents.append(sp_doc)
    #
    #     project.dump_pickle(opt_fp)
    #
    # else:
    with open(project_fp, "rb") as project_in:
        project = pickle.load(project_in)

    newline_filter = lambda x: x.text!=u"\n"

    for i, doc in enumerate(project.spacy_documents[:2]):
        for j, (doc_tok, spac_tok) in enumerate(zip_longest(project.documents[i].tokens, filter(newline_filter, doc))):
            # print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)
            toktxt = doc_tok.text if doc_tok is not None else "NO TOKEN"
            sptxt = spac_tok.text if spac_tok is not None else "NO TOKEN"
            print(j, toktxt, sptxt)
        #
        # # svg = spacy.displacy.render(list(doc.sents), style='dep', options={"compact": True})
        # html = spacy.displacy.render(list(doc.sents), style='dep', options={"compact": True}, page=True)
        # file_name = project.documents[i].annotator_id + "_" + project.documents[i].title.replace('.txt', '.html')
        # output_path = Path('./dep_images/' + file_name)
        # output_path.parent.mkdir(parents=True, exist_ok=True)
        # output_path.open('w', encoding='utf-8').write(html)