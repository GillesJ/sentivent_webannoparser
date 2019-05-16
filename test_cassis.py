#!/usr/bin/env python3
'''
Simple parsing test https://github.com/dkpro/dkpro-cassis repo on Sentivent WebAnno project XMI CAS export.
The main branch of the git repo is automatically pulled and loaded as module using the module import_from_github_com.

'pip install import_from_github_com'
test_cassis.py
webannoparser 
1/14/19
'''
from github_com.dkpro.cassis import *
from pathlib import Path
from glob import glob


# def get_annotation_filenames(dirp):
#     for zip_fp in glob(f"{}")

xmi_export_dirp = "example_data/t04_as-states-test-waters-att-hopes-to-catch-all-with-firstnet-fi.txt/webanno1468212693444888492export" # unzipped dir of WebAnno annotation export
typesystem_fp = Path(xmi_export_dirp, "typesystem.xml")
casxml_fp = Path(glob(f"{xmi_export_dirp}/*.xmi")[0])

with open(typesystem_fp, 'rb') as f:
    typesystem = load_typesystem(f)

with open(casxml_fp, 'rb') as f:
   cas = load_cas_from_xmi(f, typesystem=typesystem)

for sentence in cas.select('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence'):
    for token in cas.select_covered('de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token', sentence):
        print(cas.get_covered_text(token))

        # Annotation values can be accessed as properties
        print('Token: begin={0}, end={1}'.format(token.begin, token.end))