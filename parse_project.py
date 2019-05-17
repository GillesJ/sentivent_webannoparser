#!/usr/bin/env python3
'''
parse_project.py
webannoparser 
5/17/19
Copyright (c) Gilles Jacobs. All rights reserved.

Calling script to parse a webanno project.
'''
from parser import WebannoProject
import util
import settings

def parse_and_pickle(project_dirp, opt_fp):

    project = WebannoProject(project_dirp)
    project.parse_annotation_project()
    util.pickle_webanno_project(project, opt_fp)

if __name__ == "__main__":

    parse_and_pickle(settings.IAA_XMI_DIRP, settings.IAA_PARSER_OPT)