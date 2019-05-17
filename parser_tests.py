#!/usr/bin/env python3
'''
parser_tests.py
webannoparser 
5/17/19
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
from parser import *
import unittest
import util
import os
import settings

class CustomAssertions:

    def assertFileExists(self, path):
        if not os.path.lexists(path):
            raise AssertionError('File not exists in path "' + path + '".')

    def assertAllUnique(self, it):
        '''
        Assertion method for iterators to test the uniqueness of elements.
        Works with unhashable elements.
        '''
        seen = list()
        return not any(i in seen or seen.append(i) for i in it)


class TestWebannoProject(unittest.TestCase, CustomAssertions):

    @classmethod
    def setUpClass(cls):
        cls.project = util.unpickle_webanno_project(settings.IAA_PARSER_OPT)

    def test_extent_token_id_uniqueness(self):
        '''
        Test if the get_extent_token_id function returns unique tokens.
        :return:
        '''
        token_ids = []
        for d in self.project.annotation_documents:
            for ev in d.events:
                token_ids.extend(ev.get_extent_token_ids())
        self.assertAllUnique(token_ids)

if __name__ == '__main__':
    project = util.unpickle_webanno_project(settings.IAA_PARSER_OPT)
    # unittest.main()