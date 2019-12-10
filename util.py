#!/usr/bin/env python3
'''
util.py
sentivent_webannoparser
10/10/18
Copyright (c) Gilles Jacobs. All rights reserved.  
'''

import dill
from collections import abc, defaultdict
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def pickle_webanno_project(project_obj, pickle_fp):
    '''
    Serialize the parsed project to a binary pickle at pickle_fp.
    We use dill as a pickling library with the highest protocol level (=4)
    We enable pickling by class reference (class definition is included for portability).
    :param project_obj:
    :param pickle_fp:
    :return:
    '''
    '''
    :param project_obj:
    :param pickle_fp:
    '''

    with open(pickle_fp, "wb") as proj_out:
        dill.dump(project_obj, proj_out, byref=True, protocol=4, recurse=True)
    print(f"Written project object pickle to {pickle_fp}.")


def unpickle_webanno_project(pickle_fp):
    '''
    Deserialize the parsed project from a binary pickle at pickle_fp.
    We use dill as a pickling library with the highest protocol level (=4).
    :param pickle_fp:
    :return: webanno project object
    '''

    with open(pickle_fp, "rb") as proj_in:
        proj = dill.load(proj_in)
    print(f"Loaded project object pickle from {pickle_fp}.")

    return proj


def dict_zip(*dicts, fillvalue=None):
    all_keys = {k for d in dicts for k in d.keys()}
    return {k: [d.get(k, fillvalue) for d in dicts] for k in all_keys}

def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key,locs) for key,locs in tally.items()
                            if len(locs)>1)

def flatten(l):
    return [item for sublist in l for item in sublist]

def count_avg(iterable_obj, attr, return_counts=False):
    try:
        if not isinstance(iterable_obj, abc.Iterable): raise TypeError
        cnt = 0
        for x in iterable_obj:
            attrib_val = getattr(x, attr)
            if attrib_val: # only add to count if attribute value is not None and empty list
                cnt += len(attrib_val)
        avg = cnt / float(len(iterable_obj))
        if return_counts:
            return avg, cnt
        else:
            return avg
    except TypeError as te:
        print(f"{iterable_obj} is not an iterable. {str(te)}")

def rank_dataframe_column(df, **kwargs):
    '''
    Creates a rank column with numeric rank for each column in dataframe.
    :param df: df Pandas dataframe
    :return: df with rank column(s)
    '''
    for c in df:
        rank_name = c + "_rank"
        df[rank_name] = df[c].rank(**kwargs).astype(int)

    return df

def filter_stopwords(text, language="english"):
    '''
    Filters stopwords using nltk stopwords package.
    :param text: str with multiple tokens
    :param language: nltk language identifier
    :return: list of tokens without stopwords
    '''
    tokens = text.split()     # #simple tokenize because annotated docs are already tokenized with space inserted
    return " ".join(t for t in tokens if t.lower() not in set(stopwords.words(language)))

def stem_english(text):
    # simple tokenize because annotated docs are already tokenized with space inserted
    tokens_stemmed = [SnowballStemmer("english").stem(t) for t in text.split()]
    return " ".join(t for t in tokens_stemmed)

def n_iterate(iterator, n):
    '''
    Yield n tuples for an iterator.
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    :param iterator:
    :param n:
    :return:
    '''
    itr = iter(iterator)
    while True:
        try:
            yield tuple([next(itr) for i in range(n)])
        except StopIteration:
            return