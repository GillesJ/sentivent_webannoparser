'''
Testing BIO labeling function for correctness
'''

def get_bio_tags(context_tokens, annotation_tokens):

    token_bio_seq = ["O"] * len(context_tokens)
    for unit in annotation_tokens:
        unit_tags = ["B"] + ["I"]*(len(unit)-1)
        idc = unit
        for i, j in enumerate(idc):
            token_bio_seq[j] = unit_tags[i]

    return token_bio_seq

context = [0,1,2,3,4,5]

separated = [[0,1],[3,4]]
separated_tags = ["B","I","O","B","I","O"]

separated_res = get_bio_tags(context, separated)
assert separated_tags == separated_res

adjacent = [[0,1,2],[3,4]]
adjacent_tags = ["B","I","I","B","I","O"]

adjacent_res = get_bio_tags(context, adjacent)
assert adjacent_tags == adjacent_res

overlap = [[0,1,2],[2,3,4]]
overlap_tags = ["B","I","B","I","I", "O"]
# replacing the tags of preceding annotation is best choice, since this rare and an annotation artifact
# we assume later B boundary annotations in text are more intentional.
overlap_res = get_bio_tags(context, overlap)
assert overlap_tags == overlap_res

pass


