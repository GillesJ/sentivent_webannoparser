# Parser for Webanno XMI exported projects

Parse WebAnno project exports in UIMA XMI format to Python objects.
Note:
- Current prototype is project-specific to my annotation project and requires to manually define the annotation layer/feature keys for extraction.

Repository also contains code for inter-annotator studies, reference scoring of events, data viz, and corpus analysis 
for the SENTiVENT project (due for modularization and cleaning in the future).

## Usage and contents

- Parse the unzipped Webanno XMI format with `parser.py` (convenient calling functions in `parse_project.py`)
- Interannotator-agreement study scripts:
  - `span_iaa.py`: match spans and produce .csv files for Agreestat360.com
  - `sentiment_event_iaa.py`: produce Agreestat360.com files for sentiment polarity on events (does not need matching)
- Nugget scorer: `nugget_scorer.py` ERE-like event nugget scoring.
- Inspection of annotations as helper scripts during correction, finding common mistakes
  - `scripts/event_inspect_annotations.py`: Inspect event annotations for potential issues during event annotation.
  - `scripts/inspect_sentiment_annotations.py`: Inspect during sentiment annotations.
- Generating figures and tables with stats:
  - `corpus_stats_viz.py`: Event annotations.
  - `sentiment_viz.py`: Sentiment annotations.

DEPRECATED/Prototyping (do not use) for archival purposes:
- `scripts/iaa_eval.py`: ugly prototype attempt at nugget scoring, probably contains error in matching.
- `scripts/iaa_nugget_metrics.py`: Rewrite of the `iaa_eval.py` to allow easier pairwise computations.
- `scripts/iaa_token_metrics.py`: pairwise-token level IAA > naive not good for change-corrected metrics, can be used for P, R, F1
- `scripts/to_spacy.py`, `parser_tests.py`, `test_cassis.py`: unfinished testing and prototype code.

## TODO
-[ ] Clean and split this repo into multiple repos for IAA, corpus viz /stat analysis, etc.
-[ ] Link Spacy API to document representation for parse trees.
-[ ] ~~implement general parser using [cassis](https://github.com/dkpro/dkpro-cassis):~~ Not very useful as custom mapping code to objects is still needed, annotation edits and exports are not straight-forward.

## Repo Mirrors
LAN:
- gillesLatitude: ~/repos
- share: lt3_sentivent

WAN:
- https://github.com/GillesJ/sentivent_webannoparser

## Contact
- Gilles Jacobs: gilles@jacobsgill.es, gilles.jacobs@ugent.be
- Veronique Hoste: veronique.hoste@ugent.be