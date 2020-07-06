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
- link Spacy API.
- implement general parser using [cassis](https://github.com/dkpro/dkpro-cassis): not feasible as a custom mapping to objects is still needed.