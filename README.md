# Parser for Webanno XMI exported projects

- Parses WebAnno exports in UIMA XMI format.
Note:
- Current prototype is project-specific to my annotation project and requires to manually define the annotation layer/feature keys for extraction.

## Usage

1. Parse the unzipped Webanno XMI format with `parser.py`

a. Run stats and viz on filtered docs

b. Run 

## TODO
- link Spacy API.
- implement general parser using [cassis](https://github.com/dkpro/dkpro-cassis).
- implement mapping of CAS to WebAnno annotation layer and feature types.