# Interannotator agreement study design

Metrics: pair-wise F1-score as with ACE and ERE

Annotations to measure:
- Event trigger detection: token-span for event triggers
    - exact tokens: check if event mention detected on tokens
    - discontiguous matching: check if 
    
- partial span scoring:
    - span similarity score between 0 and 1
    - pair of mentions (G, S) is represented as a set of token ids (Tg, Ts)
    - 
    - Dice coefficient is equivalent to F1-score
    
- Mention mapped attributes:
    - Type
    - Subtype
    - Participants
    - Fillers
    - Arguments (combine Participants and Fillers)
    

Literature:
```latex
@article{Liu2015,
author = {Liu, Zhengzhong and Mitamura, Teruko and Hovy, Eduard},
file = {:home/gilles/repos/senteventannotation/literature/event extraction/ace tac ere event extraction/Evaluation Algorithms for Event Nugget Detection A Pilot Study{\_}2015{\_}Liu, Mitamura, Hovy.pdf:pdf},
journal = {Proceedings of the 3rd Workshop on EVENTS at the NAACL-HLT},
pages = {53--57},
title = {{Evaluation Algorithms for Event Nugget Detection : A Pilot Study}},
url = {http://www.aclweb.org/anthology/W/W15/W15-0807.pdf},
year = {2015}
}
```
- Details the scoring used for TAC-KBP event nugget detection
- 

```latex
@article{Mitamura2015,
author = {Mitamura, Teruko and Yamakawa, Yukari and Holm, Susan and Song, Zhiyi and Bies, Ann and Kulick, Seth and Strassel, Stephanie},
doi = {10.3115/v1/W15-0809},
file = {:home/gilles/.local/share/data/Mendeley Ltd./Mendeley Desktop/Downloaded/Event Nugget Annotation Processes and Issues{\_}2015{\_}Mitamura et al.pdf:pdf},
pages = {66--76},
title = {{Event Nugget Annotation: Processes and Issues}},
year = {2015}
}
```
- Contains scores for Tac-KBP corpus and ACE-2005 which can be used for comparison
- 