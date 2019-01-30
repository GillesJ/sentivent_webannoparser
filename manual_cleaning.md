# Manual Annotation Cleaning
---------------------------

We manually look at annotations and correct them.

## Common issues

All:
- MacroEconomics: remove MacroEconomics die niet over specifiek bedrijf of ligging in markt gaan.
    - verwijder macroeconomics zonder affectedcompany participants.
- Split MacroEconomics: Marketshare (tbd van bedrijf/product?) en Competitie discussions zijn duidelijk de belangrijkste.
- FinancialReport: kijk na of niet verward wordt met Profit/Loss, Expense, Revenue (vooral Anno_01)
- TIME Filler participants not always tagged, wrongly tagged (not a specific temportal point/duration)
- Earnings per share: Profit/Loss not FinancialReport of Revenue: check dit manual op string. Als token "earnings" gevonden: check of event of als event gevonden check dat type Profit/Loss is.
- Under/Overvalued: SecurityValue: check this manually
- Underweight/Overweight: analyst rating

Anno_01:
- Annotated a lot of events without participants.
- Financialreport en Expense vaak verward.
- SecurityValue en Rating vaak verward of verkeerd aangeduid.
- Earnings are Profit/Loss: confuses earnings per share a lot
- Stable subtype is tagged where no subtype should be tagged
- A lot events with no participants

Anno_02:
- TIME filler not full consitutent but NP only: resolve this automatically in preprocessing
- SalesVolume trigger often do not include "sales" but only the main verb: this is a problem.
- Overvalued is wrongly annotated as SecurityValue while it Rating.underperform: CHECK FOR ALL.
- (Content obligations are a film distributer EXPENSES not Investments regarding Netflix()

Anno_03:
- SalesVolume_subtype: does not annotate Amount of sales if is percentage
- SalesVolume trigger: missing verb or "sales"

# TODO:
- Identify largely unannotated docs and check them manually or remove them.
- Check if all below are possible with WebAnno automations.
- Remove macroeconomic events without affectedcompany
- FinancialReport: make "Result" participant part of the trigger, .
- Check corpus for specialization:
    - Check division of documents by company.
    - Check divisions of documents by sector.
    - Check division of events (all and by type) by company.
    - Check divisions of events (all and by type) by sector.
- Run separability checks on dataset trigger annotations.

# Typology Improvement Plan: Proposed typology changes
- Product/service:
    - requires a PRODUCTION (increase/decrease subtype) or a separate production type (encountered 8x when manually correcting, not really often, maybe not necessary)
    - Pricing comes up a lot too (40x when manually correcting): In coming version add Product/Service.Pricing.
- Macroeconomics: split in types/possible subtypes: Market Share, and rest: competitor.: Market Share is a good annotation to add.
- On all types that are metrics with Increase/Decrease/Stable types: add HistoricalAmount, Increase/DecreaseAmount, CurrentAmount
- On all financial statement and accounting metrics: add a Source participant that indicates the source (SalesVolume has Product/Sales), Revenue needs this, Expense needs this, Profit/Loss needs this.
- Rename Profit/Loss to Earnings.
- Legal: add Complainant on all subtypes.

V 1.1 changes:
- add Marketshare subtype on Macroeconomics
- add participants on financial metrics??? => too big an edit


# Discussion with Veronique (5/12/2018):
- Zelf verbeteren is beste oplossing: 2 weken nemen
- Rond kerst: schrijven.
- Publications:
    - LRE journal A1: resource description.
    - CB bullying paper: WebOfScience-indexed conferentie: conferenties met IEEE of AAAI conferenties in WoS.