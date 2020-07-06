#!/usr/bin/env python3
'''
Convert the WebAnno contraints.txt file into the typology.
For SENTiVENT annotation project, the constraints file contains all data from the typology.
convert_constraints.py
webannoparser 
12/18/18
Copyright (c) Gilles Jacobs. All rights reserved.  
'''
import re

if __name__ == "__main__":


    manual = [
        {"type": "CSR/Brand",
         "Participant": ["Company_CSR/Brand", ],
         "FILLER": ["PLACE", "TIME", "CAPITAL", ],
         },
        {"type": "Deal",
         "Participant": ["Goal", "Partner", ],
         "FILLER": ["PLACE", "TIME", "CAPITAL", ],
         },
        {"type": "Dividend",
         "Participant": ["Company_Dividend", "YieldRatio", "Amount_Dividend", ],
         "FILLER": ["PLACE", "TIME", "CAPITAL", ],
         "subtype": [
             {"type": "Payment"},
             {"type": "YieldRaise",
              "Participant": ["HistoricalYieldRatio", ]},
             {"type": "YieldReduction",
              "Participant": ["HistoricalYieldRatio", ]},
             {"type": "YieldStable",
              "Participant": ["HistoricalYieldRatio", ]},]
        },
        {"type": "Employment",
         "Participant": ["Employer", "Employee", ],
         "FILLER": ["PLACE", "TITLE", "TIME", "CAPITAL", ],
         "subtype": [
             {"type": "Start",
              "Participant": ["Replacing", ]},
             {"type": "End",
              "Participant": ["Replacer", ]},
             {"type": "Compensation",
              "Participant": ["Amount_Employment", ]},]
         },
    {"type": "Expense",
     "Participant": ["Amount_Expense", "Company_Expense", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     "subtype": [
         {"type": "Increase_Expense",
          "Participant": ["HistoricalAmount_Expense"]},
         {"type": "Decrease_Expense",
          "Participant": ["HistoricalAmount_Expense"]},
         {"type": "Stable_Expense",
          "Participant": ["HistoricalAmount_Expense"]},]
    },
    {"type": "Facility",
     "Participant": ["Facility", "Company_Facility", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     "subtype": [
         {"type": "Open"},
         {"type": "Close"},]
    },{"type": "FinancialReport",
     "Participant": ["Reportee", "Result", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     "subtype": [
         {"type": "Beat"},
         {"type": "Miss"},
         {"type": "Stable_FinancialReport"},]
    },
    {"type": "Financing",
     "Participant": ["Amount_Financing", "Financee", "Financer", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     },
    {"type": "Investment",
     "Participant": ["Investee", "CapitalInvested", "Investor", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     },
    {"type": "Legal",
     "Participant": ["Defendant", ],
     "FILLER": ["ALLEGATION", "PLACE", "TIME", "CAPITAL", ],
     b_Subtype = "Proceeding" | b_Subtype = "Conviction/Settlement" | b_Subtype = "Acquit" | b_Subtype = "Appeal";
    b_Subtype = "Proceeding",
                "Participant": ["Adjudicator", "Complainant", ],
                               b_Subtype = "Conviction/Settlement",
                                           "Participant": ["Adjudicator", ],
                                                          "FILLER": ["SENTENCE", ],
    b_Subtype = "Acquit",
                "Participant": ["Adjudicator", ],
                               b_Subtype = "Appeal",
                                           "Participant": ["Adjudicator", "Complainant", ],
    },
    {"type": "Macroeconomics",
     "Participant": ["AffectedCompany", "Sector", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     },
    {"type": "Merger/Acquisition",
     "Participant": ["Cost", "Target", "Acquirer", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     },
    {"type": "Product/Service",
     "Participant": ["ProductService", "Producer", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     b_Subtype = "Launch" | b_Subtype = "Cancellation/Recall" | b_Subtype = "Trial";
    b_Subtype = "Trial",
                "Participant": ["Trialer", ],
    },
    {"type": "Profit/Loss",
     "Participant": ["Amount_Profit/Loss", "Profiteer", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     b_Subtype = "Increase_Profit/Loss" | b_Subtype = "Decrease_Profit/Loss" | b_Subtype = "Stable_Profit/Loss";
    b_Subtype = "Increase_Profit/Loss",
                "Participant": ["HistoricalAmount_Profit/Loss", ],
                               b_Subtype = "Decrease_Profit/Loss",
                                           "Participant": ["HistoricalAmount_Profit/Loss", ],
                                                          b_Subtype = "Stable_Profit/Loss",
                                                                      "Participant": ["HistoricalAmount_Profit/Loss", ],
    },
    {"type": "Rating",
     "Participant": ["Analyst", "Security_Rating", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     b_Subtype = "BuyOutperform" | b_Subtype = "SellUnderperform" | b_Subtype = "Hold" | b_Subtype = "Upgrade" | b_Subtype = "Downgrade" | b_Subtype = "Maintain" | b_Subtype = "PriceTarget" | b_Subtype = "Credit/Debt";
    b_Subtype = "Upgrade",
                "Participant": ["HistoricalRating", ],
                               b_Subtype = "Downgrade",
                                           "Participant": ["HistoricalRating", ],
                                                          b_Subtype = "Maintain",
                                                                      "Participant": ["HistoricalRating", ],
                                                                                     b_Subtype = "PriceTarget",
                                                                                                 "Participant": [
                                                                                                                    "TargetPrice", ],
    },
    {"type": "Revenue",
     "Participant": ["Amount_Revenue", "Company_Revenue", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     b_Subtype = "Increase_Revenue" | b_Subtype = "Decrease_Revenue" | b_Subtype = "Stable_Revenue";
    b_Subtype = "Increase_Revenue",
                "Participant": ["HistoricalAmount_Revenue", "IncreaseAmount_Revenue", ],
                               b_Subtype = "Decrease_Revenue",
                                           "Participant": ["HistoricalAmount_Revenue", "DecreaseAmount_Revenue", ],
                                                          b_Subtype = "Stable_Revenue",
                                                                      "Participant": ["HistoricalAmount_Revenue", ],
    },
    {"type": "SalesVolume",
     "Participant": ["Amount_SalesVolume", "Buyer", "GoodsService", "Seller", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     b_Subtype = "Increase_SalesVolume" | b_Subtype = "Decrease_SalesVolume" | b_Subtype = "Stable_SalesVolume";
    b_Subtype = "Increase_SalesVolume",
                "Participant": ["HistoricalAmount_SalesVolume", ],
                               b_Subtype = "Decrease_SalesVolume",
                                           "Participant": ["HistoricalAmount_SalesVolume", ],
                                                          b_Subtype = "Stable_SalesVolume",
                                                                      "Participant": ["HistoricalAmount_SalesVolume", ],
    },
    {"type": "SecurityValue",
     "Participant": ["Security_SecurityValue", "Price", ],
     "FILLER": ["PLACE", "TIME", "CAPITAL", ],
     b_Subtype = "Increase_SecurityValue" | b_Subtype = "Decrease_SecurityValue" | b_Subtype = "Stable_SecurityValue";
    b_Subtype = "Increase_SecurityValue",
                "Participant": ["HistoricalPrice", "IncreaseAmount_SecurityValue", ],
                               b_Subtype = "Decrease_SecurityValue",
                                           "Participant": ["DecreaseAmount_SecurityValue", "HistoricalPrice", ],
                                                          b_Subtype = "Stable_SecurityValue",
                                                                      "Participant": ["HistoricalPrice", ],
    }
    ]

    constraints_fp = "/home/gilles/00 sentivent fwosb doctoraat 2017-2020/00-event-annotation/webanno-event-implementation/constraints/eventconstraints1.0.txt"

    with open(constraints_fp, "rt") as constr_in:
        constraints_txt = constr_in.read()

    # clean imports, whitespaces, etc
    constraints_txt = re.sub(r"import.*", "", constraints_txt)
    constraints_txt = re.sub(r"^(.*?)Event\s\{", "", constraints_txt, flags=re.MULTILINE)
    constraints_txt = re.sub(r"\/\*.*", "", constraints_txt, flags=re.MULTILINE)
    constraints_txt = re.sub(r"^[\s\n\t]+", "", constraints_txt, flags=re.MULTILINE)
    constraints_txt = re.sub(r"a_Type =", "},\n{\"type\": ", constraints_txt, flags=re.MULTILINE)
    constraints_txt = re.sub(r"^\w_(\w+)\.role = ", "\"\\1\": [", constraints_txt, flags=re.MULTILINE)
    constraints_txt = re.sub(r"\(\!\)\s+[;\|]$", ", ],", constraints_txt, flags=re.MULTILINE)
    constraints_txt = re.sub(r"\s+\(\!\)\s+\|\s+", ", ", constraints_txt, flags=re.MULTILINE)
    constraints_txt = re.sub(r"\w_(\w+)\.role = ", "", constraints_txt, flags=re.MULTILINE)
    constraints_txt = re.sub(r"\s->", ",", constraints_txt, flags=re.MULTILINE)
    constraints_txt = re.sub(r"\s->", ",", constraints_txt, flags=re.MULTILINE)

    constraints_txt = re.sub(r"\s->", ",", constraints_txt, flags=re.MULTILINE)
    print(constraints_txt)