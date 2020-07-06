#!/usr/bin/env python3
"""
Generate some figures on the SENTiVENT sentiment corpus annotations.

sentiment_viz.py in sentivent_webannoparser
6/16/20 Copyright (c) Gilles Jacobs
"""
from cycler import cycler
import numpy as np
import pandas as pd
import settings
from parse_project import parse_project
import matplotlib.pyplot as plt


def annotations_to_df(project):
    def to_data(u):
        ud = u.__dict__
        to_add = {
            "annotation_unit": type(u).__name__,
            "object": u,
        }
        ud.update(to_add)
        return ud

    unames = ["sentiment_expressions", "events"]
    all_df = pd.DataFrame(
        to_data(u)
        for uname in unames
        for u in project.get_annotation_from_documents(uname)
    )

    # join columns with same info but different name
    join_map = [("polarity_negation", "negated"), ("modality", "uncertain")]
    for a, b in join_map:
        all_df[a].update(all_df.pop(b))
    all_df["polarity_negation"] = all_df["polarity_negation"].replace(
        {"false": "positive", "true": "negative"}
    )
    all_df["modality"] = all_df["modality"].replace(
        {"false": "certain", "true": "other"}
    )

    return all_df


def plot_sentiment_polarity_chart(all_df, opt_fp):
    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    cats_in_order = ["positive", "neutral", "negative"]

    all_df = all_df.replace({"SentimentExpression": "Sentiment spans"})

    df = (
        all_df.groupby("annotation_unit")
        .polarity_sentiment.value_counts(normalize=True)
        .unstack()
        * 100
    )
    df_count = (
        all_df.groupby("annotation_unit")
        .polarity_sentiment.value_counts(normalize=False)
        .unstack()
    )
    df = df[cats_in_order]
    df_count = df_count[cats_in_order]
    y_lim_upper = round(df.max().max() * 1.2, 2)  # add more upper space for annotations

    ax = plt.figure(figsize=(4.2, 4)).add_subplot(111)
    df.plot(
        ax=ax,
        kind="bar",
        stacked=False,
        rot=0,
        ylim=(0.0, y_lim_upper),
        width=0.8,
        title=None,
        alpha=0.99,  # needed to unbreak hatching in pdf
        edgecolor="white",
    )
    # set counts on bar
    for c, col_patches in enumerate(
        chunker(ax.patches, df.shape[0])
    ):  # patches iterate over columns
        for r, p in enumerate(col_patches):
            n = df_count.iloc[r, c]
            pct = round(df.iloc[r, c], 1)
            ax.annotate(
                f"{pct}%\n({n})",
                (p.get_x() + p.get_width() / 2.0, p.get_height() + 1),
                ha="center",
                va="center",
                xytext=(0, 10),
                fontsize=8,
                textcoords="offset points",
            )
    # add hatching for monochrome printing
    bars = ax.patches
    hatches = "".join(h * len(df) for h in "/ .xO+-")

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch * 2)

    ax.get_xaxis().get_label().set_visible(False)
    plt.legend(cats_in_order, frameon=False)
    plt.tight_layout()
    plt.savefig(opt_fp)
    plt.close()

    df_count.plot(kind="bar", stacked=True, rot=0, width=0.75)
    plt.legend(cats_in_order, frameon=False)
    plt.tight_layout()
    plt.savefig(opt_fp.replace(".pdf", "_stacked.pdf"))
    plt.close()


if __name__ == "__main__":

    anno_id_allowed = ["jefdhondt", "elinevandewalle", "haiyanhuang"]
    project = parse_project(settings.SENTIMENT_IAA)
    project.annotation_documents = [
        d for d in project.annotation_documents if d.annotator_id in anno_id_allowed
    ]

    # freq_stats
    df_all = annotations_to_df(project)
    print(df_all.polarity_negation.value_counts())
    print(df_all.modality.value_counts())
    print(df_all.polarity_sentiment.value_counts(normalize=True))

    plot_sentiment_polarity_chart(df_all, "output/plots/iaa_freq_sentiment_pol.pdf")
    df_se = df_all[df_all["annotation_unit"] == "SentimentExpression"]
    df_ev = df_all[df_all["annotation_unit"] == "Event"]

    # make same plot for final project
    project = parse_project(settings.SENTIMENT_ANNO)
    df_all = annotations_to_df(project)
    plot_sentiment_polarity_chart(df_all, "output/plots/anno_freq_sentiment_pol.pdf")

    pass
