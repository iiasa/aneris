from pathlib import Path

import seaborn as sns
from concordia.report import (
    HEADING_TAGS,
    add_plotly_header,
    add_sticky_toc,
    embed_image,
)
from dominate import document
from dominate.tags import div
from pandas_indexing import concat, isin, ismatch
from tqdm.auto import tqdm


px = None  # replace with plotly import


DEFAULT_LEVELS = ["gas", "sector", "region"]


def plot_harm(
    harmonized,
    hist,
    model,
    sel,
    scenario=None,
    levels=DEFAULT_LEVELS,
    useplotly=False,
):
    model_sel = sel if scenario is None else sel & ismatch(scenario=scenario)
    h = harmonized.loc[model_sel]

    data = concat(
        dict(
            History=hist.loc[sel],
            Unharmonized=model.loc[model_sel],
            Harmonized=h,
        ),
        keys="pathway",
    )

    non_uniques = [lvl for lvl in levels if len(h.pix.unique(lvl)) > 1]
    if not non_uniques:
        non_uniques = ["region"]
        data = data.pix.semijoin(h.pix.unique(levels), how="right")

    (non_unique,) = non_uniques

    if useplotly:
        g = px.line(
            data.pix.to_tidy(),
            x="year",
            y="value",
            color="pathway",
            style="version",
            facet_col=non_unique,
            facet_col_wrap=4,
            labels=dict(value=data.pix.unique("unit").item(), pathway="Trajectory"),
        )
        g.update_yaxes(matches=None)

    num_facets = len(data.pix.unique(non_unique))
    multirow_args = dict(col_wrap=4, height=2, aspect=1.5) if num_facets > 1 else dict()
    g = sns.relplot(
        data.pix.to_tidy(),
        kind="line",
        x="year",
        y="value",
        col=non_unique,
        hue="pathway",
        facet_kws=dict(sharey=False),
        legend=True,
        **multirow_args,
    ).set(ylabel=data.pix.unique("unit").item())
    for label, ax in g.axes_dict.items():
        ax.set_title(f"{non_unique} = {label}", fontsize=9)
    return g


def what_changed(next, prev):
    length = len(next)
    if prev is None:
        return range(length)
    for i in range(length):
        if prev[i] != next[i]:
            return range(i, length)


def make_doc(harmonized, hist, model, order, compact=False, useplotly=False):
    index = harmonized.index.pix.unique(order).sort_values()
    doc = document(title="Harmonization results")

    main = doc.add(div())
    prev_idx = None
    for idx in tqdm(index):
        main.add([HEADING_TAGS[i](idx[i]) for i in what_changed(idx, prev_idx)])

        try:
            ax = plot_harm(
                harmonized,
                hist,
                model,
                isin(**dict(zip(index.names, idx)), ignore_missing_levels=True),
                useplotly=useplotly,
            )
        except ValueError:
            print(
                f"During plot_harm(isin(**{dict(zip(index.names, idx))}, ignore_missing_levels=True))"
            )
            raise
        main.add(embed_image(ax, close=True))

        prev_idx = idx

    add_sticky_toc(doc, max_level=2, compact=compact)
    if useplotly:
        add_plotly_header(doc)
    return doc


def make_scenario_facets(
    harmonized,
    hist,
    model,
    order=["gas", "sector"],
    out_path="results",
    suffix="",
    useplotly=False,
):
    out_path = Path(out_path)
    out_path.mkdir(exist_ok=True)

    suffix = "-plotly" if useplotly and suffix == "" else suffix
    fn = out_path / f"harmonization-facet{suffix}.html"

    # lock = FileLock(out_path / ".lock")
    doc = make_doc(harmonized, hist, model, order=order, useplotly=useplotly)

    # with lock:
    with open(fn, "w", encoding="utf-8") as f:
        print(doc, file=f)
    return fn
