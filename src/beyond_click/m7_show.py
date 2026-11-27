"""M7 Show: map, comparison chart, model chart (Plotly HTML) + galleries + summary.

Writes (docs/INTERFACES.md): outputs/map.html, compare.html, model.html (need plotly),
                             outputs/gallery.html, elements.html, summary.md (no plotly needed)
Also provides check_pages() for M4: check_groups.html, check_elements.html with top/bottom hidden.
"""
import html
import json
import os
import pandas as pd

from .config import abs_path
from .m4_describe import group_labels, load_cuts
from .store import load_labeled, merge_checked

BLUE, ORANGE, GRAY = "#2563eb", "#ea580c", "#9ca3af"
BORDER = {"top": BLUE, "bottom": ORANGE}


def _load(cfg):
    d = merge_checked(load_labeled(cfg), cfg, "groups.csv", "m3")
    d["group_name"] = d.group.map(group_labels(cfg)).fillna("ungrouped")
    return d


def _banner(cfg) -> str:
    return "FAKE DATA: test run, not real results" if cfg["mode"] == "fake" else ""


def charts(cfg, d, comp, pred):
    import plotly.graph_objects as go
    note = f" [{_banner(cfg)}]" if _banner(cfg) else ""
    # 1 map: + top, − bottom, gray middle
    fig = go.Figure()
    for lab, col, sym in [("middle", GRAY, "circle"), ("bottom", ORANGE, "line-ew-open"), ("top", BLUE, "cross")]:
        s = d[d.label == lab]
        fig.add_trace(go.Scatter(x=s.map_x, y=s.map_y, mode="markers", name=lab,
                                 marker=dict(color=col, symbol=sym, size=9 if lab != "middle" else 5,
                                             line=dict(width=2, color=col)),
                                 text=s.title + "<br>" + s.channel + "<br>" + s.group_name + "<br>ratio " + s.ratio.round(2).astype(str),
                                 hoverinfo="text"))
    fig.update_layout(title="Map: close = look alike (layout for browsing only)" + note, template="simple_white",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    fig.write_html(cfg["out"] / "map.html", include_plotlyjs="cdn")

    # 2 groups + elements: difference with error bar
    c = comp[comp.verdict != "uninformative"].sort_values(["kind", "effect"])
    colors = c.verdict.map({"more in top": BLUE, "more in bottom": ORANGE}).fillna(GRAY)
    fig = go.Figure(go.Scatter(
        x=c.effect * 100, y=c.kind + ": " + c.name, mode="markers", marker=dict(color=colors, size=10),
        error_x=dict(type="data", symmetric=False, array=(c.ci_high - c.effect) * 100,
                     arrayminus=(c.effect - c.ci_low) * 100, color="#6b7280")))
    fig.add_vline(x=0, line_color="#6b7280")
    fig.update_layout(title="Top vs bottom thumbnails (within channels)" + note, template="simple_white",
                      xaxis_title="← more in bottom    difference (%-points)    more in top →",
                      height=300 + 28 * len(c))
    fig.write_html(cfg["out"] / "compare.html", include_plotlyjs="cdn")

    # 3 model: average per-channel AUC
    if len(pred):
        fig = go.Figure(go.Bar(x=pred.auc, y=pred.model, orientation="h"))
        fig.add_vline(x=0.5, line_dash="dash", annotation_text="coin flip")
        fig.update_layout(title="Model test on unseen channels (AUC, average per channel)" + note,
                          template="simple_white", xaxis_range=[0.3, 1])
        fig.write_html(cfg["out"] / "model.html", include_plotlyjs="cdn")


def _img(cfg, r, blind=False, caption=None) -> str:
    src = os.path.relpath(abs_path(cfg, r.thumb_path), cfg["out"])   # link from outputs/ to the image
    colour = "#ddd" if blind else BORDER.get(r.label, "#ddd")        # blind pages never reveal top/bottom
    img = (f"<img src='{src}' width='192' style='border:4px solid {colour};margin:3px' "
           f"title='{html.escape(str(r.title))}'>")
    if caption is None:
        return img
    # small text under the image (e.g. the element score); never top/bottom
    return (f"<figure style='display:inline-block;margin:0 2px;text-align:center'>{img}"
            f"<figcaption style='font-size:12px'>{html.escape(caption)}</figcaption></figure>")


def _page(cfg, title, body, blind=False) -> str:
    banner = f"<p style='background:#fde68a;padding:8px'><b>{_banner(cfg)}</b></p>" if _banner(cfg) else ""
    legend = ("<p>Check page: top/bottom labels are hidden on purpose.</p>" if blind else
              "<p>Frame colour: <b style='color:#2563eb'>blue = top</b>, <b style='color:#ea580c'>orange = bottom</b>, gray = middle</p>")
    return f"<html><body style='font-family:sans-serif'>{banner}<h1>{title}</h1>{legend}{body}</body></html>"


def gallery(cfg, d, n=12, blind=False, filename="gallery.html"):
    """Per group: the most typical thumbnails. Use it to check and fix the group names."""
    parts = []
    for name, g in d[d.group >= 0].groupby("group_name"):
        info = f"{len(g)} thumbnails · {g.channel_handle.nunique()} channels"
        if not blind:
            info += f" · {g.label.value_counts().to_dict()}"
        parts.append(f"<h2>{html.escape(name)}</h2><p>{info}</p><div>")
        parts += [_img(cfg, r, blind) for r in g.sort_values("membership", ascending=False).head(n).itertuples()]
        parts.append("</div>")
    (cfg["out"] / filename).write_text(_page(cfg, "Groups: the most typical thumbnails per group", "".join(parts), blind))


def element_gallery(cfg, d, n=10, blind=False, filename="elements.html"):
    """Per element: the thumbnails scored highest, nearest the cut and lowest. Use it to check the scores by eye."""
    el = merge_checked(d, cfg, "elements.csv", "m4")
    cuts = load_cuts(cfg)
    parts = []
    for name in cfg["elements"]:
        share = el[f"has_{name}"].mean()
        cut, status = cuts[name]["cut"], cuts[name]["status"]
        parts.append(f"<h2>{html.escape(name)}</h2><p>present in {share:.0%} of thumbnails · cut {cut:.3f} ({status}) · "
                     f"phrases: {html.escape(' / '.join(cfg['elements'][name]))}</p>")
        col = f"p_{name}"
        # thumbnails whose score is closest to the cut: shows whether the yes/no line is in the right place
        near = el.loc[(el[col] - cut).abs().sort_values().index[:n]].sort_values(col, ascending=False)
        for title, part in (("Highest scores (should show it)", el.nlargest(n, col)),
                            (f"Near the cut {cut:.3f} (score above the cut = counted as present, below = absent)", near),
                            ("Lowest scores (should NOT show it)", el.nsmallest(n, col))):
            imgs = "".join(_img(cfg, r, blind, caption=f"{getattr(r, col):.2f} · {'yes' if getattr(r, col) > cut else 'no'}")
                           for r in part.itertuples())   # score + yes/no under every image
            parts.append(f"<h4>{title}</h4><div>{imgs}</div>")
    (cfg["out"] / filename).write_text(_page(cfg, "Elements: do the scores match what you see?", "".join(parts), blind))


def check_pages(cfg):
    """Blind check pages, written at the end of M4 (before any top/bottom comparison).

    Use them to fix group names and to drop elements whose scores are wrong.
    They hide top/bottom, so checking them cannot bias the later results.
    """
    d = _load(cfg)
    gallery(cfg, d, blind=True, filename="check_groups.html")
    element_gallery(cfg, d, blind=True, filename="check_elements.html")
    print("  check pages (labels hidden): outputs/check_groups.html, outputs/check_elements.html")


def summary(cfg, d, comp, pred):
    lines = ["# Summary", ""]
    if _banner(cfg):
        lines += [f"**{_banner(cfg)}**", ""]
    info_path = cfg["processed"] / "grouping.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    lines += [f"- Thumbnails: {len(d)} from {d.channel_handle.nunique()} channels "
              f"(top {int((d.label == 'top').sum())}, bottom {int((d.label == 'bottom').sum())})",
              f"- Grouping: {info.get('n_groups', '?')} groups · {info.get('method', '?')} · "
              f"channel look removed first: {info.get('channel_centred', '?')}", ""]
    for verdict in ("more in top", "more in bottom"):
        hits = comp[comp.verdict == verdict]
        lines.append(f"- **{verdict}:** " + (", ".join(f"{r['kind']} '{r['name']}' ({r['effect'] * 100:+.0f} pts)"
                                                       for _, r in hits.iterrows()) or "none"))
    if len(pred):
        lines += ["", "**Model test on unseen channels (AUC averaged per channel; 0.5 = coin flip):**"]
        lines += [f"- {r['model']}: {r['auc']:.2f}" for _, r in pred.iterrows()]
    lines += ["", "Association only: topic, title and timing also affect views."]
    (cfg["out"] / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def run(cfg):
    d = _load(cfg)
    comp = pd.read_csv(cfg["out"] / "compare.csv")
    pred_path = cfg["out"] / "predict_scores.csv"
    pred = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
    try:
        charts(cfg, d, comp, pred)
    except ImportError:
        print("  plotly not installed: charts skipped")
    gallery(cfg, d)
    element_gallery(cfg, d)
    summary(cfg, d, comp, pred)
