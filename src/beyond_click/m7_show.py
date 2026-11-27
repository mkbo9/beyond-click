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


# Verdict → colour + marker shape + text, so the chart never relies on colour alone (colour-blind safe)
VERDICT_STYLE = {
    "more in top":    (BLUE, "triangle-right"),
    "more in bottom": (ORANGE, "triangle-left"),
    "unclear":        ("#4b5563", "circle"),
    "uninformative":  (GRAY, "circle-open"),
    "unreliable":     (GRAY, "x-open"),
}


def _chart_page(cfg, filename, title, intro, fig, side="", post_script=None):
    """One Plotly chart inside a small HTML page: FAKE banner, title, short explanation, optional side panel."""
    div = fig.to_html(full_html=False, include_plotlyjs="cdn", div_id="chart",   # cdn: page loads plotly.js online
                      post_script=post_script, config={"displaylogo": False})
    banner = f"<p style='background:#fde68a;padding:8px'><b>{_banner(cfg)}</b></p>" if _banner(cfg) else ""
    body = (f"<div style='display:flex;gap:16px;align-items:flex-start'><div style='flex:1;min-width:0'>{div}</div>{side}</div>"
            if side else div)
    (cfg["out"] / filename).write_text(
        f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title></head>"
        f"<body style='font-family:sans-serif;margin:16px'>{banner}<h1>{html.escape(title)}</h1>{intro}{body}</body></html>",
        encoding="utf-8")


def map_chart(cfg, d):
    """Map: close points = thumbnails that look alike. + top, − bottom, gray dot middle. Hover/click shows the image."""
    import plotly.graph_objects as go
    fig = go.Figure()
    styles = [("middle", GRAY, "circle", 6, "· middle"), ("bottom", ORANGE, "line-ew-open", 12, "− bottom"),
              ("top", BLUE, "cross", 11, "+ top")]                       # drawn in this order: top/bottom on top
    for lab, col, sym, size, name in styles:
        s = d[d.label == lab]
        src = [os.path.relpath(abs_path(cfg, p), cfg["out"]) for p in s.thumb_path]   # image path seen from outputs/
        custom = list(zip(src, s.title.astype(str), s.channel.astype(str), s.group_name.astype(str),
                          s.ratio.round(2), s.label))
        fig.add_trace(go.Scatter(
            x=s.map_x, y=s.map_y, mode="markers", name=name, customdata=custom,
            marker=dict(color=col, symbol=sym, size=size, opacity=0.6 if lab == "middle" else 1,
                        line=dict(width=3 if lab == "bottom" else 1, color=col)),
            hovertemplate="<b>%{customdata[1]}</b><br>%{customdata[2]}<br>%{customdata[3]}"
                          "<br>ratio %{customdata[4]} · %{customdata[5]}<extra></extra>"))
    fig.update_layout(template="simple_white", height=650, margin=dict(l=10, r=10, t=10, b=10),
                      legend=dict(orientation="h", y=-0.02, x=0, yanchor="top"), hovermode="closest",  # below: clear of the toolbar
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    side = ("<div id='panel' style='width:340px;position:sticky;top:10px;border:1px solid #ddd;padding:10px'>"
            "<img id='p_img' style='width:320px;display:none'>"
            "<p id='p_title' style='font-weight:bold'>Hover over or click a point to see its thumbnail.</p>"
            "<p id='p_info' style='font-size:14px'></p></div>")
    # small script: on hover/click copy the point's data into the side panel (textContent = no HTML injection)
    js = """
var gd = document.getElementById('{plot_id}');
function show(ev) {
  var c = ev.points[0].customdata; if (!c) return;
  var img = document.getElementById('p_img'); img.src = c[0]; img.style.display = 'block';
  document.getElementById('p_title').textContent = c[1];
  document.getElementById('p_info').textContent = c[2] + ' \\u00b7 ' + c[3] + ' \\u00b7 ratio ' + c[4] + ' \\u00b7 ' + c[5];
}
gd.on('plotly_hover', show); gd.on('plotly_click', show);
"""
    intro = ("<p>Each point is one thumbnail. Close points look alike (UMAP layout, for browsing only; "
             "the statistics use the groups, not this map). <b>+ blue = top</b>, <b>− orange = bottom</b>, "
             "gray dot = middle. Ratio = views ÷ the channel's usual views. Scroll to zoom, double-click to reset.</p>")
    _chart_page(cfg, "map.html", "Map of all thumbnails", intro, fig, side=side, post_script=js)


def compare_chart(cfg, comp):
    """One row per group and element: effect with 95% interval; colour, shape AND text show the verdict."""
    import plotly.graph_objects as go
    c = comp.copy()
    c["row"] = c.kind + " · " + c.name.astype(str)
    c["is_group"] = c.kind == "group"
    # sorted: groups first, then elements; within each, largest "more in top" at the top of the chart
    c = c.sort_values(["is_group", "effect"], ascending=[False, False]).reset_index(drop=True)
    order = list(c.row[::-1])                                  # plotly draws the first category at the bottom
    fig = go.Figure()
    for verdict, (col, sym) in VERDICT_STYLE.items():
        s = c[c.verdict == verdict]
        if s.empty:
            continue
        fig.add_trace(go.Scatter(
            x=s.effect * 100, y=s.row, mode="markers", name=verdict,
            marker=dict(color=col, symbol=sym, size=12, line=dict(width=2, color=col)),
            error_x=dict(type="data", symmetric=False, array=(s.ci_high - s.effect) * 100,
                         arrayminus=(s.effect - s.ci_low) * 100, color=col, thickness=2),
            customdata=list(zip(s.verdict, s.n_thumbnails, s.channels_positive, s.channels,
                                s.ci_low * 100, s.ci_high * 100, s.q.fillna(-1))),
            hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>difference %{x:+.0f} pts "
                          "(95%: %{customdata[4]:+.0f} to %{customdata[5]:+.0f})"
                          "<br>%{customdata[1]} thumbnails · %{customdata[2]} of %{customdata[3]} channels positive"
                          "<br>q = %{customdata[6]:.3f} (−1 = not tested)<extra></extra>"))
    for r in c.itertuples():                                   # verdict as text at the right edge of every row
        fig.add_annotation(x=1.01, y=r.row, xref="paper", yref="y", text=r.verdict, showarrow=False,
                           xanchor="left", font=dict(color=VERDICT_STYLE.get(r.verdict, ("#4b5563",))[0], size=13))
    fig.add_vline(x=0, line_color="#6b7280", line_width=1)
    fig.update_layout(template="simple_white", height=160 + 34 * len(c), margin=dict(l=10, r=140, t=30, b=60),
                      xaxis_title="← more in bottom      difference in share, top − bottom (%-points)      more in top →",
                      yaxis=dict(categoryorder="array", categoryarray=order, title=None),
                      legend=dict(orientation="h", y=-0.12 if len(c) > 6 else -0.3), hovermode="closest")
    intro = ("<p>For each group and element: how much more often it appears in a channel's top thumbnails than in "
             "its bottom ones (%-points, averaged over channels). Bar = 95% interval over channels. "
             "<i>more in top / more in bottom</i> only if the shuffle test passes the false-discovery correction "
             "and the interval excludes 0. Uninformative = in fewer than 5% or more than 95% of thumbnails; "
             "unreliable = element scores disagree with the hand marks. Association only: topic, title and "
             "timing also affect views.</p>")
    _chart_page(cfg, "compare.html", "Top vs bottom thumbnails (within channels)", intro, fig)


def model_chart(cfg, pred):
    """Per-channel average AUC per model, next to the shuffled-labels check and the 0.5 line."""
    import plotly.graph_objects as go
    p = pred.iloc[::-1].reset_index(drop=True)                # first model at the top
    shuffled = p.model.str.startswith("shuffled")
    fig = go.Figure(go.Bar(
        x=p.auc, y=p.model, orientation="h", marker=dict(color=[GRAY if s else BLUE for s in shuffled]),
        text=[f"{a:.2f}" + ("  (check)" if s else "") for a, s in zip(p.auc, shuffled)],
        textposition="inside", insidetextanchor="start", textfont=dict(color="white", size=14),  # clear of whiskers
        error_x=dict(type="data", symmetric=False, array=p.auc_max_channel - p.auc,
                     arrayminus=p.auc - p.auc_min_channel, color="#374151", thickness=1.5),
        customdata=list(zip(p.auc_pooled, p.auc_min_channel, p.auc_max_channel)),
        hovertemplate="<b>%{y}</b><br>average per channel %{x:.3f}<br>pooled %{customdata[0]:.3f}"
                      "<br>range %{customdata[1]:.3f} – %{customdata[2]:.3f}<extra></extra>"))
    # 0.5 line drawn as an explicit shape on top of the bars, so it stays visible
    fig.add_shape(type="line", x0=0.5, x1=0.5, xref="x", y0=0, y1=1, yref="paper", layer="above",
                  line=dict(color="#111827", width=2, dash="dash"))
    fig.add_annotation(x=0.5, y=1, xref="x", yref="paper", yanchor="bottom", showarrow=False,
                       text="0.5 = coin flip")
    fig.update_layout(template="simple_white", height=120 + 70 * len(p), margin=dict(l=10, r=30, t=40, b=50),
                      xaxis=dict(range=[0, 1.05], title="AUC (average over test channels; 1 = perfect, 0.5 = coin flip)"),
                      yaxis=dict(title=None), showlegend=False)
    intro = ("<p>Each model is tested on channels it never saw in training. Bar = AUC averaged over the test channels; "
             "whisker = lowest to highest channel. Gray bar = sanity check with shuffled top/bottom labels "
             "(mean and range over 20 shuffles): it should sit near 0.5. A model bar is only interesting if it "
             "is clearly to the right of the gray bar.</p>")
    _chart_page(cfg, "model.html", "Model test on unseen channels", intro, fig)


def charts(cfg, d, comp, pred):
    map_chart(cfg, d)
    compare_chart(cfg, comp)
    if len(pred):
        model_chart(cfg, pred)
    print("  charts: outputs/map.html, compare.html" + (", model.html" if len(pred) else ""))


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
    return f"<html><head><meta charset='utf-8'></head><body style='font-family:sans-serif'>{banner}<h1>{title}</h1>{legend}{body}</body></html>"


def gallery(cfg, d, n=12, blind=False, filename="gallery.html"):
    """Per group: the most typical thumbnails. Use it to check and fix the group names."""
    parts = []
    for _, g in d[d.group >= 0].groupby("group"):              # by group number: g0, g1, g2 … (not g0, g10)
        name = g.group_name.iloc[0]
        info = f"{len(g)} thumbnails · {g.channel_handle.nunique()} channels"
        if not blind:
            info += f" · {g.label.value_counts().to_dict()}"
        parts.append(f"<h2>{html.escape(name)}</h2><p>{info}</p><div>")
        parts += [_img(cfg, r, blind) for r in g.sort_values("membership", ascending=False).head(n).itertuples()]
        parts.append("</div>")
    (cfg["out"] / filename).write_text(_page(cfg, "Groups: the most typical thumbnails per group", "".join(parts), blind),
                                     encoding="utf-8")


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
    (cfg["out"] / filename).write_text(_page(cfg, "Elements: do the scores match what you see?", "".join(parts), blind),
                                     encoding="utf-8")


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
        import plotly  # noqa: F401
    except ImportError:
        print("  plotly not installed: charts skipped")
    else:
        charts(cfg, d, comp, pred)                 # real errors inside the charts are not hidden
    gallery(cfg, d)
    element_gallery(cfg, d)
    summary(cfg, d, comp, pred)
