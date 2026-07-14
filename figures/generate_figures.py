#!/usr/bin/env python3
"""Generate ICAIS submission figures — ICLR/Q-RAG TikZ-aligned raster fallbacks for Word."""

from pathlib import Path

OUT = Path(__file__).resolve().parent


def esc(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def figure1_flowchart() -> str:
    """Grayscale agent/environment diagram aligned with figure1_flow.tex."""
    w, h = 920, 210
    font = "'Times New Roman', Times, serif"
    mono = "'Courier New', Courier, monospace"
    stroke = "#333333"
    light = "#f7f7f7"
    agent_fill = "#fafafa"
    env_stroke = "#888888"

    def box(x, y, bw, bh, title, sub="", dashed=False):
        dash = ' stroke-dasharray="4,3"' if dashed else ""
        sub_y = y + bh - 11 if sub else y + bh
        sub_el = (
            f'<text x="{x+bw/2}" y="{sub_y}" text-anchor="middle" font-family="{mono}" '
            f'font-size="8.5" fill="#444">{esc(sub)}</text>'
            if sub
            else ""
        )
        return f"""
    <rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="2" fill="white" stroke="{stroke}" stroke-width="0.9"{dash}/>
    <text x="{x+bw/2}" y="{y+16}" text-anchor="middle" font-family="{font}" font-size="10"
          font-weight="bold" fill="#111">{esc(title)}</text>{sub_el}"""

    def diamond(cx, cy, dw, dh, text):
        pts = f"{cx},{cy-dh/2} {cx+dw/2},{cy} {cx},{cy+dh/2} {cx-dw/2},{cy}"
        return f"""
    <polygon points="{pts}" fill="#f5f5f5" stroke="{stroke}" stroke-width="0.9"/>
    <text x="{cx}" y="{cy+4}" text-anchor="middle" font-family="{mono}" font-size="9" fill="#222">{esc(text)}</text>"""

    def arr(x1, y1, x2, y2, dashed=False):
        dash = ' stroke-dasharray="4,3"' if dashed else ""
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="1"{dash} marker-end="url(#arr)"/>'
        )

    def lbl(x, y, text):
        return (
            f'<text x="{x}" y="{y}" text-anchor="middle" font-family="{font}" '
            f'font-size="8.5" fill="#555">{esc(text)}</text>'
        )

    # Coordinates
    env = (24, 22, 872, 168)
    inst = (44, 62, 108, 52)
    route = (176, 62, 118, 52)
    coco = (314, 62, 118, 52)
    m2 = (450, 62, 92, 52)
    gate_cx, gate_cy, gate_w, gate_h = 590, 88, 58, 34
    out = (668, 62, 88, 52)
    fb = (314, 132, 150, 44)
    cy = 62 + 26

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        "<defs>",
        '<marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">',
        f'<path d="M0,0 L7,3 L0,6 Z" fill="{stroke}"/>',
        "</marker>",
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
        # environment
        f'<rect x="{env[0]}" y="{env[1]}" width="{env[2]}" height="{env[3]}" rx="4" '
        f'fill="{light}" stroke="{env_stroke}" stroke-width="0.9" stroke-dasharray="5,4"/>',
        f'<text x="{env[0]+8}" y="{env[1]+14}" font-family="{font}" font-size="9.5" fill="#666">'
        f'ProsQA inference environment</text>',
        # agent box
        f'<rect x="162" y="48" width="610" height="132" rx="3" fill="{agent_fill}" '
        f'stroke="#aaa" stroke-width="0.8"/>',
        f'<text x="170" y="58" font-family="{font}" font-size="9" font-weight="bold" fill="#777">'
        f'confidence_fallback agent</text>',
    ]

    ix, iy, iw, ih = inst
    parts.append(box(ix, iy, iw, ih, "ProsQA instance", "graph G, root r, targets"))
    parts.append(
        f'<text x="{ix+iw/2}" y="{iy-6}" text-anchor="middle" font-family="{font}" '
        f'font-size="9" font-weight="bold" fill="#444">Input x</text>'
    )
    parts.append(box(*route, "Structure route", "n0 = clamp(d)"))
    parts.append(box(*coco, "Coconut", "y0, h_n0"))
    parts.append(box(*m2, "M2 head", "p0 = sigmoid(M2)"))
    parts.append(diamond(gate_cx, gate_cy, gate_w, gate_h, "p0 >= tau"))
    parts.append(box(*out, "Output", "y"))
    parts.append(box(*fb, "kNN + M2 fallback", "online stop search", dashed=True))

    parts.append(arr(ix + iw, cy, route[0], cy))
    parts.append(arr(route[0] + route[2], cy, coco[0], cy))
    parts.append(arr(coco[0] + coco[2], cy, m2[0], cy))
    parts.append(arr(m2[0] + m2[2], cy, gate_cx - gate_w / 2, cy))
    parts.append(arr(gate_cx + gate_w / 2, cy, out[0], cy))
    parts.append(lbl(gate_cx + gate_w / 2 + 18, cy - 6, "yes"))

    parts.append(arr(gate_cx, gate_cy + gate_h / 2, gate_cx, fb[1], dashed=True))
    parts.append(lbl(gate_cx - 14, gate_cy + gate_h / 2 + 14, "no"))

    fb_cx = fb[0] + fb[2] / 2
    out_cx = out[0] + out[2] / 2
    route_y = fb[1] + fb[3] + 10
    parts.append(arr(fb_cx, fb[1] + fb[3], fb_cx, route_y, dashed=True))
    parts.append(arr(fb_cx, route_y, out_cx, route_y, dashed=True))
    parts.append(arr(out_cx, route_y, out_cx, out[1] + out[3], dashed=True))

    parts.append("</svg>")
    return "\n".join(parts)


def figure2_bar_chart() -> str:
    w, h = 760, 380
    font = "'Times New Roman', Times, serif"
    baseline = 83.8
    methods = [
        ("fixed_3", 83.8),
        ("auto_route", 93.1),
        ("structure_d", 93.6),
        ("knn_min3", 92.6),
        ("conf. fallback", 95.23),
    ]
    y_min, y_max = 80, 100
    plot_l, plot_r, plot_t, plot_b = 72, 28, 56, 74
    pw = w - plot_l - plot_r
    ph = h - plot_t - plot_b
    bar_w = 62
    gap = (pw - len(methods) * bar_w) / (len(methods) + 1)

    def y_pos(val):
        return plot_t + ph * (1 - (val - y_min) / (y_max - y_min))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-family="{font}" font-size="12" '
        f'font-weight="bold" fill="#111">Comparison of answer accuracy on ProsQA (419 questions)</text>',
        f'<line x1="{plot_l}" y1="{plot_t}" x2="{plot_l}" y2="{plot_t+ph}" stroke="#333" stroke-width="0.9"/>',
        f'<line x1="{plot_l}" y1="{plot_t+ph}" x2="{plot_l+pw}" y2="{plot_t+ph}" stroke="#333" stroke-width="0.9"/>',
        f'<text x="24" y="{plot_t+ph/2+4}" text-anchor="middle" font-family="{font}" font-size="10" '
        f'fill="#333" transform="rotate(-90 24 {plot_t+ph/2+4})">Accuracy (%)</text>',
    ]

    y_base_line = y_pos(baseline)
    parts.append(
        f'<line x1="{plot_l}" y1="{y_base_line}" x2="{plot_l+pw}" y2="{y_base_line}" '
        f'stroke="#666" stroke-width="0.9" stroke-dasharray="5,4"/>'
    )
    parts.append(
        f'<text x="{plot_l+pw-4}" y="{y_base_line-4}" text-anchor="end" font-family="{font}" '
        f'font-size="8.5" fill="#666">fixed_3 baseline</text>'
    )

    for tick in range(80, 101, 5):
        y = y_pos(tick)
        parts.append(f'<line x1="{plot_l-2}" y1="{y}" x2="{plot_l+pw}" y2="{y}" stroke="#eee" stroke-width="0.8"/>')
        parts.append(
            f'<text x="{plot_l-5}" y="{y+3}" text-anchor="end" font-family="{font}" '
            f'font-size="9" fill="#666">{tick}</text>'
        )

    for i, (name, val) in enumerate(methods):
        x = plot_l + gap + i * (bar_w + gap)
        y_top = y_pos(val)
        y_base = plot_t + ph
        is_ours = i == len(methods) - 1
        fill = "#555555" if is_ours else "#dddddd"
        stroke = "#222222" if is_ours else "#888888"
        sw = 1.2 if is_ours else 0.8
        parts.append(
            f'<rect x="{x}" y="{y_top}" width="{bar_w}" height="{y_base-y_top}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        parts.append(
            f'<text x="{x+bar_w/2}" y="{y_top-5}" text-anchor="middle" font-family="{font}" '
            f'font-size="9.5" fill="#222">{val:.2f}%</text>'
        )
        label_lines = [name[:10], name[10:]] if len(name) > 10 else [name]
        ty = plot_t + ph + 14
        for j, ln in enumerate(label_lines):
            if ln:
                parts.append(
                    f'<text x="{x+bar_w/2}" y="{ty+j*11}" text-anchor="middle" font-family="{font}" '
                    f'font-size="8" fill="#333">{esc(ln)}</text>'
                )

    parts.append("</svg>")
    return "\n".join(parts)


def _make_transparent(png_path: Path, threshold: int = 252) -> None:
    from PIL import Image

    img = Image.open(png_path).convert("RGBA")
    data = img.getdata()
    cleaned = [
        (r, g, b, 0) if r > threshold and g > threshold and b > threshold else (r, g, b, a)
        for r, g, b, a in data
    ]
    img.putdata(cleaned)
    img.save(png_path, "PNG")


def export_pngs():
    try:
        import cairosvg
    except ImportError as exc:
        raise SystemExit(f"Missing dependency for PNG export: {exc}") from exc

    for name in ["figure1_confidence_fallback_flow", "figure2_main_results_bar"]:
        svg_path = OUT / f"{name}.svg"
        png_path = OUT / f"{name}.png"
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=3.0, background_color="white")
        _make_transparent(png_path)
        transparent_alias = OUT / f"{name}_transparent.png"
        transparent_alias.write_bytes(png_path.read_bytes())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f1 = OUT / "figure1_confidence_fallback_flow.svg"
    f2 = OUT / "figure2_main_results_bar.svg"
    f1.write_text(figure1_flowchart(), encoding="utf-8")
    f2.write_text(figure2_bar_chart(), encoding="utf-8")
    print(f"Wrote {f1}")
    print(f"Wrote {f2}")
    export_pngs()
    for name in ["figure1_confidence_fallback_flow", "figure2_main_results_bar"]:
        print(f"Wrote {OUT / f'{name}.png'}")
        print(f"Wrote {OUT / f'{name}_transparent.png'}")


if __name__ == "__main__":
    main()
