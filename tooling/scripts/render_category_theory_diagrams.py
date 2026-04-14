from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from matplotlib import patches
from matplotlib import pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = ROOT / "shared" / "figures" / "category-theory" / "sources" / "diagrams.json"
OUTPUT_DIR = ROOT / "shared" / "figures" / "category-theory" / "rendered"


def load_spec() -> dict:
    with SOURCE_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_color(name: str | None, style: dict) -> str:
    if name == "accent":
        return style["accent_color"]
    if name == "guide":
        return style["guide_color"]
    return style["arrow_color"]


def setup_axes(style: dict) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(style["canvas_width"], style["canvas_height"]))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def draw_node(ax: plt.Axes, node: dict, style: dict) -> None:
    width = node.get("width", 0.12)
    height = node.get("height", 0.12)
    rect = patches.FancyBboxPatch(
        (node["x"] - width / 2, node["y"] - height / 2),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=style["line_width"],
        edgecolor=style["node_edge"],
        facecolor=style["node_fill"],
    )
    ax.add_patch(rect)
    ax.text(node["x"], node["y"], node["label"], ha="center", va="center", fontsize=style["font_size"])


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], arrow: dict, style: dict) -> None:
    rad = arrow.get("curve", 0.0)
    color = resolve_color(arrow.get("color"), style)
    patch = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=13,
        linewidth=style["line_width"],
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=18,
        shrinkB=18,
    )
    ax.add_patch(patch)
    label = arrow.get("label", "")
    if label:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        offset_y = 0.05 + abs(rad) * 0.12
        offset_x = -rad * 0.10
        if end[0] < start[0]:
            offset_x *= -1
        if end[1] < start[1] and abs(end[0] - start[0]) < 0.1:
            offset_x = 0.04
            offset_y = 0.0
        ax.text(mx + offset_x, my + offset_y, label, ha="center", va="center", fontsize=style["font_size"] - 1, color=color)


def render_node_diagram(ax: plt.Axes, diagram: dict, style: dict) -> None:
    nodes = {node["id"]: node for node in diagram["nodes"]}
    for node in diagram["nodes"]:
        draw_node(ax, node, style)
    for arrow in diagram["arrows"]:
        start_node = nodes[arrow["from"]]
        end_node = nodes[arrow["to"]]
        draw_arrow(ax, (start_node["x"], start_node["y"]), (end_node["x"], end_node["y"]), arrow, style)


def draw_wire(ax: plt.Axes, y: float, label_left: str, label_right: str, style: dict) -> None:
    ax.plot([0.08, 0.92], [y, y], color=style["arrow_color"], linewidth=style["line_width"])
    ax.text(0.04, y, label_left, ha="center", va="center", fontsize=style["font_size"])
    ax.text(0.96, y, label_right, ha="center", va="center", fontsize=style["font_size"])


def render_string_diagram(ax: plt.Axes, diagram: dict, style: dict) -> None:
    for wire in diagram["wires"]:
        draw_wire(ax, wire["y"], wire["label_left"], wire["label_right"], style)

    boxes = {}
    for box in diagram["boxes"]:
        width = box.get("width", 0.12)
        height = box.get("height", 0.12)
        rect = patches.Rectangle(
            (box["x"] - width / 2, box["y"] - height / 2),
            width,
            height,
            linewidth=style["line_width"],
            edgecolor=style["node_edge"],
            facecolor=style["node_fill"],
        )
        ax.add_patch(rect)
        ax.text(box["x"], box["y"], box["label"], ha="center", va="center", fontsize=style["font_size"] - 1)
        boxes[(box["x"], box["y"])] = box

    for wire in diagram["wires"]:
        for box in diagram["boxes"]:
            if abs(wire["y"] - box["y"]) < 0.02:
                ax.plot([0.08, box["x"] - 0.06], [wire["y"], wire["y"]], color=style["arrow_color"], linewidth=style["line_width"])
                ax.plot([box["x"] + 0.06, 0.92], [wire["y"], wire["y"]], color=style["arrow_color"], linewidth=style["line_width"])

    for join in diagram.get("joins", []):
        wire = diagram["wires"][join["from_wire"]]
        box = diagram["boxes"][join["to_box"]]
        target_y = box["y"] + 0.06 if join["anchor"] == "top" else box["y"] - 0.06
        ax.plot([0.56, box["x"]], [wire["y"], target_y], color=style["guide_color"], linewidth=style["line_width"])

    if "bypass" in diagram:
        bypass = diagram["bypass"]
        ax.plot([0.08, 0.92], [bypass["y"], bypass["y"]], color=style["accent_color"], linewidth=style["line_width"])
        ax.plot([0.08, 0.08], [0.50, bypass["y"]], color=style["accent_color"], linewidth=style["line_width"])
        ax.plot([0.92, 0.92], [0.50, bypass["y"]], color=style["accent_color"], linewidth=style["line_width"])
        ax.text(0.50, bypass["y"] + 0.05, bypass["label"], ha="center", va="center", fontsize=style["font_size"] - 1, color=style["accent_color"])


def render_diagram(diagram: dict, style: dict) -> None:
    fig, ax = setup_axes(style)
    ax.text(0.02, 0.96, diagram["title"], ha="left", va="top", fontsize=style["title_size"], color=style["node_edge"], weight="bold")

    if diagram["kind"] == "node":
        render_node_diagram(ax, diagram, style)
    elif diagram["kind"] == "string":
        render_string_diagram(ax, diagram, style)
    else:
        raise ValueError(f"Unsupported diagram kind: {diagram['kind']}")

    ax.text(0.02, 0.06, diagram["caption"], ha="left", va="bottom", fontsize=style["font_size"] - 2, color=style["guide_color"], wrap=True)

    svg_path = OUTPUT_DIR / f"{diagram['slug']}.svg"
    png_path = OUTPUT_DIR / f"{diagram['slug']}.png"
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    spec = load_spec()
    style = spec["style"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for diagram in spec["diagrams"]:
        render_diagram(diagram, style)
    print(f"Rendered {len(spec['diagrams'])} diagrams to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
