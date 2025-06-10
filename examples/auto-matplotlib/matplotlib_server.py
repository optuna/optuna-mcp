from __future__ import annotations

import os

import matplotlib.pyplot as plt
from mcp.server.fastmcp import FastMCP
import numpy as np


mcp = FastMCP("AutoMatplotlib")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 16
rng = np.random.RandomState(42)
X1 = rng.random(size=(3, 100, 40)) * 10 - 5
X2 = np.clip(rng.normal(size=(3, 100, 40)) * 2.5, -5, 5)
os.makedirs("figs/", exist_ok=True)


@mcp.tool()
def target_generator(trial_number: int, bbox_to_anchor_y: float) -> str:
    """
    Generate a plot figure based on the trial suggested by Optuna MCP.

    Args:
        trial_number: The trial number.
        bbox_to_anchor_y:
            The `bbox_to_anchor_y` stored in `params` of a `trial` suggested by Optuna MCP.
    """
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(10, 5), sharex=True)
    dx = np.arange(100) + 1
    for i, d in enumerate([5, 10, 20, 40]):
        ax = axes[i // 2][i % 2]

        def _subplot(ax: plt.Axes, values: list[list[float]]) -> plt.Line2D:
            cum_values = np.minimum.accumulate(values, axis=-1)
            mean = np.mean(cum_values, axis=0)
            stderr = np.std(cum_values, axis=0) / np.sqrt(len(values))
            (line,) = ax.plot(dx, mean)
            ax.fill_between(dx, mean - stderr, mean + stderr, alpha=0.2)
            return line

        lines = []
        ax.set_title(f"{d}D")
        lines.append(_subplot(ax, np.sum((X1[..., :d] - 2) ** 2, axis=-1)))
        lines.append(_subplot(ax, np.sum((X2[..., :d] - 2) ** 2, axis=-1)))

    fig.supxlabel("Number of Trials")
    fig.supylabel("Objective Values")
    labels = ["Uniform", "Gaussian"]
    loc = "lower center"
    bbox_to_anchor = (0.5, bbox_to_anchor_y)
    fig.legend(handles=lines, labels=labels, loc=loc, ncols=2, bbox_to_anchor=bbox_to_anchor)
    fig_path = f"figs/fig{trial_number}.png"
    plt.savefig(fig_path, bbox_inches="tight")
    return f"{fig_path} generated for Trial {trial_number} with {bbox_to_anchor_y=}"


if __name__ == "__main__":
    mcp.run()
