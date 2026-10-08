"""Render verified historical SVM confusion counts; optional matplotlib dependency."""
import argparse
from pathlib import Path
import numpy as np
if __package__:
    from .audit import audit
else:
    from audit import audit


def draw(destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    result = audit()
    cm = np.asarray(result["historical_final_window"]["confusion_matrix"])
    off = cm.copy()
    np.fill_diagonal(off, 0)
    assert np.argwhere(off == off.max()).tolist() == [[4, 1]] and off[4, 1] == 14
    fig, ax = plt.subplots(figsize=(4.6, 4.1))
    image = ax.imshow(cm, cmap="Blues", vmin=0, vmax=190)
    ax.set_xticks(range(5), list("ABCDE"))
    ax.set_yticks(range(5), list("ABCDE"))
    ax.set(xlabel="Predicted", ylabel="True")
    ax.set_title("Calibrated SVM: 887/950 windows\nHistorical reused holdout; red: maximum error", fontsize=10)
    ax.add_patch(Rectangle((0.5, 3.5), 1, 1, fill=False, edgecolor="red", linewidth=2))
    for i in range(5):
        for j in range(5):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10,
                    color="white" if cm[i, j] > 95 else "black")
    fig.colorbar(image, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("confusion_rerender.png"))
    draw(parser.parse_args().output)
