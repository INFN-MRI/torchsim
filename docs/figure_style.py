"""Figure style shared by the documentation gallery and explanation figures.

Every figure is drawn on a transparent canvas, in tones that clear 3:1 against
the light theme's white and the dark theme's background alike, so a figure reads
the same in either. A gallery script sets no font size, DPI or colour of its
own: it imports the colours it needs from here, sizes its figures as fractions
of :data:`PAGE_WIDTH`, and puts its legend outside the axes with
:func:`legend_outside`.
"""

from __future__ import annotations

from cycler import cycler

#: Documentation-column width used by full-width figures, in inches.
PAGE_WIDTH = 8.6

#: Axis furniture and text.
INK = "#717c8b"
MUTED = "#7d8996"
FAINT = "#7d899659"

#: The foreground the explanation figures draw text and axes in.
FOREGROUND = INK

#: Categorical hues, assigned in order.
SERIES = (
    "#2a78d6",
    "#eb6834",
    "#169869",
    "#b47900",
    "#c5678b",
    "#008300",
    "#7d6fd4",
    "#e34948",
)

STYLE = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.edgecolor": "none",
    "savefig.transparent": True,
    "figure.dpi": 110,
    "savefig.dpi": 110,
    "figure.constrained_layout.use": True,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "legend.title_fontsize": 11,
    "figure.titlesize": 13,
    "legend.frameon": False,
    "text.color": INK,
    "axes.labelcolor": MUTED,
    "axes.titlecolor": INK,
    "axes.edgecolor": FAINT,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,
    "grid.color": FAINT,
    "grid.alpha": 1.0,
    "legend.labelcolor": INK,
    "axes.prop_cycle": cycler(color=list(SERIES)),
    "image.interpolation": "nearest",
}


def apply() -> None:
    """Apply the documentation figure style to Matplotlib."""
    import matplotlib

    matplotlib.rcParams.update(STYLE)


def reset(gallery_conf, fname) -> None:
    """Reset Matplotlib, then apply the documentation style for one example."""
    import matplotlib

    matplotlib.rcdefaults()
    apply()


def legend_outside(where, ncols=1, title=None):
    """Put the legend to the right of the axes, never over the data.

    ``where`` is a figure, whose panels all show the same series and share one
    legend, or an axis, which gets a legend of its own beside it.
    """
    if hasattr(where, "add_subplot"):
        seen = {}
        for axis in where.axes:
            for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
                seen.setdefault(label, handle)
        return where.legend(
            list(seen.values()),
            list(seen),
            loc="outside right upper",
            ncols=ncols,
            title=title,
        )
    return where.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        ncols=ncols,
        title=title,
    )
