from typing import TYPE_CHECKING, Any

from numpy import abs as np_abs
from numpy import arange, argwhere, array, concatenate, quantile, transpose

from metabogeneva.barca.model import DecisionModelMPLS

if TYPE_CHECKING:
    from matplotlib.figure import Figure

    from metabogeneva.barca.classification import Classification
    from metabogeneva.barca.tree import DecisionNode


def classification_plot(
    classification: "Classification",
    individuals: bool = True,
    force_show: bool = False,
    figure_args: dict[str, Any] | None = None,
) -> "Figure":
    from matplotlib.pyplot import figure

    matrix, targets, sources = classification.matrix(not individuals)

    source_changes = concatenate(
        [[0], 1 + argwhere(sources[:-1] != sources[1:])[:, 0], [len(sources)]]
    )
    source_midpoints = (0.5 * source_changes[:-1] + 0.5 * source_changes[1:]).astype(
        int
    )

    matplotlib_figure_args: dict[str, Any] = dict(figsize=(5, 5))
    if figure_args is None:
        figure_args = {}
    matplotlib_figure_args.update(**figure_args)
    fig = figure(**matplotlib_figure_args)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(matrix, aspect="auto", cmap="gray", interpolation="none", vmin=0, vmax=1)
    ax.set_xticks(arange(0, len(targets)))
    ax.set_xticklabels(targets)
    ax.set_yticks(source_midpoints)
    ax.set_yticklabels(sources[source_midpoints])
    ax.hlines(source_changes - 0.5, -0.5, len(targets) - 0.5, color=(1, 0, 0, 0.5))
    ax.vlines(
        arange(0, len(targets) + 1) - 0.5,
        -0.5,
        len(sources) - 0.5,
        color=(1, 0, 0, 0.5),
    )

    if force_show:
        fig.show()

    return fig


def coefficient_plot(
    tree: "DecisionNode",
    force_show: bool = False,
    figure_args: dict[str, Any] | None = None,
) -> "Figure":
    if isinstance(tree.model, DecisionModelMPLS):
        raise NotImplementedError("Coefficient plots are not supported for mpls models")

    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.pyplot import figure
    from sklearn.cluster import AgglomerativeClustering

    def flatten_model(
        node: "DecisionNode", depth: int = 0
    ) -> list[tuple[int, "DecisionNode"]]:
        result = [(depth, node)]
        if node.accept_child is not None:
            result = flatten_model(node.accept_child, depth + 1) + result
        if node.reject_child is not None:
            result = result + flatten_model(node.reject_child, depth + 1)
        return result

    def shorten(name: str, maxlen: int = 20) -> str:
        return "\n".join(
            [
                piece[: maxlen - 3] + "..." if len(piece) > maxlen else piece
                for piece in name.split("\n")
            ]
        )

    def modelname(model: "DecisionNode") -> str:
        name = " vs "
        if len(model.accept_groups) == 1:
            name = model.accept_groups[0] + name
        else:
            name = "▼" + name
        if len(model.reject_groups) == 1:
            name = name + model.reject_groups[0]
        else:
            name = name + "▲"
        return name

    flattened = flatten_model(tree)
    shortnames = array([shorten(name) for name in tree.variables])
    modelnames = [modelname(model) for _, model in flattened]
    maxdepth = max(d for d, _ in flattened)

    coefficients = array([node.model.loadings for _, node in flattened])
    coefficientrange = quantile(np_abs(coefficients), 1.0)

    hca = AgglomerativeClustering(linkage="ward", metric="euclidean")
    hca.fit(transpose(coefficients))
    n_vars = len(shortnames)
    max_hca_i = 2 * (n_vars - 1)

    def hca_indices(i: int = max_hca_i) -> list[int]:
        a, b = hca.children_[i - n_vars]
        result: list[int] = []
        if a < n_vars:
            result = [a] + result
        else:
            result = hca_indices(a) + result
        if b < n_vars:
            result = result + [b]
        else:
            result = result + hca_indices(b)
        return result

    hca_sorting = list(reversed(hca_indices()))

    shortnames = shortnames[hca_sorting]
    coefficients = coefficients[:, hca_sorting]

    matplotlib_figure_args: dict[str, Any] = dict(figsize=(9, 5))
    if figure_args is None:
        figure_args = {}
    matplotlib_figure_args.update(**figure_args)
    fig = figure(**matplotlib_figure_args)
    ax = fig.add_subplot(1, 1, 1)

    cmap = LinearSegmentedColormap.from_list(
        "", ["#ee2222", "#440000", "#000000", "#004400", "#22ee22"]
    )
    ax.imshow(
        coefficients,
        vmin=-coefficientrange,
        vmax=+coefficientrange,
        cmap=cmap,
        zorder=0,
    )
    for y, (d, model) in enumerate(flattened):
        x = d - maxdepth - 2

        ax.plot(
            [-maxdepth - 4, x],
            [y, y],
            color="white",
            linestyle="dashed",
            linewidth=0.5,
            zorder=-4,
        )
        line_shift = 0.3
        if len(model.accept_groups) == 1:
            group = model.accept_groups[0]
            if group in tree.group_metadata:
                color = tree.group_metadata[group].color
            else:
                color = "#AAAAAA"
            ax.plot(
                [x, x + 1.0, -0.5],
                [y, y - line_shift, y - line_shift],
                color=color,
                linewidth=4,
                zorder=-3,
            )
        if len(model.reject_groups) == 1:
            group = model.reject_groups[0]
            if group in tree.group_metadata:
                color = tree.group_metadata[group].color
            else:
                color = "#AAAAAA"
            ax.plot(
                [x, x + 1.0, -0.5],
                [y, y + line_shift, y + line_shift],
                color=color,
                linewidth=4,
                zorder=-2,
            )

        accept_y = None
        reject_y = None
        for i, (_, other) in enumerate(flattened):
            if model.accept_child == other:
                accept_y = i
            if model.reject_child == other:
                reject_y = i
        if accept_y is not None:
            ax.plot(
                [x, x + 1],
                [y, accept_y],
                color="#cccccc",
                linewidth=1,
                marker="o",
                markersize=6,
                zorder=-1,
                markerfacecolor="black",
            )
        if reject_y is not None:
            ax.plot(
                [x, x + 1],
                [y, reject_y],
                color="#cccccc",
                linewidth=1,
                marker="o",
                markersize=6,
                zorder=-1,
                markerfacecolor="black",
            )
    ax.plot(
        [-0.5, -0.5],
        [-0.5, -0.5 + len(shortnames)],
        color="#000000",
        linestyle="solid",
        linewidth=1,
        zorder=1,
    )
    ax.set_xticks(range(len(shortnames)))
    ax.set_xticklabels(shortnames, rotation=90)
    ax.set_yticks(range(len(modelnames)))
    ax.set_yticklabels(labels=modelnames)
    ax.set_xlim((-0.5 - maxdepth - 3, -0.5 + len(shortnames)))
    ax.set_ylim((-0.5, -0.5 + len(flattened)))
    ax.set_facecolor("#000000")
    for spine in ax.spines.values():
        spine.set_edgecolor("#000000")

    if force_show:
        fig.show()

    return fig
