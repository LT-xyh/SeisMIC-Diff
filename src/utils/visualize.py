import os

import numpy as np
from matplotlib import pyplot as plt


def save_visualize_image(img, filename="", title="Image", show=False, save=True, cmap="jet", extent=[0, 700, 700, 0],
                         figsize=(5, 5), use_colorbar=True, x_label="Length (m)", y_label="Depth (m)"):
    """
    Save a single image with optional axes and colorbar formatting.

    Args:
        img: Image array to render.
        filename: Output figure path.
        title: Unused reserved title argument kept for compatibility.
        show: Whether to display the figure.
        save: Whether to save the figure.
        cmap: Matplotlib colormap name.
        extent: Display extent passed to ``imshow``.
        figsize: Output figure size.
        use_colorbar: Whether to add a colorbar.
        x_label: X-axis label.
        y_label: Y-axis label.
    """
    del title
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(img, aspect="auto", cmap=cmap, extent=extent, vmax=1, vmin=-1)
    # Keep titles and ticks disabled so exported figures stay compact.
    # ax.set_title(title)
    ax.set_xlabel(x_label, loc="center")
    ax.set_ylabel(y_label, loc="center")
    # ax.set_xticks(extent[:2])
    # ax.set_yticks(extent[2:])
    if use_colorbar:
        cbar = fig.colorbar(im, ax=ax, orientation="vertical")
        original_ticks = np.linspace(0, 1, 5)
        target_ticks = np.linspace(1500, 4500, 5)
        cbar.set_ticks(original_ticks)
        cbar.set_ticklabels([f"{int(t)}" for t in target_ticks])
        cbar.set_label("Velocity (m/s)")

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    if save:
        plt.savefig(filename)

    if show:
        plt.show()

    plt.close(fig)


def save_multiple_curves(curves, labels=None, filename="", title="Well log", x_label="Depth (m)",
                         y_label="Velocity (m/s)", show=True, save=False, figsize=(6, 6), colors=None,
                         linestyles=None):
    """
    Plot multiple curves on the same figure.

    Args:
        curves: List of curve arrays.
        labels: Optional legend labels.
        filename: Output figure path.
        title: Unused reserved title argument kept for compatibility.
        x_label: X-axis label.
        y_label: Y-axis label.
        show: Whether to display the figure.
        save: Whether to save the figure.
        figsize: Output figure size.
        colors: Optional list of line colors.
        linestyles: Optional list of line styles.
    """
    del title
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if labels is None:
        # labels = [f"Well log {i + 1}" for i in range(len(curves))]
        labels = ["Well log"]

    if colors is None:
        colors = [None] * len(curves)

    if linestyles is None:
        linestyles = ["-"] * len(curves)

    for curve, label, color, linestyle in zip(curves, labels, colors, linestyles):
        x_values = range(0, len(curve) * 10, 10)
        ax.plot(x_values, curve, label=label, color=color, linestyle=linestyle, linewidth=2)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if save:
        plt.savefig(filename)

    if show:
        plt.show()

    plt.close(fig)


def test_visualize(dataset_name):
    save_path = "images/datasets/"
    os.makedirs(save_path, exist_ok=True)
    show = True
    save = False
    depth_vel = np.load(f"data/openfwi/{dataset_name}/depth_vel/27006.npy").squeeze()
    depth_vel = (depth_vel - depth_vel.min()) / (depth_vel.max() - depth_vel.min())
    save_visualize_image(depth_vel, filename=os.path.join(save_path, f"{dataset_name}_depth_vel.svg"),
                         title="Depth Velocity", show=show, save=save, cmap="jet", extent=[0, 700, 700, 0],
                         figsize=(5, 5), use_colorbar=True, x_label="Length (m)", y_label="Depth (m)")
    rms_vel = np.load(f"data/openfwi/{dataset_name}/rms_vel/27006.npy").squeeze()
    save_visualize_image(rms_vel, filename=os.path.join(save_path, f"{dataset_name}_time _vel.svg"),
                         title="Time Velocity", show=show, save=save, cmap="jet", extent=[0, 700, 1, 0],
                         figsize=(5, 5), use_colorbar=False, x_label="Length (m)", y_label="Time (s)")
    migrated_image = np.load(f"data/openfwi/{dataset_name}/migrated_image/27006.npy").squeeze()
    save_visualize_image(migrated_image, filename=os.path.join(save_path, f"{dataset_name}_migrated_image.svg"),
                         title="Migrated Image", show=show, save=save, cmap="gray", extent=[0, 700, 1, 0],
                         figsize=(5, 5), use_colorbar=False, x_label="Length (m)", y_label="Time (s)")

    horizon = np.load(f"data/openfwi/{dataset_name}/horizon/27006.npy").squeeze()
    save_visualize_image(horizon, filename=os.path.join(save_path, f"{dataset_name}_horizon.svg"), title="Horizon",
                         show=show, save=save, cmap="gray", extent=[0, 700, 700, 0], figsize=(5, 5),
                         use_colorbar=False, x_label="Length (m)", y_label="Depth (m)")
    well_log = depth_vel[:, 20]
    save_multiple_curves([well_log], filename=os.path.join(save_path, f"{dataset_name}_well_log.svg"), show=show,
                         save=save)


if __name__ == "__main__":
    for dataset_name in ["CurveVelA", "FlatVelA", "FlatVelB", "CurveVelB"]:
        test_visualize(dataset_name)
        break
