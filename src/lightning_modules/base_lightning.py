import os

import lightning
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from diffusers import EMAModel

from utils.metrics import ValMetrics
from utils.visualize import save_multiple_curves


class BaseLightningModule(lightning.LightningModule):
    """
    Shared Lightning base class.

    It centralizes metric setup, EMA checkpoint handling, and common
    visualization helpers used by the training, validation, and test loops.
    """

    def __init__(self, batch_size, lr=2e-5, data_range=2.0):
        super().__init__()
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.data_range = data_range
        self.train_metrics = None
        self.val_metrics = None
        self.test_metrics = None
        self._last_train_batch = None
        self._last_val_batch = None
        self.ema = None
        self._ema_parameters = None

    def _ema_params(self):
        params = self._ema_parameters if self._ema_parameters is not None else list(self.parameters())
        if self.ema is not None:
            try:
                param_device = next(p.device for p in params if p.requires_grad)
            except StopIteration:
                param_device = None
            if param_device is not None and len(self.ema.shadow_params) > 0:
                if self.ema.shadow_params[0].device != param_device:
                    self.ema.to(param_device)
        return params

    def setup(self, stage):
        if self.ema is not None:
            self.ema.to(self.device)
        self.train_metrics = ValMetrics(data_range=self.data_range, device=self.device)
        self.val_metrics = ValMetrics(data_range=self.data_range, device=self.device)
        self.test_metrics = ValMetrics(data_range=self.data_range, device=self.device)

    def on_fit_start(self):
        if self.ema is not None:
            self.ema.to(self.device)

    def training_step(self, batch, batch_idx):
        return

    def validation_step(self, batch, batch_idx):
        return

    def test_step(self, batch, batch_idx):
        return

    def on_train_epoch_end(self):
        """Log and reset training metrics once per epoch."""
        results = self.train_metrics.compute()
        for k, v in results.items():
            self.log(f"train/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        self.train_metrics.reset()

    def on_validation_epoch_start(self):
        if self.ema is not None:
            params = self._ema_params()
            self.ema.store(params)
            self.ema.copy_to(params)

    def on_validation_epoch_end(self):
        """Restore weights, log validation metrics, and export sample grids."""
        if self.ema is not None:
            self.ema.restore(self._ema_params())

        results = self.val_metrics.compute()
        for k, v in results.items():
            self.log(f"val/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        self.val_metrics.reset()

        if self._last_val_batch is not None and self.logger is not None:
            image_num = min(self.batch_size, 8)
            x, recon = self._last_val_batch
            x = x[:image_num]
            recon = recon[:image_num]

            # Stack inputs above reconstructions so TensorBoard shows pairs.
            comparison = torch.cat([x, recon], dim=0)
            grid = torchvision.utils.make_grid(comparison, nrow=image_num, normalize=True, value_range=(0, 1))
            self.logger.experiment.add_image("val/reconstruction_vs_input", grid, self.current_epoch)

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if "ema" not in checkpoint:
            return
        if self.ema is None:
            self.ema = EMAModel(parameters=self._ema_params())
        self.ema.load_state_dict(checkpoint["ema"])

    def on_test_start(self):
        if self.ema is not None:
            params = self._ema_params()
            self.ema.store(params)
            self.ema.copy_to(params)

    def on_test_end(self):
        if self.ema is not None:
            self.ema.restore(self._ema_params())

    def on_test_epoch_end(self):
        """Log and reset test metrics after the test epoch finishes."""
        results = self.test_metrics.compute()
        for k, v in results.items():
            self.log(f"test/{k}", v, prog_bar=False, on_step=False, on_epoch=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        """Create the default Adam optimizer over trainable parameters."""
        return torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)

    @staticmethod
    def normalize_to_neg_one_to_one(x):
        """
        Map tensors from ``[0, 1]`` to ``[-1, 1]``.

        Args:
            x: Input tensor assumed to be normalized to ``[0, 1]``.

        Returns:
            Tensor normalized to ``[-1, 1]``.
        """
        return x * 2 - 1

    @staticmethod
    def unnormalize_from_neg_one_to_one(x):
        """
        Map tensors from ``[-1, 1]`` back to ``[0, 1]``.

        Args:
            x: Input tensor assumed to be normalized to ``[-1, 1]``.

        Returns:
            Tensor normalized to ``[0, 1]``.
        """
        return (x + 1) / 2

    def save_batch_images(self, batch_idx, x, recon, save_dir, title2="Reconstructed"):
        os.makedirs(save_dir, exist_ok=True)
        batch_size = x.shape[0]
        x_numpys = x.cpu().detach().float().numpy().squeeze()
        recon_numpys = recon.cpu().detach().float().numpy().squeeze()
        for b in range(batch_size):
            filename = os.path.join(save_dir, f"{batch_idx}_{b}.svg")
            self.save_images(img1=x_numpys[b], img2=recon_numpys[b], filename=filename, title2=title2)
        return

    def save_batch_torch(self, batch_idx, recon, save_dir, well_log=None, well_threshold=-1.0, visualize_well=True):
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"{batch_idx}.pt")
        torch.save(recon.cpu().detach().float(), filename)
        if well_log is not None:
            well_dir = os.path.join(save_dir, "well_log")
            os.makedirs(well_dir, exist_ok=True)
            well_filename = os.path.join(well_dir, f"{batch_idx}.pt")
            torch.save(well_log.cpu().detach().float(), well_filename)
            if visualize_well:
                # Visualize the first sample's well-log mask and any active columns.
                well_np = well_log.detach().cpu().float().numpy()
                if well_np.ndim >= 3:
                    first = well_np[0]
                    if first.ndim == 3 and first.shape[0] == 1:
                        first = first[0]
                    img_path = os.path.join(well_dir, f"{batch_idx}_well_log.svg")
                    self.save_single_image(first, filename=img_path, title="Well Log", show=False, save=True,
                                           cmap="jet", extent=None, figsize=(5, 5), use_colorbar=False,
                                           x_label="Length (m)", y_label="Depth (m)")
                    mask = first >= well_threshold
                    cols = np.where(mask.any(axis=0))[0].tolist()
                    if cols:
                        curves = [first[:, col] for col in cols]
                        labels = [f"x={col}" for col in cols]
                        curve_path = os.path.join(well_dir, f"{batch_idx}_well_curves.svg")
                        save_multiple_curves(curves, labels=labels, filename=curve_path, title="Well log",
                                             x_label="Depth (m)", y_label="Velocity (normalized)", show=False,
                                             save=True, figsize=(6, 6), colors=None, linestyles=None)
        return

    @staticmethod
    def save_images(img1, img2, filename, title1="Ground truth", title2="Reconstructed", show=False, save=True,
                    cmap="jet", extent=[0, 700, 700, 0], figsize=(10, 5)):
        """
        Save a side-by-side comparison figure for two images.

        Args:
            img1: First image to display.
            img2: Second image to display.
            filename: Output figure path.
            title1: Title for the first subplot.
            title2: Title for the second subplot.
            show: Whether to display the figure interactively.
            save: Whether to write the figure to disk.
            cmap: Matplotlib colormap name.
            extent: Display extent passed to ``imshow``.
            figsize: Output figure size.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)

        im1 = axes[0].imshow(img1, aspect="auto", cmap=cmap, extent=extent)
        axes[0].set_title(title1)
        axes[0].set_xlabel("Length (m)")
        axes[0].set_ylabel("Depth (m)")

        axes[1].imshow(img2, aspect="auto", cmap=cmap, extent=extent)
        axes[1].set_title(title2)
        axes[1].set_xlabel("Length (m)")
        axes[1].set_ylabel("Depth (m)")

        cbar = fig.colorbar(im1, ax=axes, orientation="vertical")

        # Remap normalized colorbar ticks to the original velocity range.
        original_ticks = np.linspace(0, 1, 5)
        target_ticks = np.linspace(1500, 4500, 5)
        cbar.set_ticks(original_ticks)
        cbar.set_ticklabels([f"{int(t)}" for t in target_ticks])
        cbar.set_label("Velocity (m/s)")

        if save:
            plt.savefig(filename)

        if show:
            plt.show()

        plt.close(fig)

    @staticmethod
    def save_single_image(img, filename="", title="Velocity Model", show=False, save=True, cmap="jet",
                          extent=[0, 700, 700, 0], figsize=(5, 5), use_colorbar=True, x_label="Length (m)",
                          y_label="Depth (m)"):
        """
        Save a single image figure.

        Args:
            img: Image array to render.
            filename: Output figure path.
            title: Figure title.
            show: Whether to display the figure interactively.
            save: Whether to write the figure to disk.
            cmap: Matplotlib colormap name.
            extent: Display extent passed to ``imshow``.
            figsize: Output figure size.
            use_colorbar: Whether to add a colorbar.
            x_label: X-axis label.
            y_label: Y-axis label.
        """
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(img, aspect="auto", cmap=cmap, extent=extent)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        if use_colorbar:
            cbar = fig.colorbar(im, ax=ax, orientation="vertical")
            original_ticks = np.linspace(0, 1, 5)
            target_ticks = np.linspace(1500, 4500, 5)
            cbar.set_ticks(original_ticks)
            cbar.set_ticklabels([f"{int(t)}" for t in target_ticks])
            cbar.set_label("Velocity (m/s)")

        if save:
            plt.savefig(filename)

        if show:
            plt.show()

        plt.close(fig)

    @staticmethod
    def de_normalization(x, data_max, data_min):
        """
        Undo min-max normalization.

        Args:
            x: Normalized tensor.
            data_max: Maximum value of the original domain.
            data_min: Minimum value of the original domain.

        Returns:
            Tensor mapped back to the original value range.
        """
        return x * (data_max - data_min) + data_min

    @staticmethod
    def grad_loss_yx(pred, gt, w_y=1.0, w_x=0.5):
        dy = lambda x: x[:, :, 1:, :] - x[:, :, :-1, :]
        dx = lambda x: x[:, :, :, 1:] - x[:, :, :, :-1]
        Ly = F.l1_loss(dy(pred), dy(gt))
        Lx = F.l1_loss(dx(pred), dx(gt))
        return w_y * Ly + w_x * Lx

    @staticmethod
    def well_match_metrics(recon, target, well_log, well_threshold=-1.0):
        if well_log is None:
            return None

        # Restrict the metric to cells that belong to known well trajectories.
        mask = well_log >= well_threshold
        mask_f = mask.float()
        valid = mask_f.sum()
        device = recon.device
        if valid <= 0:
            zero = torch.tensor(0.0, device=device)
            return {"well_mae": zero, "well_mse": zero, "well_cc": zero, "well_count": zero}

        diff = recon - target
        well_mae = (diff.abs() * mask_f).sum() / valid
        well_mse = ((diff ** 2) * mask_f).sum() / valid

        # Pearson correlation over well positions only.
        tgt_vals = target[mask]
        rec_vals = recon[mask]
        tgt_centered = tgt_vals - tgt_vals.mean()
        rec_centered = rec_vals - rec_vals.mean()
        denom = torch.sqrt((tgt_centered ** 2).sum()) * torch.sqrt((rec_centered ** 2).sum())
        if denom <= 0:
            well_cc = torch.tensor(0.0, device=device)
        else:
            well_cc = (tgt_centered * rec_centered).sum() / denom
        return {"well_mae": well_mae, "well_mse": well_mse, "well_cc": well_cc, "well_count": valid}
