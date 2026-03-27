import torch
import torch.nn.functional as F
from ignite import metrics
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms


class ValMetrics:
    """
    Lightweight validation metrics container for PSNR and SSIM.

    By default ``data_range=2.0`` means the tensors are expected in ``[-1, 1]``.
    Passing ``data_range=1.0`` switches the target domain to ``[0, 1]``.
    Inputs outside the expected domain are remapped automatically.
    """

    def __init__(self, data_range: float = 2.0, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        assert data_range in (1.0, 2.0), "data_range only supports 1.0 ([0, 1]) or 2.0 ([-1, 1])."
        self.prefer_m11 = (data_range == 2.0)

        self.psnr = metrics.PSNR(data_range=data_range, device=self.device)
        self.ssim = metrics.SSIM(data_range=data_range, device=self.device)

    @staticmethod
    def _is_in_01(x: torch.Tensor) -> bool:
        # Use a slightly loose tolerance so small numerical drift does not trigger remapping.
        return (x.min() >= -1e-3) and (x.max() <= 1.0 + 1e-3)

    def _harmonize_domain(self, x: torch.Tensor) -> torch.Tensor:
        """Map tensors into the domain implied by ``data_range``."""
        if self.prefer_m11:
            if self._is_in_01(x):
                x = x * 2.0 - 1.0
            return x.clamp(-1.0, 1.0)
        else:
            if not self._is_in_01(x):
                x = (x + 1.0) / 2.0
            return x.clamp(0.0, 1.0)

    def update(self, real: torch.Tensor, recon: torch.Tensor):
        # Align dtype/device first, then sanitize NaN and Inf values before metric updates.
        real = real.detach()
        recon = recon.detach()
        if recon.dtype != real.dtype:
            recon = recon.to(real.dtype)
        real = torch.nan_to_num(real, nan=0.0, posinf=1.0, neginf=-1.0).to(self.device)
        recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=-1.0).to(self.device)

        real_h = self._harmonize_domain(real)
        recon_h = self._harmonize_domain(recon)

        self.psnr.update((recon_h, real_h))
        self.ssim.update((recon_h, real_h))

    def compute(self):
        return {
            "psnr": self.psnr.compute(),
            "ssim": self.ssim.compute(),
        }

    def reset(self):
        self.psnr.reset()
        self.ssim.reset()


def get_psnr(original, reconstruct, data_range=1.0):
    """
    Compute PSNR.

    Args:
        original: Tensor shaped like ``(batch, channel, height, width)``.
        reconstruct: Reconstructed tensor with the same shape.
        data_range: Expected value range.
    """
    psnr_metric = _get_psnr_metric(data_range)
    psnr_metric.reset()
    psnr_metric.update((reconstruct, original))
    psnr = psnr_metric.compute()
    return psnr


def get_ssim(original, reconstruct, data_range=1.0):
    """
    Compute SSIM.

    Args:
        original: Tensor shaped like ``(batch, channel, height, width)``.
        reconstruct: Reconstructed tensor with the same shape.
        data_range: Expected value range.
    """
    ssim_metric = _get_ssim_metric(data_range)
    ssim_metric.reset()
    ssim_metric.update((original, reconstruct))
    ssim = ssim_metric.compute()
    return ssim


def expand_image_to_size(image, target_size=128):
    """Zero-pad an image tensor so its spatial size becomes ``target_size x target_size``."""
    _, _, h, w = image.shape
    pad_h = (target_size - h) // 2
    pad_w = (target_size - w) // 2
    padding = (pad_w, target_size - w - pad_w, pad_h, target_size - h - pad_h)
    return F.pad(image, padding, "constant", 0)


def get_is(reconstruct):
    """
    Compute Inception Score for generated images.

    Args:
        reconstruct: Tensor shaped like ``(batch, channel, height, width)``.
    """
    if reconstruct.shape[3] < 128:
        reconstruct = expand_image_to_size(reconstruct, 128)
    if reconstruct.shape[1] == 1:
        reconstruct = reconstruct.repeat(1, 3, 1, 1)
    is_metric = _get_is_metric()
    is_metric.reset()
    is_metric.update(reconstruct)
    inception_score = is_metric.compute()
    return inception_score


def get_fid(original, reconstruct):
    """
    Compute Frechet Inception Distance.

    Args:
        original: Tensor shaped like ``(batch, channel, height, width)``.
        reconstruct: Tensor shaped like ``(batch, channel, height, width)``.
    """
    if original.shape[3] < 128:
        original = expand_image_to_size(original, 128)
    if reconstruct.shape[3] < 128:
        reconstruct = expand_image_to_size(reconstruct, 128)
    if original.shape[1] == 1:
        original = original.repeat(1, 3, 1, 1)
    if reconstruct.shape[1] == 1:
        reconstruct = reconstruct.repeat(1, 3, 1, 1)
    fid_metric = _get_fid_metric()
    fid_metric.reset()
    fid_metric.update((reconstruct, original))
    fid_score = fid_metric.compute()
    return fid_score


def get_kid(original, reconstruct):
    """
    Compute Kernel Inception Distance and return ``(kid_mean, kid_std)``.

    Args:
        original: Tensor shaped like ``(batch, channel, height, width)``.
        reconstruct: Tensor shaped like ``(batch, channel, height, width)``.
    """
    if original.shape[3] < 128:
        original = expand_image_to_size(original, 128)
    if reconstruct.shape[3] < 128:
        reconstruct = expand_image_to_size(reconstruct, 128)
    if original.shape[1] == 1:
        original = original.repeat(1, 3, 1, 1)
    if reconstruct.shape[1] == 1:
        reconstruct = reconstruct.repeat(1, 3, 1, 1)

    # Convert float tensors to uint8-like tensors because the metric expects image-like inputs.
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte()),
    ])
    processed_generated_images = torch.stack([preprocess(img) for img in reconstruct])
    processed_real_images = torch.stack([preprocess(img) for img in original])

    num_samples = min(original.shape[0], reconstruct.shape[0])
    subset_size = min(num_samples // 2, 10)

    kid = _get_kid_metric(subset_size)
    kid.update(processed_real_images, real=True)
    kid.update(processed_generated_images, real=False)
    kid_mean, kid_std = kid.compute()
    return kid_mean, kid_std


_PSNR_METRICS = {}
_SSIM_METRICS = {}
_IS_METRIC = None
_FID_METRIC = None
_KID_METRICS = {}


def _get_psnr_metric(data_range: float):
    metric = _PSNR_METRICS.get(data_range)
    if metric is None:
        metric = metrics.PSNR(data_range=data_range)
        _PSNR_METRICS[data_range] = metric
    return metric


def _get_ssim_metric(data_range: float):
    metric = _SSIM_METRICS.get(data_range)
    if metric is None:
        metric = metrics.SSIM(data_range=data_range)
        _SSIM_METRICS[data_range] = metric
    return metric


def _get_is_metric():
    global _IS_METRIC
    if _IS_METRIC is None:
        _IS_METRIC = metrics.InceptionScore()
    return _IS_METRIC


def _get_fid_metric():
    global _FID_METRIC
    if _FID_METRIC is None:
        _FID_METRIC = metrics.FID()
    return _FID_METRIC


def _get_kid_metric(subset_size: int):
    metric = _KID_METRICS.get(subset_size)
    if metric is None:
        metric = KernelInceptionDistance(subset_size=subset_size)
        _KID_METRICS[subset_size] = metric
    return metric


def test_evaluate():
    # Build random test tensors to sanity-check the metric helpers.
    batch_size = 4
    channels = 1
    height = 70
    width = 70
    original_images = torch.rand(batch_size, channels, height, width)
    reconstructed_images = torch.rand(batch_size, channels, height, width)

    psnr = get_psnr(original_images, reconstructed_images)
    print(f"PSNR: {psnr}")
    ssim = get_ssim(original_images, reconstructed_images)
    print(f"SSIM: {ssim}")

    is_score = get_is(reconstructed_images)
    print(f"Inception Score: {is_score}")

    fid_score = get_fid(original_images, reconstructed_images)
    print(f"FID Score: {fid_score}")

    kid_mean, kid_std = get_kid(original_images, reconstructed_images)
    print(f"KID Mean: {kid_mean}, KID Std: {kid_std}")


def test_VAEMetrics():
    # Build random test tensors to sanity-check the running metric container.
    batch_size = 4
    channels = 1
    height = 70
    width = 70
    original_images = torch.rand(batch_size, channels, height, width).to("cuda:4")
    reconstructed_images = torch.rand(batch_size, channels, height, width).to("cuda:4")
    test_metrics = ValMetrics(device="cuda:4")
    test_metrics.reset()
    test_metrics.update(original_images, reconstructed_images)
    print(test_metrics.compute())
    test_metrics.reset()


if __name__ == "__main__":
    test_VAEMetrics()
