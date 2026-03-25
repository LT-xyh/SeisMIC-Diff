import torch
import torch.nn.functional as F
from ignite import metrics
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import transforms


class ValMetrics:
    """
    仅计算 PSNR / SSIM
    - 默认按 [-1, 1] 评估（data_range=2.0），也可传 1.0 表示 [0,1]
    - 若输入不在目标域，会自动映射：
        [0,1] -> [-1,1] （当 data_range=2.0）
        [-1,1] -> [0,1] （当 data_range=1.0）
    """

    def __init__(self, data_range: float = 2.0, device: str | None = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        assert data_range in (1.0, 2.0), "data_range 仅支持 1.0([0,1]) 或 2.0([-1,1])"
        self.prefer_m11 = (data_range == 2.0)

        self.psnr = metrics.PSNR(data_range=data_range, device=self.device)
        self.ssim = metrics.SSIM(data_range=data_range, device=self.device)

    # ---------- helpers ----------
    @staticmethod
    def _is_in_01(x: torch.Tensor) -> bool:
        # 宽松判定：几乎在 [0,1]
        return (x.min() >= -1e-3) and (x.max() <= 1.0 + 1e-3)

    def _harmonize_domain(self, x: torch.Tensor) -> torch.Tensor:
        """将张量统一到与 data_range 匹配的域。"""
        if self.prefer_m11:  # 目标 [-1,1]
            if self._is_in_01(x):
                x = x * 2.0 - 1.0
            return x.clamp(-1.0, 1.0)
        else:  # 目标 [0,1]
            if not self._is_in_01(x):
                x = (x + 1.0) / 2.0
            return x.clamp(0.0, 1.0)

    # ---------- public api ----------
    def update(self, real: torch.Tensor, recon: torch.Tensor):
        # dtype/device 对齐 + NaN/Inf 清理
        real = real.detach()
        recon = recon.detach()
        if recon.dtype != real.dtype:
            recon = recon.to(real.dtype)
        real = torch.nan_to_num(real, nan=0.0, posinf=1.0, neginf=-1.0).to(self.device)
        recon = torch.nan_to_num(recon, nan=0.0, posinf=1.0, neginf=-1.0).to(self.device)

        # 统一到与 data_range 一致的域
        real_h = self._harmonize_domain(real)
        recon_h = self._harmonize_domain(recon)

        # 仅更新 PSNR / SSIM
        self.psnr.update((recon_h, real_h))
        self.ssim.update((recon_h, real_h))

    def compute(self):
        return {
            'psnr': self.psnr.compute(),
            'ssim': self.ssim.compute(),
        }

    def reset(self):
        self.psnr.reset()
        self.ssim.reset()


def get_psnr(original, reconstruct, data_range=1.0):
    """
    计算 PSNR 值
    :param original: 输入 (batch_size, channel, height, width)
    :param reconstruct: 重建后 (batch_size, channel, height, width)
    :param data_range: 数据范围
    :return: psnr值
    """
    psnr_metric = _get_psnr_metric(data_range)
    psnr_metric.reset()
    psnr_metric.update((reconstruct, original))
    psnr = psnr_metric.compute()
    return psnr


def get_ssim(original, reconstruct, data_range=1.0):
    """
    计算 SSIM 值
    :param original: 输入 (batch_size, channel, height, width)
    :param reconstruct: 重建后 (batch_size, channel, height, width)
    :param data_range: 数据范围
    :return: SSIM 值
    """
    ssim_metric = _get_ssim_metric(data_range)
    ssim_metric.reset()
    ssim_metric.update((original, reconstruct))
    ssim = ssim_metric.compute()
    return ssim


def expand_image_to_size(image, target_size=128):
    """
    将图像扩展到指定大小，通过零填充实现
    """
    _, _, h, w = image.shape
    pad_h = (target_size - h) // 2
    pad_w = (target_size - w) // 2
    # 计算左右上下需要填充的大小
    padding = (pad_w, target_size - w - pad_w, pad_h, target_size - h - pad_h)
    return F.pad(image, padding, "constant", 0)


def get_is(reconstruct):
    """
    计算 Inception Score (IS)
    :param reconstruct: 生成的图像，形状为 (batch_size, channel, height, width)
    :return: Inception Score
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
    计算 Frechet Inception Distance (FID)
    :param original: 原始图像，形状为 (batch_size, channel, height, width)
    :param reconstruct: 重建后的图像，形状为 (batch_size, channel, height, width)
    :return: FID Score
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
    计算 KID
    :param original: 原始图像，形状为 (batch_size, channel, height, width)
    :param reconstruct: 重建后的图像，形状为 (batch_size, channel, height, width)
    :return: kid_mean, kid_std
    """
    if original.shape[3] < 128:
        original = expand_image_to_size(original, 128)
    if reconstruct.shape[3] < 128:
        reconstruct = expand_image_to_size(reconstruct, 128)
    if original.shape[1] == 1:
        original = original.repeat(1, 3, 1, 1)
    if reconstruct.shape[1] == 1:
        reconstruct = reconstruct.repeat(1, 3, 1, 1)
    # 数据预处理：将 float32 转换为 uint8
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte())
    ])
    # 对生成图像和真实图像进行预处理
    processed_generated_images = torch.stack([preprocess(img) for img in reconstruct])
    processed_real_images = torch.stack([preprocess(img) for img in original])

    num_samples = min(original.shape[0], reconstruct.shape[0])
    # 设置 subset_size 为样本数量的一半或更小
    subset_size = min(num_samples // 2, 10)  # 可以根据实际情况调整

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
    # 模拟原始图像和重建图像数据
    batch_size = 4
    channels = 1
    height = 70
    width = 70
    original_images = torch.rand(batch_size, channels, height, width)
    reconstructed_images = torch.rand(batch_size, channels, height, width)

    # 计算各项指标
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
    # 模拟原始图像和重建图像数据
    batch_size = 4
    channels = 1
    height = 70
    width = 70
    original_images = torch.rand(batch_size, channels, height, width).to('cuda:4')
    reconstructed_images = torch.rand(batch_size, channels, height, width).to('cuda:4')
    test_metrics = ValMetrics(device="cuda:4")
    test_metrics.reset()
    test_metrics.update(original_images, reconstructed_images)
    print(test_metrics.compute())
    test_metrics.reset()


if __name__ == "__main__":
    test_VAEMetrics()
