import torch
import torch.nn as nn


def _batched_tridiagonal_solve(a, b, c, d):
    """
    Solve tridiagonal systems with Thomas algorithm.

    a: (..., n-1) lower diagonal
    b: (..., n)   main diagonal
    c: (..., n-1) upper diagonal
    d: (..., n)   RHS

    Returns x: (..., n)
    """
    # Make working copies
    a = a.clone()
    b = b.clone()
    c = c.clone()
    d = d.clone()

    n = b.shape[-1]
    # Forward sweep
    for i in range(1, n):
        w = a[..., i - 1] / (b[..., i - 1] + 1e-12)
        b[..., i] = b[..., i] - w * c[..., i - 1]
        d[..., i] = d[..., i] - w * d[..., i - 1]

    # Back substitution
    x = torch.zeros_like(d)
    x[..., n - 1] = d[..., n - 1] / (b[..., n - 1] + 1e-12)
    for i in range(n - 2, -1, -1):
        x[..., i] = (d[..., i] - c[..., i] * x[..., i + 1]) / (b[..., i] + 1e-12)
    return x


def _tikhonov_smooth_1d(v0, lam):
    """
    1D Tikhonov smoothing along last dimension:
        min_v ||v - v0||^2 + lam ||D v||^2,  D = first-difference.
    v0: (..., n)
    lam: float (>=0)

    Returns v: (..., n)
    """
    if lam <= 0:
        return v0

    n = v0.shape[-1]
    device = v0.device
    dtype = v0.dtype

    # D^T D has diag [1,2,...,2,1] and offdiag [-1,...,-1]
    # So (I + lam D^T D) is tridiagonal.
    b = torch.empty((*v0.shape[:-1], n), device=device, dtype=dtype)
    b[..., 0] = 1.0 + lam
    b[..., 1:-1] = 1.0 + 2.0 * lam
    b[..., -1] = 1.0 + lam

    a = torch.full((*v0.shape[:-1], n - 1), -lam, device=device, dtype=dtype)  # lower
    c = torch.full((*v0.shape[:-1], n - 1), -lam, device=device, dtype=dtype)  # upper

    return _batched_tridiagonal_solve(a, b, c, v0)


class SmoothDix(nn.Module):
    """
    rms_vel (B,1,nt,nx) -> time_vel (B,1,nt,nx) -> depth_vel (B,1,nz,nx)

    - Dix interval velocity in time domain
    - Tikhonov smoothness constraint along time (per x-trace)
    - Time->Depth mapping using z(t)=∫ v/2 dt, then resample on regular depth grid
    """
    def __init__(
        self,
        dt: float = 0.001,
        dz: float = 10.0,
        nz: int = 70,
        vmin: float = 1200.0,
        vmax: float = 6000.0,
        smooth_lambda: float = 10.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.dt = float(dt)
        self.dz = float(dz)
        self.nz = int(nz)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.smooth_lambda = float(smooth_lambda)
        self.eps = float(eps)

    def forward(self, rms_vel: torch.Tensor):
        """
        rms_vel: (B,1,nt,nx)
        returns:
          time_vel:  (B,1,nt,nx)  (interval vel in time domain)
          depth_vel: (B,1,nz,nx)  (interval vel on depth grid)
        """
        assert rms_vel.ndim == 4 and rms_vel.shape[1] == 1, "Expect (B,1,nt,nx)"
        B, _, nt, nx = rms_vel.shape
        device = rms_vel.device
        dtype = rms_vel.dtype

        vrms = rms_vel[:, 0]  # (B,nt,nx)

        # Build time vector t_i = i*dt
        t = torch.arange(nt, device=device, dtype=dtype) * self.dt  # (nt,)

        # y(t) = t * vrms^2
        y = (vrms ** 2) * t.view(1, nt, 1)

        # Discrete derivative dy/dt (backward diff for stability)
        dy = y[:, 1:, :] - y[:, :-1, :]
        vint2 = dy / self.dt  # (B,nt-1,nx)

        # Handle i=0: set v_int(0) ~ v_rms(0) (common practical choice)
        vint2_0 = (vrms[:, 0:1, :] ** 2).clamp(self.vmin**2, self.vmax**2)

        # Concatenate to length nt
        vint2 = torch.cat([vint2_0, vint2], dim=1)  # (B,nt,nx)

        # Clip to physical range, sqrt
        vint2 = vint2.clamp(self.vmin**2, self.vmax**2)
        time_vel = torch.sqrt(vint2 + self.eps)  # (B,nt,nx)

        # Smoothness constraint along time for each trace (B,nx,nt)
        time_vel_bnxt = time_vel.permute(0, 2, 1).contiguous()  # (B,nx,nt)
        time_vel_sm = _tikhonov_smooth_1d(time_vel_bnxt, self.smooth_lambda)  # (B,nx,nt)
        time_vel_sm = time_vel_sm.permute(0, 2, 1).contiguous()  # (B,nt,nx)

        # Time->Depth cumulative map: z(t_i) = sum_{k<=i} v(k)*dt/2
        dz_dt = 0.5 * time_vel_sm * self.dt  # (B,nt,nx)
        z_curve = torch.cumsum(dz_dt, dim=1)  # (B,nt,nx), monotone increasing

        # Depth grid (nz,)
        z_grid = (torch.arange(self.nz, device=device, dtype=dtype) * self.dz)  # (nz,)

        # Interpolate v(z_grid) by inverting z_curve(t) using searchsorted (per trace)
        # Work in shape (B,nx,nt) for searchsorted
        z_bxnt = z_curve.permute(0, 2, 1).contiguous()      # (B,nx,nt)
        v_bxnt = time_vel_sm.permute(0, 2, 1).contiguous()  # (B,nx,nt)

        # idx = first index where z_curve >= z_grid
        # values broadcast to (B,nx,nz)
        z_bxnt_contiguous = z_bxnt.contiguous()
        z_grid_contiguous = z_grid.view(1, 1, -1).expand(B, nx, -1).contiguous()
        idx = torch.searchsorted(z_bxnt_contiguous, z_grid_contiguous, right=False)
        # Clamp idx to [0, nt-1]
        idx1 = idx.clamp(0, nt - 1)
        idx0 = (idx1 - 1).clamp(0, nt - 1)

        # Gather z0,z1,v0,v1
        z0 = torch.gather(z_bxnt, dim=2, index=idx0)
        z1 = torch.gather(z_bxnt, dim=2, index=idx1)
        v0 = torch.gather(v_bxnt, dim=2, index=idx0)
        v1 = torch.gather(v_bxnt, dim=2, index=idx1)

        # Linear interp in depth
        denom = (z1 - z0).clamp_min(self.eps)
        w = (z_grid.view(1, 1, -1) - z0) / denom
        w = w.clamp(0.0, 1.0)
        v_z = v0 + w * (v1 - v0)  # (B,nx,nz)

        depth_vel = v_z.permute(0, 2, 1).contiguous().unsqueeze(1)  # (B,1,nz,nx)
        time_vel_out = time_vel_sm.unsqueeze(1)  # (B,1,nt,nx)
        return depth_vel, time_vel_out

if __name__ == '__main__':
    rms_vel = torch.randn(3, 1, 1000, 70) * 3000 + 1500
    model = SmoothDix()
    depth_vel, aux = model(rms_vel)
    print(depth_vel.shape)