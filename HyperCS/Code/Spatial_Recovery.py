"""
Spatial reconstruction of the compressed CS-CIS measurements using DRUNet-PnP.

Each compressed sensor measurement Z (q x W) is reconstructed into the
full-frame image Y_hat (H x W) by solving the regularized inverse problem
of Eq. (6) in the manuscript with the plug-and-play ADMM framework, in
which the proximal step is replaced by the pretrained DRUNet denoiser.
The row-wise averaging sensing operator Phi follows the readout model of
the CS-CIS at a spatial CR of 25.0% (CR = 4).

Usage (from the repository root):
    python Code/Spatial_Recovery.py                     # both letter samples
    python Code/Spatial_Recovery.py --sample letter_CS_CR4

We thank Dr. Kai Zhang for sharing the DRUNet architecture and pretrained
weights on GitHub (https://github.com/cszn/KAIR); the network definition in
Code/Util/external/kair is taken from his original codebase.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat, savemat

# ---------------------------------------------------------------- config ---
SAMPLES    = ['letter_CS_CR4', 'letter_CI_CR4']   # Data/<name>.mat
VAR_NAME   = 'CR4'    # variable name inside each .mat file

CR         = 4        # spatial compression factor (spatial CR = 25.0%)
IMG_SIZE   = 256      # full-frame height/width [pixels]
RHO        = 0.7      # ADMM penalty parameter
N_ITER     = 500      # number of PnP-ADMM iterations
SIGMA_MAX  = 50.0     # denoiser noise level schedule (start) [/255]
SIGMA_MIN  = 0.001    # denoiser noise level schedule (end)   [/255]

REPO_ROOT  = Path(__file__).resolve().parents[1]
KAIR_DIR   = REPO_ROOT / 'Code' / 'Util' / 'external' / 'kair'
MODEL_PATH = REPO_ROOT / 'model_zoo' / 'drunet_gray.pth'


# ------------------------------------------------------------- operators ---
def make_phi(n_rows: int, n_cols: int, group: int) -> np.ndarray:
    """Row-averaging sensing matrix: each output row averages `group` inputs."""
    B = np.zeros((n_rows, n_cols), dtype=np.float32)
    for i in range(n_rows):
        B[i, group * i: group * i + group] = 1.0
    return B / float(group)


def build_closed_form_xinv(cr: int, n_pix: int, rho: float) -> np.ndarray:
    """Closed-form inverse of (Phi^T Phi + rho I) for the block-averaging Phi."""
    inv = np.zeros((n_pix, n_pix), dtype=np.float64)
    coeff = 1.0 / (rho * cr * (rho * cr + 1.0))
    for s in range(0, n_pix, cr):
        e = s + cr
        inv[s:e, s:e] = (1.0 / rho) * np.eye(cr) - coeff * np.ones((cr, cr))
    return inv


def load_denoiser(device: torch.device):
    """Load the pretrained DRUNet denoiser (grayscale)."""
    sys.path.insert(0, str(KAIR_DIR))
    from network_unet import UNetRes
    sys.path.remove(str(KAIR_DIR))

    net = UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4,
                  act_mode='R', downsample_mode='strideconv',
                  upsample_mode='convtranspose', bias=False).to(device)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
    net.eval()
    return net


# ---------------------------------------------------------------- solver ---
def recon_drunet_pnp(measurements: np.ndarray, Phi: np.ndarray,
                     net, device: torch.device) -> np.ndarray:
    """PnP-ADMM reconstruction with the DRUNet denoiser prior.

    The output is the final iterate of the decreasing sigma schedule.
    """
    ATy  = Phi.T @ measurements
    Xinv = build_closed_form_xinv(CR, Phi.shape[1], RHO)
    x = CR * ATy
    z = x.copy()
    u = np.zeros_like(x)

    sigmas = np.linspace(SIGMA_MAX, SIGMA_MIN, N_ITER)
    residual = float('nan')

    print(f'[DRUNet-PnP] device={device}, rho={RHO}, iterations={N_ITER}')
    with torch.no_grad():
        for i, sig in enumerate(sigmas):
            # x-update: closed-form solution of the data-fidelity subproblem
            x = Xinv @ (ATy + RHO * (z - u))

            # z-update: DRUNet denoising of (x + u) at noise level sigma
            img_xu = np.clip(x + u, 0.0, 1.0).astype(np.float32)
            img_t  = torch.from_numpy(img_xu)[None, None].to(device)
            inp    = torch.cat([img_t, torch.full_like(img_t, float(sig / 255.0))], dim=1)
            z      = net(inp).clamp(0.0, 1.0)[0, 0].cpu().numpy().astype(np.float64)

            # dual update
            u = u + x - z

            residual = float(np.linalg.norm(x - z))

            if (i + 1) % 50 == 0:
                print(f'[DRUNet-PnP] iter {i+1:4d}  sigma {sig:6.2f}  '
                      f'residual {residual:.5f}')

    print(f'[DRUNet-PnP] finished {N_ITER} iterations, final residual={residual:.5f}')
    return np.clip(z, 0.0, 1.0)


# ------------------------------------------------------------------ main ---
def main() -> None:
    ap = argparse.ArgumentParser(description='DRUNet-PnP spatial reconstruction')
    ap.add_argument('--sample', choices=SAMPLES, default=None,
                    help='reconstruct a single sample (default: all)')
    args = ap.parse_args()

    samples = [args.sample] if args.sample else SAMPLES

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = load_denoiser(device)
    Phi = make_phi(IMG_SIZE // CR, IMG_SIZE, CR)

    for name in samples:
        in_path  = REPO_ROOT / 'Data' / f'{name}.mat'
        out_path = REPO_ROOT / 'Results' / f'{name}_recon.mat'

        meas = np.asarray(loadmat(str(in_path))[VAR_NAME], dtype=np.float64)
        expected = (IMG_SIZE // CR, IMG_SIZE)
        if meas.shape != expected:
            sys.exit(f'{name}: measurement shape {meas.shape} != expected {expected}')
        print(f'\n=== {name} ===')
        print(f'Measurement: {meas.shape} from {in_path}')

        rec = recon_drunet_pnp(meas, Phi, net, device)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        savemat(str(out_path), {VAR_NAME: rec.astype(np.float64)})
        print(f'Saved: {out_path}  ({rec.shape[0]}x{rec.shape[1]})')


if __name__ == '__main__':
    main()
