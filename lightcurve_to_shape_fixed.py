import os
import time
import random
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.special import sph_harm_y
from torch.utils.data import DataLoader, TensorDataset

# ============================================================
# Configuration
# ============================================================
NUM_RAYS = 642
SH_L_MAX = 7
NUM_PHASES = 36
NUM_TRAIN = 2000
NUM_VAL = 500
BATCH_SIZE = 64
EPOCHS = 30
LAMBDAS = [0.0, 0.01, 0.1, 1.0]
DATA_SEED = 2026

# All synthetic and DAMIT shapes are normalized to mean radial distance = 1.
RADIUS_MIN = 0.1
RADIUS_MAX = 3.0

# Simplified rotational Lambertian forward model.
# The body rotates around the z-axis while Sun and observer directions are fixed.
SUN_DIR = (1.0, 0.0, 0.0)
VIEW_DIR = (0.0, 0.0, 1.0)

CONVEX_IDS = [
    "1620", "433", "1862", "1566", "6489",
    "1981", "3200", "2063", "4179", "1685",
]
NON_CONVEX_IDS = [
    "5920", "5921", "5922", "5923", "5924",
    "5925", "5926", "5927", "5928", "5929",
]
MESH_FIGURE_ID = "5920"
CERES_LC_PATH = None

SAVE_MODELS = True
MODEL_DIR = "trained_models_v2"


def set_seed(seed=DATA_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Reproducibility is preferred for the poster experiment.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_damit_url(model_id):
    return (
        "https://damit.cuni.cz/projects/damit/generated_files/open/"
        f"AsteroidModel/{model_id}/shape.obj"
    )


# ============================================================
# Geometry: 642-direction icosphere
# ============================================================
def create_icosphere(subdivisions, device):
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=np.float32)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ], dtype=np.int64)

    midpoint_cache = {}

    def get_midpoint(i1, i2, v_list):
        edge = tuple(sorted((i1, i2)))
        if edge in midpoint_cache:
            return midpoint_cache[edge]
        mid = (v_list[i1] + v_list[i2]) / 2.0
        v_list.append(mid)
        idx = len(v_list) - 1
        midpoint_cache[edge] = idx
        return idx

    v_list, f_list = list(verts), list(faces)
    for _ in range(subdivisions):
        new_faces = []
        midpoint_cache.clear()
        for f in f_list:
            a = get_midpoint(f[0], f[1], v_list)
            b = get_midpoint(f[1], f[2], v_list)
            c = get_midpoint(f[2], f[0], v_list)
            new_faces.extend([
                [f[0], a, c],
                [f[1], b, a],
                [f[2], c, b],
                [a, b, c],
            ])
        f_list = new_faces

    verts_np = np.array(v_list, dtype=np.float32)
    verts_np = verts_np / np.linalg.norm(verts_np, axis=1, keepdims=True)
    return (
        torch.tensor(verts_np, dtype=torch.float32, device=device),
        torch.tensor(f_list, dtype=torch.long, device=device),
    )


# ============================================================
# Spherical harmonics
# IMPORTANT: the SH basis is built on the SAME 642 directions
# used by the radial shape representation.
# ============================================================
def build_spherical_harmonics_basis(directions, l_max=SH_L_MAX):
    dirs_np = directions.detach().cpu().numpy() if isinstance(directions, torch.Tensor) else directions
    x, y, z = dirs_np[:, 0], dirs_np[:, 1], dirs_np[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)

    basis_list = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m < 0:
                y_lm = np.sqrt(2.0) * (-1) ** m * sph_harm_y(
                    l, abs(m), theta, phi
                ).imag
            elif m == 0:
                y_lm = sph_harm_y(l, 0, theta, phi).real
            else:
                y_lm = np.sqrt(2.0) * (-1) ** m * sph_harm_y(
                    l, m, theta, phi
                ).real
            basis_list.append(y_lm)

    return torch.tensor(
        np.stack(basis_list, axis=1), dtype=torch.float32, device=directions.device
    )


def build_sh_projection_matrix(sh_basis):
    # 642 x 64 basis -> 64 x 642 pseudo-inverse.
    return torch.linalg.pinv(sh_basis)


# ============================================================
# Ray / triangle intersection
# ============================================================
def ray_mesh_intersection(
    ray_dirs,
    mesh_verts,
    mesh_faces,
    chunk_size=100,
    return_hit_mask=False,
):
    """
    Moller-Trumbore style ray-triangle intersection.

    For each ray, the nearest positive intersection is used as the radial
    distance. For non-star-shaped geometry, some rays may have no intersection
    from the chosen origin. Those rays are reported by hit_mask; a temporary
    median-radius fallback keeps the pipeline numerically stable.
    """
    device = ray_dirs.device
    if mesh_verts.dim() == 2:
        mesh_verts = mesh_verts.unsqueeze(0)

    B = mesh_verts.shape[0]
    V = ray_dirs.shape[0]
    all_radii = []
    all_hits = []

    for b_start in range(0, B, chunk_size):
        b_end = min(b_start + chunk_size, B)
        verts_chunk = mesh_verts[b_start:b_end]
        v0 = verts_chunk[:, mesh_faces[:, 0]]
        v1 = verts_chunk[:, mesh_faces[:, 1]]
        v2 = verts_chunk[:, mesh_faces[:, 2]]

        edge1 = v1 - v0
        edge2 = v2 - v0

        D = ray_dirs.view(1, V, 1, 3)
        e1 = edge1.unsqueeze(1)
        e2 = edge2.unsqueeze(1)
        v0_b = v0.unsqueeze(1)

        pvec = torch.cross(D, e2, dim=-1)
        det = torch.sum(e1 * pvec, dim=-1)
        det_safe = torch.where(
            det.abs() > 1e-8,
            det,
            torch.full_like(det, 1e-8),
        )
        inv_det = 1.0 / det_safe

        tvec = -v0_b
        u = torch.sum(tvec * pvec, dim=-1) * inv_det
        qvec = torch.cross(tvec, e1, dim=-1)
        v = torch.sum(D * qvec, dim=-1) * inv_det
        t = torch.sum(e2 * qvec, dim=-1) * inv_det

        valid = (
            (det.abs() > 1e-6)
            & (u >= 0.0)
            & (v >= 0.0)
            & (u + v <= 1.0)
            & (t > 1e-7)
        )

        t_masked = torch.where(
            valid,
            t,
            torch.full_like(t, float("inf")),
        )
        min_t, _ = torch.min(t_masked, dim=-1)
        hit_mask = torch.isfinite(min_t)

        # Keep missing rays numerically stable, but explicitly report them.
        finite_values = min_t[hit_mask]
        if finite_values.numel() > 0:
            fallback = finite_values.median()
        else:
            fallback = torch.tensor(1.0, device=device)
        radii_chunk = torch.where(hit_mask, min_t, fallback)

        all_radii.append(radii_chunk)
        all_hits.append(hit_mask)

    radii = torch.cat(all_radii, dim=0)
    hit_mask = torch.cat(all_hits, dim=0)
    radii = torch.clamp(radii, RADIUS_MIN, 5.0)

    if return_hit_mask:
        return radii, hit_mask
    return radii


def normalize_radii_mean(radii, eps=1e-8):
    mean_r = radii.mean(dim=-1, keepdim=True)
    return radii / (mean_r + eps)


# ============================================================
# Differentiable rotational Lambertian renderer
# ============================================================
def _make_rotation_z(angle, device, dtype):
    c = torch.cos(angle)
    s = torch.sin(angle)
    R = torch.stack([
        torch.stack([c, -s, torch.zeros_like(c)]),
        torch.stack([s, c, torch.zeros_like(c)]),
        torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)]),
    ])
    return R.to(device=device, dtype=dtype)


def render_lambert_lightcurves(
    radii,
    ray_dirs,
    faces,
    num_phases=NUM_PHASES,
    sun_dir=SUN_DIR,
    view_dir=VIEW_DIR,
):
    """
    Simplified rotational Lambertian photometry.

    - The shape rotates around the z-axis.
    - Sun and observer directions are fixed in the world frame.
    - Per-face flux is proportional to
        area * max(n . sun, 0) * max(n . view, 0).

    This is a differentiable photometric consistency model, not a full
    asteroid light-scattering/occlusion simulator.
    """
    dev = radii.device
    dtype = radii.dtype
    B = radii.shape[0]

    verts = radii.unsqueeze(-1) * ray_dirs.unsqueeze(0)
    v0 = verts[:, faces[:, 0]]
    v1 = verts[:, faces[:, 1]]
    v2 = verts[:, faces[:, 2]]

    cross_prod = torch.cross(v1 - v0, v2 - v0, dim=-1)
    face_areas = 0.5 * torch.norm(cross_prod, dim=-1)
    face_normals = nn.functional.normalize(cross_prod, dim=-1, eps=1e-8)

    sun = torch.tensor(sun_dir, dtype=dtype, device=dev)
    view = torch.tensor(view_dir, dtype=dtype, device=dev)

    # Avoid duplicate phase 0 and 2pi samples.
    angles = torch.arange(num_phases, device=dev, dtype=dtype) * (2.0 * torch.pi / num_phases)

    lcs = []
    for angle in angles:
        R = _make_rotation_z(angle, dev, dtype)
        rotated_normals = torch.matmul(face_normals, R.T)

        cos_i = torch.clamp(torch.sum(rotated_normals * sun.view(1, 1, 3), dim=-1), min=0.0)
        cos_e = torch.clamp(torch.sum(rotated_normals * view.view(1, 1, 3), dim=-1), min=0.0)

        flux = torch.sum(face_areas * cos_i * cos_e, dim=1)
        lcs.append(flux)

    lcs_tensor = torch.stack(lcs, dim=1)
    min_v = lcs_tensor.min(dim=1, keepdim=True).values
    max_v = lcs_tensor.max(dim=1, keepdim=True).values
    lcs_norm = (lcs_tensor - min_v) / (max_v - min_v + 1e-6)
    return lcs_norm.unsqueeze(1)


# ============================================================
# Synthetic ellipsoid data
# ============================================================
def generate_synthetic_shapes(
    num_samples,
    ray_dirs,
    faces,
    sh_basis,
    sh_pinv,
    seed=DATA_SEED,
):
    """
    Generate ellipsoids and normalize each shape to mean radius = 1.
    This removes the absolute-scale ambiguity because the normalized
    light curve does not contain absolute size information.
    """
    device = ray_dirs.device
    rng = np.random.default_rng(seed)

    a = torch.tensor(
        rng.uniform(1.0, 2.2, size=(num_samples, 1, 1)),
        dtype=torch.float32,
        device=device,
    )
    b = torch.tensor(
        rng.uniform(0.6, 1.3, size=(num_samples, 1, 1)),
        dtype=torch.float32,
        device=device,
    )
    c = torch.tensor(
        rng.uniform(0.4, 0.9, size=(num_samples, 1, 1)),
        dtype=torch.float32,
        device=device,
    )

    ellipsoid_verts = ray_dirs.unsqueeze(0) * torch.cat([a, b, c], dim=-1)
    radii = ray_mesh_intersection(
        ray_dirs,
        ellipsoid_verts,
        faces,
        chunk_size=100,
    )
    radii = normalize_radii_mean(radii)

    coeffs = radii @ sh_pinv.T
    return radii, coeffs


# ============================================================
# Model
# ============================================================
class Improved1DCNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 8, 512),
            nn.GELU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.fc(self.features(x))


# ============================================================
# Training / prediction
# ============================================================
def train_model(
    model,
    train_loader,
    sh_basis,
    ray_dirs,
    faces,
    name,
    lambda_phys=0.0,
    is_sh=False,
    epochs=EPOCHS,
):
    device = ray_dirs.device
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    criterion_mse = nn.MSELoss()

    model.train()
    print(f"\n--- Training {name} ---")
    for epoch in range(epochs):
        total_loss = 0.0
        total_target = 0.0
        total_phys = 0.0
        n_batches = 0

        for lcs, radii, coeffs in train_loader:
            lcs = lcs.to(device)
            radii = radii.to(device)
            coeffs = coeffs.to(device)

            optimizer.zero_grad()
            pred = model(lcs)

            if is_sh:
                pred_r = pred @ sh_basis.T
                loss_target = criterion_mse(pred, coeffs)
            else:
                pred_r = pred
                loss_target = criterion_mse(pred_r, radii)

            if lambda_phys > 0.0:
                pred_r_for_render = torch.clamp(
                    pred_r, min=RADIUS_MIN, max=RADIUS_MAX
                )
                pred_lc = render_lambert_lightcurves(
                    pred_r_for_render, ray_dirs, faces
                )
                loss_phys = criterion_mse(pred_lc, lcs)
                loss = loss_target + lambda_phys * loss_phys
            else:
                loss_phys = torch.zeros((), device=device)
                loss = loss_target

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_target += loss_target.item()
            total_phys += loss_phys.item()
            n_batches += 1

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1:02d}/{epochs:02d} | "
                f"Total: {total_loss / n_batches:.6f} | "
                f"Target: {total_target / n_batches:.6f} | "
                f"Phys: {total_phys / n_batches:.6f}"
            )

    return model


def predict_radii(model, lc, sh_basis, is_sh_model):
    pred = model(lc)
    pred_radii = pred @ sh_basis.T if is_sh_model else pred
    return torch.clamp(pred_radii, min=RADIUS_MIN, max=RADIUS_MAX)


@torch.no_grad()
def evaluate_model_on_synthetic(
    model,
    lcs,
    gt_radii,
    sh_basis,
    ray_dirs,
    faces,
    is_sh_model,
):
    model.eval()
    pred_r = predict_radii(model, lcs, sh_basis, is_sh_model)
    cd, radii_mse, lc_mse = compute_metrics(pred_r, gt_radii, ray_dirs, faces)
    return {
        "Chamfer_Distance": cd,
        "Radii_MSE": radii_mse,
        "LC_MSE": lc_mse,
    }


# ============================================================
# Metrics
# ============================================================
def compute_chamfer_distance(pc1, pc2):
    dist = torch.cdist(pc1, pc2, p=2) ** 2
    return (
        torch.mean(dist.min(dim=2).values)
        + torch.mean(dist.min(dim=1).values)
    ).item()


def compute_metrics(pred_r, gt_r, ray_dirs, faces):
    radii_mse = nn.functional.mse_loss(pred_r, gt_r).item()

    gt_lc = render_lambert_lightcurves(gt_r, ray_dirs, faces)
    pred_lc = render_lambert_lightcurves(pred_r, ray_dirs, faces)
    lc_mse = nn.functional.mse_loss(pred_lc, gt_lc).item()

    pc_gt = gt_r.unsqueeze(-1) * ray_dirs.unsqueeze(0)
    pc_pred = pred_r.unsqueeze(-1) * ray_dirs.unsqueeze(0)
    cd_val = compute_chamfer_distance(pc_pred, pc_gt)
    return cd_val, radii_mse, lc_mse


# ============================================================
# DAMIT OBJ loading / normalization
# ============================================================
def load_damit_obj(obj_file_path_or_url):
    verts = []
    faces = []

    if obj_file_path_or_url.startswith(("http://", "https://")):
        with urllib.request.urlopen(obj_file_path_or_url, timeout=30) as req:
            lines = [line.decode("utf-8", errors="ignore") for line in req.readlines()]
    else:
        with open(obj_file_path_or_url, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if parts[0] == "v" and len(parts) >= 4:
            verts.append([
                float(parts[1]),
                float(parts[2]),
                float(parts[3]),
            ])
        elif parts[0] == "f" and len(parts) >= 4:
            # OBJ may contain v/vt/vn, so keep only the vertex index.
            idx = []
            for p in parts[1:]:
                v_idx = int(p.split("/")[0])
                if v_idx < 0:
                    v_idx = len(verts) + v_idx + 1
                idx.append(v_idx - 1)
            # Triangulate polygons using a fan.
            for j in range(1, len(idx) - 1):
                faces.append([idx[0], idx[j], idx[j + 1]])

    if not verts or not faces:
        raise ValueError(f"Invalid or empty OBJ: {obj_file_path_or_url}")

    return (
        torch.tensor(verts, dtype=torch.float32),
        torch.tensor(faces, dtype=torch.long),
    )


def load_normalized_damit_shape(url, device):
    verts, faces = load_damit_obj(url)
    verts = verts - verts.mean(dim=0, keepdim=True)
    mean_r = torch.norm(verts, dim=1).mean()
    if mean_r <= 0:
        raise ValueError(f"Degenerate DAMIT model: {url}")
    verts = verts / mean_r
    return verts.to(device), faces.to(device)


def evaluate_single_damit_obj_normalized(
    url,
    model,
    ray_dirs,
    mesh_faces,
    sh_basis,
    is_sh_model=False,
):
    model.eval()
    damit_verts, damit_faces = load_normalized_damit_shape(url, ray_dirs.device)

    gt_radii, hit_mask = ray_mesh_intersection(
        ray_dirs,
        damit_verts.unsqueeze(0),
        damit_faces,
        chunk_size=100,
        return_hit_mask=True,
    )

    hit_fraction = hit_mask.float().mean().item()
    if hit_fraction < 0.99:
        print(
            f"  [Warning] {url.split('/')[-2]} radial coverage: "
            f"{hit_fraction * 100:.2f}% of rays hit the mesh. "
            "Missing rays use the median-hit radius."
        )

    # DAMIT input is converted through the same forward model used for training.
    damit_lc = render_lambert_lightcurves(gt_radii, ray_dirs, mesh_faces)

    with torch.no_grad():
        pred_radii = predict_radii(model, damit_lc, sh_basis, is_sh_model)

    cd, radii_mse, lc_mse = compute_metrics(
        pred_radii, gt_radii, ray_dirs, mesh_faces
    )
    return {
        "Chamfer_Distance": cd,
        "Radii_MSE": radii_mse,
        "LC_MSE": lc_mse,
        "Hit_Fraction": hit_fraction,
    }


def evaluate_single_damit_with_retry(
    url,
    model,
    ray_dirs,
    mesh_faces,
    sh_basis,
    is_sh_model=False,
    max_retries=5,
):
    for attempt in range(max_retries):
        try:
            return evaluate_single_damit_obj_normalized(
                url,
                model,
                ray_dirs,
                mesh_faces,
                sh_basis,
                is_sh_model=is_sh_model,
            )
        except Exception as e:
            if "503" in str(e) or "HTTP Error" in str(e):
                model_id = url.split("/")[-2]
                print(
                    f"  [Retry {attempt + 1}/{max_retries}] "
                    f"Waiting 3s for ID {model_id}..."
                )
                time.sleep(3)
            else:
                raise

    raise RuntimeError(
        f"Failed to fetch {url} after {max_retries} attempts due to server error."
    )


def run_full_eval(
    convex_urls,
    non_convex_urls,
    model_dict,
    ray_dirs,
    mesh_faces,
    sh_basis,
):
    records = []
    datasets = [
        ("Convex", convex_urls),
        ("Non-Convex", non_convex_urls),
    ]

    for group_name, url_list in datasets:
        print(f"\n>>> Processing {group_name} Group (Total: {len(url_list)})")
        for url in url_list:
            model_id = url.split("/")[-2]
            for model_name, (model, is_sh) in model_dict.items():
                try:
                    res = evaluate_single_damit_with_retry(
                        url,
                        model,
                        ray_dirs,
                        mesh_faces,
                        sh_basis,
                        is_sh_model=is_sh,
                    )
                    records.append({
                        "Group": group_name,
                        "Asteroid_ID": model_id,
                        "Model": model_name,
                        "CD": res["Chamfer_Distance"],
                        "Radii_MSE": res["Radii_MSE"],
                        "LC_MSE": res["LC_MSE"],
                        "Hit_Fraction": res["Hit_Fraction"],
                    })
                except Exception as e:
                    print(
                        f"  └ Failed ID {model_id} ({model_name}): {e}"
                    )

    return pd.DataFrame(records)


# ============================================================
# Summary / model selection
# ============================================================
def format_mean_std(mean_series, std_series):
    std_series = std_series.fillna(0.0)
    return [
        f"{m:.5f} ± {s:.5f}"
        for m, s in zip(mean_series, std_series)
    ]


def summarize_results(df):
    summary = df.groupby(["Group", "Model"], sort=False).agg(
        CD_mean=("CD", "mean"),
        CD_std=("CD", "std"),
        Radii_MSE_mean=("Radii_MSE", "mean"),
        Radii_MSE_std=("Radii_MSE", "std"),
        LC_MSE_mean=("LC_MSE", "mean"),
        LC_MSE_std=("LC_MSE", "std"),
        Hit_Fraction_mean=("Hit_Fraction", "mean"),
    ).reset_index()

    return pd.DataFrame({
        "Group": summary["Group"],
        "Model": summary["Model"],
        "Chamfer Distance": format_mean_std(
            summary["CD_mean"], summary["CD_std"]
        ),
        "Radii MSE": format_mean_std(
            summary["Radii_MSE_mean"], summary["Radii_MSE_std"]
        ),
        "LC MSE": format_mean_std(
            summary["LC_MSE_mean"], summary["LC_MSE_std"]
        ),
        "Hit Fraction Mean": summary["Hit_Fraction_mean"].map(
            lambda x: f"{x:.4f}"
        ),
    })


def select_lambda_from_validation(val_records):
    """
    Select λ using synthetic validation only.

    We avoid selecting λ from the DAMIT evaluation set, so DAMIT remains
    an external/generalization evaluation rather than a tuning set.

    The selection score is the average of normalized CD and LC-MSE across
    the SH models. This is a simple, explicitly defined multi-objective
    criterion; it is not a claim of a universally optimal λ.
    """
    df = pd.DataFrame(val_records)
    df_sh = df[df["Model"].str.startswith("SH")].copy()
    if df_sh.empty:
        return None, df

    cd_scale = max(df_sh["CD"].max(), 1e-8)
    lc_scale = max(df_sh["LC_MSE"].max(), 1e-8)
    df_sh["Selection_Score"] = (
        df_sh["CD"] / cd_scale + df_sh["LC_MSE"] / lc_scale
    ) / 2.0

    score_df = df_sh.groupby("Model", as_index=False)["Selection_Score"].mean()
    best_row = score_df.loc[score_df["Selection_Score"].idxmin()]
    selected_model = best_row["Model"]

    lambda_value = None
    if "λ=" in selected_model:
        try:
            lambda_value = float(
                selected_model.split("λ=")[1].replace(" (No Phys)", "")
            )
        except ValueError:
            lambda_value = None

    return lambda_value, pd.merge(
        df,
        score_df,
        on="Model",
        how="left",
    )


# ============================================================
# Plotting
# ============================================================
def render_mesh_surface(ax, verts, faces, title, elev=20, azim=45):
    verts_np = verts.detach().cpu().numpy() if torch.is_tensor(verts) else verts
    faces_np = faces.detach().cpu().numpy() if torch.is_tensor(faces) else faces

    poly3d = Poly3DCollection(
        verts_np[faces_np],
        alpha=0.9,
        edgecolor="#333333",
        linewidths=0.1,
    )
    poly3d.set_facecolor("#4C72B0")
    ax.add_collection3d(poly3d)

    extent = np.max(np.abs(verts_np))
    extent = max(1.05, float(extent) * 1.15)
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.axis("off")


def plot_lambda_mesh_comparison(
    target_id,
    models_dict,
    ray_dirs,
    mesh_faces,
    sh_basis,
):
    damit_verts, damit_faces = load_normalized_damit_shape(
        make_damit_url(target_id), ray_dirs.device
    )
    gt_radii, hit_mask = ray_mesh_intersection(
        ray_dirs,
        damit_verts.unsqueeze(0),
        damit_faces,
        chunk_size=100,
        return_hit_mask=True,
    )
    if hit_mask.float().mean() < 0.99:
        print(
            f"[Mesh figure] DAMIT ID {target_id}: "
            f"hit fraction = {hit_mask.float().mean().item():.4f}"
        )

    damit_lc = render_lambert_lightcurves(
        gt_radii, ray_dirs, mesh_faces
    )

    num_models = len(models_dict) + 1
    fig = plt.figure(figsize=(3.5 * num_models, 4))

    gt_verts = (gt_radii.unsqueeze(-1) * ray_dirs).squeeze(0)
    ax_gt = fig.add_subplot(1, num_models, 1, projection="3d")
    render_mesh_surface(
        ax_gt,
        gt_verts,
        mesh_faces,
        f"DAMIT reference\n(radialized, ID: {target_id})",
    )

    for idx, (name, (model, is_sh)) in enumerate(
        models_dict.items(), start=2
    ):
        model.eval()
        with torch.no_grad():
            pred_r = predict_radii(
                model, damit_lc, sh_basis, is_sh
            )
            pred_v = (
                pred_r.unsqueeze(-1) * ray_dirs
            ).squeeze(0)

        ax = fig.add_subplot(1, num_models, idx, projection="3d")
        render_mesh_surface(ax, pred_v, mesh_faces, name)

    plt.tight_layout()
    output = f"poster_mesh_surface_lambda_{target_id}_v2.png"
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output}")


def plot_lambda_graphs(
    csv_path="damit_eval_lambda_comparison_v2.csv",
    output_png="poster_graphs_triplet_v2.png",
):
    df_res = pd.read_csv(csv_path)
    df_sh = df_res[df_res["Model"].str.startswith("SH")].copy()

    order = [
        "SH λ=0 (No Phys)",
        "SH λ=0.01",
        "SH λ=0.1",
        "SH λ=1.0",
    ]
    label_map = {
        "SH λ=0 (No Phys)": "0 (No Phys)",
        "SH λ=0.01": "0.01",
        "SH λ=0.1": "0.1",
        "SH λ=1.0": "1.0",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for group in ["Convex", "Non-Convex"]:
        sub = df_sh[df_sh["Group"] == group]
        means = sub.groupby("Model")["CD"].mean().reindex(order)
        axes[0].plot(
            range(len(order)),
            means.values,
            marker="o",
            linewidth=2,
            label=group,
        )
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels([label_map[x] for x in order])
    axes[0].set_title("Shape accuracy vs. physics-loss weight λ")
    axes[0].set_xlabel("Physics loss weight λ")
    axes[0].set_ylabel("Chamfer Distance (CD) ↓")
    axes[0].legend()

    for group in ["Convex", "Non-Convex"]:
        sub = df_sh[df_sh["Group"] == group]
        means = sub.groupby("Model")["LC_MSE"].mean().reindex(order)
        axes[1].plot(
            range(len(order)),
            means.values,
            marker="o",
            linewidth=2,
            label=group,
        )
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels([label_map[x] for x in order])
    axes[1].set_title("Photometric fit vs. physics-loss weight λ")
    axes[1].set_xlabel("Physics loss weight λ")
    axes[1].set_ylabel("Lightcurve MSE (LC MSE) ↓")
    axes[1].legend()

    summary_df = df_res.groupby("Model", as_index=False).agg(
        CD=("CD", "mean"),
        LC_MSE=("LC_MSE", "mean"),
    )
    markers = {
        "Direct CNN": "X",
        "SH λ=0 (No Phys)": "o",
        "SH λ=0.01": "^",
        "SH λ=0.1": "*",
        "SH λ=1.0": "s",
    }

    for _, row in summary_df.iterrows():
        model_name = row["Model"]
        axes[2].scatter(
            row["LC_MSE"],
            row["CD"],
            label=model_name,
            s=180 if model_name == "SH λ=0.1" else 100,
            marker=markers.get(model_name, "o"),
        )

    axes[2].set_title("Photometric fit vs. shape fidelity")
    axes[2].set_xlabel("Lightcurve MSE (LC MSE) ↓")
    axes[2].set_ylabel("Chamfer Distance (CD) ↓")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_png}")


# ============================================================
# Ceres real-data preprocessing
# ============================================================
def preprocess_ceres_lc_aligned(
    filepath,
    device,
    target_len=NUM_PHASES,
    value_kind="magnitude",
    use_first_half=True,
):
    """
    Preprocess an observed Ceres light curve.

    value_kind="magnitude" converts astronomical magnitude to relative flux
    before normalization. If the second column is already flux/brightness,
    set value_kind="flux".

    The existing experiment used the first half of the file as one period;
    that behavior is retained by default. For a publication-grade result,
    replace this with period-based phase folding when the observation period
    and timestamps are available.
    """
    if not filepath or not os.path.exists(filepath):
        print(f"Ceres file not found: {filepath}")
        return None

    raw_values = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    float(parts[0])
                    raw_values.append(float(parts[1]))
                except ValueError:
                    continue

    if not raw_values:
        print("No valid Ceres data points found.")
        return None

    raw_values = np.asarray(raw_values, dtype=np.float64)

    if value_kind.lower() == "magnitude":
        # Smaller magnitude = brighter -> convert to flux-like quantity.
        values = 10.0 ** (-0.4 * raw_values)
    elif value_kind.lower() == "flux":
        values = raw_values.copy()
    else:
        raise ValueError("value_kind must be 'magnitude' or 'flux'.")

    if use_first_half:
        half_len = len(values) // 2
        if half_len < 2:
            raise ValueError("Ceres light curve is too short for half-period extraction.")
        values = values[:half_len]

    values = (values - values.min()) / (
        values.max() - values.min() + 1e-8
    )

    lc_tensor = torch.tensor(
        values,
        dtype=torch.float32,
        device=device,
    ).view(1, 1, -1)
    lc_resampled = torch.nn.functional.interpolate(
        lc_tensor,
        size=target_len,
        mode="linear",
        align_corners=False,
    ).squeeze()

    # Keep the historical phase alignment, but describe it as alignment rather
    # than a physical spin-axis determination.
    max_idx = torch.argmax(lc_resampled).item()
    lc_aligned = torch.roll(
        lc_resampled,
        shifts=-max_idx,
        dims=0,
    )
    lc_aligned = (lc_aligned - lc_aligned.min()) / (
        lc_aligned.max() - lc_aligned.min() + 1e-8
    )

    return (
        lc_aligned.unsqueeze(0).unsqueeze(0).to(device),
        lc_aligned.detach().cpu().numpy(),
    )


def plot_ceres_comparison(
    ceres_tensor,
    lc_plot_data,
    model_direct,
    model_sh,
    ray_dirs,
    mesh_faces,
    sh_basis,
    gt_verts=None,
    gt_faces=None,
    output_png="poster_ceres_real_domain_aligned_comparison_v2.png",
):
    model_direct.eval()
    model_sh.eval()

    with torch.no_grad():
        pred_r_direct = predict_radii(
            model_direct, ceres_tensor, sh_basis, False
        )
        pred_v_direct = (
            pred_r_direct.unsqueeze(-1) * ray_dirs
        ).squeeze(0)

        pred_r_sh = predict_radii(
            model_sh, ceres_tensor, sh_basis, True
        )
        pred_v_sh = (
            pred_r_sh.unsqueeze(-1) * ray_dirs
        ).squeeze(0)

    fig = plt.figure(figsize=(20, 5))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.plot(
        np.linspace(0, 1, NUM_PHASES),
        lc_plot_data,
        "o-",
        linewidth=2,
        markersize=5,
        label="Observed Ceres LC",
    )
    ax1.set_title(
        "Input: observed Ceres LC\n"
        "(preprocessed to 36 samples)",
        fontsize=12,
        fontweight="bold",
    )
    ax1.set_xlabel("Normalized phase [0, 1]")
    ax1.set_ylabel("Normalized brightness")
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    def setup_3d_ax(ax, title):
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.view_init(elev=20, azim=45)
        ax.axis("off")

    ax2 = fig.add_subplot(1, 4, 2, projection="3d")
    if gt_verts is not None and gt_faces is not None:
        render_mesh_surface(ax2, gt_verts, gt_faces, "")
    else:
        ax2.text(0, 0, 0, "No DAMIT reference supplied", ha="center")
    setup_3d_ax(ax2, "(a) Reference shape")

    ax3 = fig.add_subplot(1, 4, 3, projection="3d")
    render_mesh_surface(ax3, pred_v_direct, mesh_faces, "")
    setup_3d_ax(ax3, "(b) Direct CNN (λ=0)")

    ax4 = fig.add_subplot(1, 4, 4, projection="3d")
    render_mesh_surface(ax4, pred_v_sh, mesh_faces, "")
    setup_3d_ax(ax4, "(c) SH + photometric loss (λ=0.1)")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.show()
    print(f"Saved: {output_png}")


# ============================================================
# Main experiment
# ============================================================
def main():
    set_seed(DATA_SEED)
    os.makedirs(MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --------------------------------------------------------
    # 1. Common geometry and SH basis
    # --------------------------------------------------------
    ray_directions, mesh_faces = create_icosphere(
        subdivisions=3,
        device=device,
    )
    assert ray_directions.shape[0] == NUM_RAYS

    # CRITICAL FIX: SH basis uses the exact same directions as the 642 radii.
    sh_basis = build_spherical_harmonics_basis(
        ray_directions,
        l_max=SH_L_MAX,
    )
    sh_pinv = build_sh_projection_matrix(sh_basis)

    print(f"Ray directions: {ray_directions.shape}")
    print(f"Faces: {mesh_faces.shape}")
    print(f"SH basis: {sh_basis.shape}")

    # --------------------------------------------------------
    # 2. Synthetic train / validation data
    # --------------------------------------------------------
    print("\nGenerating synthetic training data...")
    train_radii, train_coeffs = generate_synthetic_shapes(
        NUM_TRAIN,
        ray_directions,
        mesh_faces,
        sh_basis,
        sh_pinv,
        seed=DATA_SEED,
    )
    train_lcs = render_lambert_lightcurves(
        train_radii,
        ray_directions,
        mesh_faces,
    )

    print("Generating synthetic validation data...")
    val_radii, val_coeffs = generate_synthetic_shapes(
        NUM_VAL,
        ray_directions,
        mesh_faces,
        sh_basis,
        sh_pinv,
        seed=DATA_SEED + 1,
    )
    val_lcs = render_lambert_lightcurves(
        val_radii,
        ray_directions,
        mesh_faces,
    )

    train_loader = DataLoader(
        TensorDataset(train_lcs, train_radii, train_coeffs),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(DATA_SEED),
    )

    # --------------------------------------------------------
    # 3. Train Direct CNN and SH models
    # --------------------------------------------------------
    models = {}

    direct_model = Improved1DCNN(output_dim=NUM_RAYS).to(device)
    direct_model = train_model(
        direct_model,
        train_loader,
        sh_basis,
        ray_directions,
        mesh_faces,
        name="Direct CNN",
        lambda_phys=0.0,
        is_sh=False,
    )
    models["Direct CNN"] = (direct_model, False)

    val_records = []
    direct_val = evaluate_model_on_synthetic(
        direct_model,
        val_lcs,
        val_radii,
        sh_basis,
        ray_directions,
        mesh_faces,
        False,
    )
    val_records.append({
        "Model": "Direct CNN",
        **direct_val,
    })
    print("Synthetic validation - Direct CNN:", direct_val)

    if SAVE_MODELS:
        torch.save(
            direct_model.state_dict(),
            os.path.join(MODEL_DIR, "direct_cnn.pt"),
        )

    for lam in LAMBDAS:
        name = "SH λ=0 (No Phys)" if lam == 0.0 else f"SH λ={lam}"
        sh_model = Improved1DCNN(output_dim=sh_basis.shape[1]).to(device)
        sh_model = train_model(
            sh_model,
            train_loader,
            sh_basis,
            ray_directions,
            mesh_faces,
            name=name,
            lambda_phys=lam,
            is_sh=True,
        )
        models[name] = (sh_model, True)

        val_result = evaluate_model_on_synthetic(
            sh_model,
            val_lcs,
            val_radii,
            sh_basis,
            ray_directions,
            mesh_faces,
            True,
        )
        val_records.append({
            "Model": name,
            **val_result,
        })
        print(f"Synthetic validation - {name}: {val_result}")

        if SAVE_MODELS:
            safe_name = name.replace(" ", "_").replace("λ", "lambda").replace("=", "")
            torch.save(
                sh_model.state_dict(),
                os.path.join(MODEL_DIR, f"{safe_name}.pt"),
            )

    # --------------------------------------------------------
    # 4. Synthetic validation summary and λ selection
    # --------------------------------------------------------
    selected_lambda, val_df = select_lambda_from_validation(val_records)
    val_df.to_csv("synthetic_validation_results_v2.csv", index=False)
    print("\n=== Synthetic validation results ===")
    print(val_df.to_string(index=False))

    if selected_lambda is not None:
        print(
            f"\nλ selected from synthetic validation: {selected_lambda}"
        )
        print(
            "Note: this is a validation-based selection criterion, "
            "not a universal optimum."
        )
    else:
        print("Could not select λ from validation results.")

    # --------------------------------------------------------
    # 5. Final external DAMIT evaluation
    # --------------------------------------------------------
    convex_urls = [make_damit_url(i) for i in CONVEX_IDS]
    non_convex_urls = [make_damit_url(i) for i in NON_CONVEX_IDS]

    df_results = run_full_eval(
        convex_urls,
        non_convex_urls,
        models,
        ray_directions,
        mesh_faces,
        sh_basis,
    )
    df_results.to_csv("damit_eval_lambda_comparison_v2.csv", index=False)

    print("\n========================================================")
    print("DAMIT 20-object final external evaluation")
    print("========================================================")
    print(summarize_results(df_results).to_string(index=False))

    # --------------------------------------------------------
    # 6. Poster figures
    # --------------------------------------------------------
    plot_lambda_mesh_comparison(
        MESH_FIGURE_ID,
        models,
        ray_directions,
        mesh_faces,
        sh_basis,
    )
    plot_lambda_graphs()

    # --------------------------------------------------------
    # 7. Optional real Ceres demonstration
    # --------------------------------------------------------
    ceres = preprocess_ceres_lc_aligned(
        CERES_LC_PATH,
        device,
        value_kind="magnitude",
        use_first_half=True,
    )
    if ceres is not None:
        ceres_tensor, ceres_lc_plot = ceres

        # Keep the previous qualitative comparison with λ=0.1.
        # If λ=0.1 is not present in LAMBDAS, this will fail loudly rather
        # than silently using a different model.
        if "SH λ=0.1" not in models:
            raise KeyError("SH λ=0.1 model is required for the Ceres figure.")

        plot_ceres_comparison(
            ceres_tensor,
            ceres_lc_plot,
            models["Direct CNN"][0],
            models["SH λ=0.1"][0],
            ray_directions,
            mesh_faces,
            sh_basis,
        )


if __name__ == "__main__":
    main()
