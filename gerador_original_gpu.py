"""Gerador 100% original em GPU (PyTorch ROCm RX 9070 XT) + Áudio ASMR Físico Sincronizado."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from scipy import signal

BASE_DIR = Path(__file__).resolve().parent
CRIADOS_DIR = BASE_DIR / "criados"
REF_AUDIO_DIR = BASE_DIR / "referencias" / "audios_asmr"

RENDER_W = 540
RENDER_H = 960
OUT_W = 1080
OUT_H = 1920
FPS = 60
DURATION_SEC = 10.0
TOTAL_FRAMES = int(FPS * DURATION_SEC)
AUDIO_SR = 44100


@dataclass(frozen=True)
class SceneSpec:
    """Especificação de uma cena 3D original e seu vínculo com o banco."""

    key: str
    ref_shortcode: str
    output_filename: str
    titulo_pt: str
    titulo_en: str
    variacao_tema: str
    hashtags: str
    audio_mode: str


SCENES: list[SceneSpec] = [
    SceneSpec(
        key="cristal",
        ref_shortcode="FDa1Oa2y0NY",
        output_filename="criado_01_prensa_cristal_esmeralda.mp4",
        titulo_pt="Prensa Hexagonal vs Cristal Esmeralda e Ouro 24K!",
        titulo_en="Hexagonal Press vs Emerald Crystal & 24K Gold!",
        variacao_tema=(
            "Octaedro chanfrado de cristal esmeralda translúcido com veios de ouro líquido 24K "
            "esmagado em câmara baixa diagonal 35° por pistão hexagonal sobre bigorna de titânio"
        ),
        hashtags="#ASMR #OddlySatisfying #PrensaHidraulica #CristalEsmeralda #3DASMR #Satisfying",
        audio_mode="crystal_press",
    ),
    SceneSpec(
        key="capsula",
        ref_shortcode="DfNRXxjou-k",
        output_filename="criado_02_corte_capsula_iridescente.mp4",
        titulo_pt="Lâmina Damasco Fatiando Cápsula Iridescente 24 Camadas!",
        titulo_en="Damascus Blade Slicing 24-Layer Iridescent Capsule!",
        variacao_tema=(
            "Cápsula toroidal de polímero iridescente (24 camadas em ametista, cobre, safira e magenta) "
            "fatiada por lâmina de aço damasco em ângulo isométrico 45° sobre bancada reflexiva"
        ),
        hashtags="#ASMR #OddlySatisfying #Slicing #DamascusSteel #PolimeroIridescente #Satisfying",
        audio_mode="iridescent_slice",
    ),
    SceneSpec(
        key="torno",
        ref_shortcode="Df6LIe7uxxw",
        output_filename="criado_03_torno_helicoidal_damasco.mp4",
        titulo_pt="Torno Helicoidal: Crosta Vulcânica para Safira Espelhada!",
        titulo_en="Helical Lathe: Volcanic Crust to Mirror Sapphire Chrome!",
        variacao_tema=(
            "Eixo helicoidal com crosta vulcânica escura usinado em tracking shot oblíquo a 60° "
            "por pastilha dourada com linha incandescente rubi revelando cromo azul-safira"
        ),
        hashtags="#ASMR #OddlySatisfying #MetalLathe #MachiningASMR #AcoDamasco #Satisfying",
        audio_mode="helical_lathe",
    ),
    SceneSpec(
        key="rodo",
        ref_shortcode="uhQXBMZ-sTg",
        output_filename="criado_04_rodo_espuma_prismatica.mp4",
        titulo_pt="Rodo de Titânio Limpando Espuma Neon em Ouro Hexagonal!",
        titulo_en="Titanium Squeegee Clearing Neon Foam on Hex Gold!",
        variacao_tema=(
            "Superfície arquitetônica hexagonal em ouro escovado e grafite coberta por espuma "
            "bioluminescente turquesa/magenta/violeta raspada por rodo duplo de titânio em ângulo de 50°"
        ),
        hashtags="#ASMR #OddlySatisfying #FoamCleaning #SatisfyingASMR #OuroHexagonal #Relaxing",
        audio_mode="prismatic_squeegee",
    ),
]


def smin(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    """Smooth minimum polinomial para fusão orgânica de SDFs."""
    h = torch.clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return torch.lerp(b, a, h) - k * h * (1.0 - h)


def smax(a: torch.Tensor, b: torch.Tensor, k: float) -> torch.Tensor:
    """Smooth maximum polinomial."""
    return -smin(-a, -b, k)


def rot2d_xy(x: torch.Tensor, y: torch.Tensor, angle: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotaciona coordenadas 2D por um ângulo em radianos (zero alocação de tensor)."""
    c = math.cos(angle)
    s = math.sin(angle)
    return c * x - s * y, s * x + c * y


def sd_box_xyz(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, bx: float, by: float, bz: float, r: float = 0.0
) -> torch.Tensor:
    """SDF de caixa arredondada sem alocar tensores intermediários."""
    qx = torch.abs(px) - bx
    qy = torch.abs(py) - by
    qz = torch.abs(pz) - bz
    outside = torch.sqrt(
        torch.clamp(qx, min=0.0) ** 2
        + torch.clamp(qy, min=0.0) ** 2
        + torch.clamp(qz, min=0.0) ** 2
        + 1e-12
    )
    inside = torch.clamp(torch.maximum(qx, torch.maximum(qy, qz)), max=0.0)
    return outside + inside - r


def sd_cylinder_y_xyz(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, radius: float, half_h: float, r_bevel: float = 0.0
) -> torch.Tensor:
    """SDF de cilindro vertical alinhado em Y com chanfro."""
    dx = torch.sqrt(px * px + pz * pz + 1e-12) - (radius - r_bevel)
    dy = torch.abs(py) - (half_h - r_bevel)
    out = torch.sqrt(torch.clamp(dx, min=0.0) ** 2 + torch.clamp(dy, min=0.0) ** 2 + 1e-12)
    ins = torch.clamp(torch.maximum(dx, dy), max=0.0)
    return out + ins - r_bevel


def sd_cylinder_x_xyz(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, radius: float, half_len: float, r_bevel: float = 0.0
) -> torch.Tensor:
    """SDF de cilindro horizontal alinhado em X."""
    dyz = torch.sqrt(py * py + pz * pz + 1e-12) - (radius - r_bevel)
    dx = torch.abs(px) - (half_len - r_bevel)
    out = torch.sqrt(torch.clamp(dyz, min=0.0) ** 2 + torch.clamp(dx, min=0.0) ** 2 + 1e-12)
    ins = torch.clamp(torch.maximum(dyz, dx), max=0.0)
    return out + ins - r_bevel


def sd_hex_prism_y_xyz(
    px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, radius: float, half_h: float, bevel: float = 0.02
) -> torch.Tensor:
    """SDF de prisma hexagonal vertical ao longo do eixo Y."""
    ax = torch.abs(px)
    az = torch.abs(pz)
    kx = -0.866025404
    kz = 0.5
    kz2 = 0.577350269
    dot_k = torch.clamp(kx * ax + kz * az, max=0.0)
    ax = ax - 2.0 * dot_k * kx
    az = az - 2.0 * dot_k * kz
    d_xz = torch.sqrt(
        (ax - torch.clamp(ax, -kz2 * radius, kz2 * radius)) ** 2 + (az - radius) ** 2 + 1e-12
    ) * torch.sign(az - radius)
    d_y = torch.abs(py) - half_h
    out = torch.sqrt(torch.clamp(d_xz, min=0.0) ** 2 + torch.clamp(d_y, min=0.0) ** 2 + 1e-12)
    ins = torch.clamp(torch.maximum(d_xz, d_y), max=0.0)
    return out + ins - bevel


def sd_octahedron_xyz(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, s: float) -> torch.Tensor:
    """SDF exato de octaedro regular (com limite L1 contínuo)."""
    apx = torch.abs(px)
    apy = torch.abs(py)
    apz = torch.abs(pz)
    m = apx + apy + apz - s
    return m * 0.57735027


def sd_torus_z_xyz(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, major_r: float, minor_r: float) -> torch.Tensor:
    """SDF de toroide com eixo ao longo de Z."""
    qx = torch.sqrt(px * px + py * py + 1e-12) - major_r
    return torch.sqrt(qx * qx + pz * pz + 1e-12) - minor_r


def hex_grid_metric(px: torch.Tensor, pz: torch.Tensor, scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Retorna (distância suavizada à borda do hexágono, id da célula)."""
    ux = px * scale
    uz = pz * scale
    r0 = 1.0
    r1 = 1.7320508
    ax = torch.remainder(ux, r0) - 0.5 * r0
    ay = torch.remainder(uz, r1) - 0.5 * r1
    bx = torch.remainder(ux - 0.5 * r0, r0) - 0.5 * r0
    by = torch.remainder(uz - 0.5 * r1, r1) - 0.5 * r1

    use_a = (ax * ax + ay * ay) < (bx * bx + by * by)
    gx = torch.where(use_a, ax, bx)
    gy = torch.where(use_a, ay, by)
    cx = torch.round((ux - gx) * 10.0) * 0.1
    cy = torch.round((uz - gy) * 10.0) * 0.1

    agx = torch.abs(gx)
    agy = torch.abs(gy)
    edge_dist = torch.maximum(0.5 * agx + 0.8660254 * agy, agx)
    cell_hash = torch.frac(torch.sin(cx * 12.9898 + cy * 78.233) * 43758.5453)
    return 0.5 - edge_dist, cell_hash


def make_ray_grid(h: int, w: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Cria grade normalizada de coordenadas de tela (-W/H..W/H, +1..-1)."""
    y, x = torch.meshgrid(
        torch.linspace(1.0, -1.0, h, device=device, dtype=torch.float32),
        torch.linspace(-w / h, w / h, w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return x, y


def build_camera_rays(
    ro_xyz: tuple[float, float, float],
    ta_xyz: tuple[float, float, float],
    fov: float,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Calcula origem e direções de raios normalizados em GPU."""
    ro = torch.tensor(ro_xyz, device=device, dtype=torch.float32)
    ta = torch.tensor(ta_xyz, device=device, dtype=torch.float32)
    ww = ta - ro
    ww = ww / torch.linalg.norm(ww)
    up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=torch.float32)
    uu = torch.cross(ww, up, dim=0)
    uu = uu / torch.linalg.norm(uu)
    vv = torch.cross(uu, ww, dim=0)

    rd = (
        x[..., None] * uu[None, None, :]
        + y[..., None] * vv[None, None, :]
        + fov * ww[None, None, :]
    )
    rd = rd / torch.linalg.norm(rd, dim=-1, keepdim=True)
    return ro, rd


def compute_normal_tetra(
    p: torch.Tensor,
    sdf_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    eps: float = 0.0022,
) -> torch.Tensor:
    """Calcula normal da superfície com 4 avaliações (método do tetraedro)."""
    px, py, pz = p[..., 0], p[..., 1], p[..., 2]
    d0 = sdf_fn(px + eps, py - eps, pz - eps)
    d1 = sdf_fn(px - eps, py - eps, pz + eps)
    d2 = sdf_fn(px - eps, py + eps, pz - eps)
    d3 = sdf_fn(px + eps, py + eps, pz + eps)
    nx = d0 - d1 - d2 + d3
    ny = -d0 - d1 + d2 + d3
    nz = -d0 + d1 - d2 + d3
    n = torch.stack([nx, ny, nz], dim=-1)
    return n / (torch.linalg.norm(n, dim=-1, keepdim=True) + 1e-8)


def soft_shadow(
    p: torch.Tensor,
    n: torch.Tensor,
    light_dir: torch.Tensor,
    sdf_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    active: torch.Tensor,
    steps: int = 8,
) -> torch.Tensor:
    """Calcula sombra suave aproximada apenas para pixels ativos."""
    res = torch.ones(p.shape[:-1], device=p.device, dtype=torch.float32)
    t = torch.full(p.shape[:-1], 0.03, device=p.device, dtype=torch.float32)
    ox = p[..., 0] + n[..., 0] * 0.008
    oy = p[..., 1] + n[..., 1] * 0.008
    oz = p[..., 2] + n[..., 2] * 0.008
    lx, ly, lz = float(light_dir[0]), float(light_dir[1]), float(light_dir[2])
    for _ in range(steps):
        d = sdf_fn(ox + lx * t, oy + ly * t, oz + lz * t)
        res = torch.where(active, torch.minimum(res, 11.0 * d / (t + 1e-4)), res)
        t = t + torch.clamp(d, 0.02, 0.25)
    return torch.clamp(res, 0.14, 1.0)


def studio_pbr_shade(
    p: torch.Tensor,
    n: torch.Tensor,
    rd: torch.Tensor,
    hit: torch.Tensor,
    albedo: torch.Tensor,
    roughness: torch.Tensor,
    metallic: torch.Tensor,
    emissive: torch.Tensor,
    sdf_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    bg_color: torch.Tensor,
    key_dir_xyz: tuple[float, float, float] = (0.55, 0.78, 0.45),
    rim_color_rgb: tuple[float, float, float] = (0.25, 0.85, 1.0),
) -> torch.Tensor:
    """Iluminação PBR de estúdio (3 luzes + Fresnel + sombra suave + AO + ACES Filmic)."""
    dev = p.device
    key_dir = torch.tensor(key_dir_xyz, device=dev, dtype=torch.float32)
    key_dir = key_dir / torch.linalg.norm(key_dir)

    fill_dir = torch.tensor([-0.75, 0.35, 0.55], device=dev, dtype=torch.float32)
    fill_dir = fill_dir / torch.linalg.norm(fill_dir)

    rim_dir = torch.tensor([-0.25, 0.55, -0.82], device=dev, dtype=torch.float32)
    rim_dir = rim_dir / torch.linalg.norm(rim_dir)

    view_dir = -rd
    n_dot_v = torch.clamp((n * view_dir).sum(dim=-1, keepdim=True), 0.001, 1.0)
    refl = rd - 2.0 * (rd * n).sum(dim=-1, keepdim=True) * n

    sh = soft_shadow(p, n, key_dir, sdf_fn, hit, steps=8)[..., None]
    ao1 = sdf_fn(p[..., 0] + n[..., 0] * 0.04, p[..., 1] + n[..., 1] * 0.04, p[..., 2] + n[..., 2] * 0.04)
    ao2 = sdf_fn(p[..., 0] + n[..., 0] * 0.11, p[..., 1] + n[..., 1] * 0.11, p[..., 2] + n[..., 2] * 0.11)
    ao = torch.clamp(
        1.0 - 2.0 * torch.clamp(0.04 - ao1, min=0.0) - 1.1 * torch.clamp(0.11 - ao2, min=0.0),
        0.25,
        1.0,
    )[..., None]

    n_dot_l = torch.clamp((n * key_dir).sum(dim=-1, keepdim=True), 0.0, 1.0)
    n_dot_fill = torch.clamp((n * fill_dir).sum(dim=-1, keepdim=True), 0.0, 1.0)
    n_dot_rim = torch.clamp((n * rim_dir).sum(dim=-1, keepdim=True), 0.0, 1.0)

    diffuse_weight = (1.0 - metallic) * albedo
    lit = diffuse_weight * (
        n_dot_l * sh * 1.42
        + n_dot_fill * 0.34
        + 0.14 * (0.5 + 0.5 * n[..., 1:2])
    )

    half_vec = key_dir + view_dir
    half_vec = half_vec / (torch.linalg.norm(half_vec, dim=-1, keepdim=True) + 1e-8)
    n_dot_h = torch.clamp((n * half_vec).sum(dim=-1, keepdim=True), 0.0, 1.0)
    spec_power = 180.0 + (14.0 - 180.0) * roughness
    spec_lobe = torch.pow(n_dot_h, spec_power) * (spec_power * 0.045 + 0.5)

    sb1 = torch.exp(-((refl[..., 0:1] - 0.35) ** 2 + (refl[..., 1:2] - 0.65) ** 2) * 8.0)
    sb2 = torch.exp(-((refl[..., 0:1] + 0.55) ** 2 + (refl[..., 1:2] - 0.45) ** 2) * 11.0)
    horizon_band = torch.exp(-torch.abs(refl[..., 1:2] - 0.08) * 10.0) * 0.38
    env_refl = (sb1 * 1.35 + sb2 * 0.95 + horizon_band + 0.06) * (1.0 - 0.65 * roughness)

    fresnel = torch.pow(1.0 - n_dot_v, 4.0)
    f0 = torch.lerp(torch.full_like(albedo, 0.06), albedo, metallic)
    spec_color = f0 + (1.0 - f0) * fresnel

    rim_col = torch.tensor(rim_color_rgb, device=dev, dtype=torch.float32)
    rim_light = torch.pow(1.0 - n_dot_v, 3.0) * (0.32 + 0.85 * n_dot_rim) * rim_col

    color = (lit + spec_color * (spec_lobe * sh + env_refl)) * ao + rim_light + emissive
    return torch.where(hit[..., None], color, bg_color)


def aces_tonemap_to_u8(hdr: torch.Tensor) -> np.ndarray:
    """Mapeamento ACES Filmic + correção gamma sRGB -> uint8 RGB para FFmpeg."""
    x = torch.clamp(hdr, min=0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    mapped = torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    srgb = torch.pow(mapped, 1.0 / 2.2)
    return (torch.clamp(srgb * 255.0, 0.0, 255.0)).to(torch.uint8).cpu().numpy()


def raymarch_scene(
    ro: torch.Tensor,
    rd: torch.Tensor,
    sdf_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    steps: int = 54,
    step_scale: float = 0.65,
    max_t: float = 11.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raymarching com passo conservador e ajuste final na superfície."""
    t = torch.zeros(rd.shape[:-1], device=rd.device, dtype=torch.float32)
    rox, roy, roz = float(ro[0]), float(ro[1]), float(ro[2])
    rdx, rdy, rdz = rd[..., 0], rd[..., 1], rd[..., 2]
    for _ in range(steps):
        d = sdf_fn(rox + rdx * t, roy + rdy * t, roz + rdz * t)
        t = torch.clamp(t + d * step_scale, 0.0, max_t)
    # Passo final exato para assentar os raios na superfície sem flutuação
    d_snap = sdf_fn(rox + rdx * t, roy + rdy * t, roz + rdz * t)
    t = torch.clamp(t + d_snap, 0.0, max_t)
    px = rox + rdx * t
    py = roy + rdy * t
    pz = roz + rdz * t
    p_hit = torch.stack([px, py, pz], dim=-1)
    d_final = sdf_fn(px, py, pz)
    hit = (t < max_t - 0.08) & (d_final < 0.04)
    return t, p_hit, hit


# ---------------------------------------------------------------------------
# CENA 1: Prensa Hexagonal vs Octaedro de Cristal Esmeralda e Ouro 24K
# ---------------------------------------------------------------------------
def render_scene_cristal(
    frame_idx: int, total_frames: int, x: torch.Tensor, y: torch.Tensor, device: torch.device
) -> np.ndarray:
    u = frame_idx / max(1, total_frames - 1)
    cam_ang = -0.62 + 1.22 * u
    cam_rad = 3.10 - 0.55 * math.sin(math.pi * min(u * 1.15, 1.0))
    cam_y = -0.06 + 0.40 * u
    ro, rd = build_camera_rays(
        (math.sin(cam_ang) * cam_rad, cam_y, math.cos(cam_ang) * cam_rad),
        (0.0, -0.08, 0.0),
        2.15,
        x,
        y,
        device,
    )

    if u < 0.56:
        piston_y = 1.15 - 1.24 * (u / 0.56) ** 1.1
    elif u < 0.72:
        piston_y = -0.09 - 0.14 * ((u - 0.56) / 0.16)
    else:
        piston_y = -0.23 + 0.92 * ((u - 0.72) / 0.28) ** 1.4

    comp = max(0.0, min(1.0, (0.42 - piston_y) / 0.65))
    shatter = max(0.0, min(1.0, (u - 0.52) / 0.48))

    squash_y = max(0.32, 1.0 - 0.68 * comp)
    stretch_xz = 1.0 + 0.52 * comp
    cryst_cy = -0.55 + 0.42 * squash_y
    cryst_rot = 0.42 + shatter * 0.14
    scale_min = min(squash_y, stretch_xz)

    frag_centers: list[tuple[float, float, float, float]] = []
    if shatter > 0.0:
        for i in range(14):
            ang = i * 2.39996 + 0.4
            spd = 0.75 + (i % 4) * 0.28
            dt = u - 0.52
            r_xy = 0.28 + spd * dt * 2.4
            vy0 = 0.55 + (i % 3) * 0.32
            fy = max(-0.50, -0.18 + vy0 * dt - 2.6 * dt * dt)
            frag_centers.append((math.cos(ang) * r_xy, fy, math.sin(ang) * r_xy, 0.055 + (i % 3) * 0.016))

    def crystal_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        cy_rel = (py - cryst_cy) / squash_y
        rx, rz = rot2d_xy(px, pz, cryst_rot)
        rx = rx / stretch_xz
        rz = rz / stretch_xz
        d_oct = sd_octahedron_xyz(rx, cy_rel, rz, 0.60) * scale_min - 0.015
        d_box = sd_box_xyz(rx, cy_rel, rz, 0.35, 0.35, 0.35, 0.015) * scale_min
        d_main = torch.maximum(d_oct, d_box - 0.025)
        if shatter > 0.0:
            crack = torch.sin(px * 15.0) * torch.cos(pz * 15.0) * 0.016 * min(1.0, shatter * 3.5)
            d_main = d_main + crack
            for fx, fy, fz, fs in frag_centers:
                df = sd_octahedron_xyz(px - fx, py - fy, pz - fz, fs)
                d_main = torch.minimum(d_main, df)
        return d_main

    def piston_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        d_hex = sd_hex_prism_y_xyz(px, py - (piston_y + 0.72), pz, 0.48, 0.72, 0.022)
        d_ring = sd_cylinder_y_xyz(px, py - (piston_y + 0.08), pz, 0.52, 0.045, 0.012)
        return torch.minimum(d_hex, d_ring)

    def anvil_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        d_cyl = sd_cylinder_y_xyz(px, py + 0.82, pz, 0.92, 0.27, 0.025)
        r_xz = torch.sqrt(px * px + pz * pz + 1e-12)
        groove = torch.clamp(torch.sin(r_xz * 36.0) * 0.003, -0.003, 0.0) * (py > -0.58)
        return torch.minimum(d_cyl - groove, py + 1.12)

    def total_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        return torch.minimum(crystal_sdf(px, py, pz), torch.minimum(piston_sdf(px, py, pz), anvil_sdf(px, py, pz)))

    _, p_hit, hit = raymarch_scene(ro, rd, total_sdf, steps=54, step_scale=0.65, max_t=10.5)
    n = compute_normal_tetra(p_hit, total_sdf)

    px, py, pz = p_hit[..., 0], p_hit[..., 1], p_hit[..., 2]
    dc = crystal_sdf(px, py, pz)
    dp = piston_sdf(px, py, pz)
    da = anvil_sdf(px, py, pz)

    is_crystal = (dc <= dp) & (dc <= da)
    is_piston = (dp < dc) & (dp <= da)

    veins = torch.abs(torch.sin(px * 11.0 + torch.sin(py * 9.0) * 1.4 + pz * 8.0))
    gold_vein = torch.exp(-veins * 15.0)
    stress_glow = comp * torch.exp(-torch.abs(torch.sin(px * 19.0 - pz * 16.0)) * 13.0)

    c_emerald = torch.tensor([0.02, 0.68, 0.42], device=device)
    c_deep_em = torch.tensor([0.01, 0.22, 0.16], device=device)
    c_gold = torch.tensor([1.0, 0.78, 0.22], device=device)
    c_cyan = torch.tensor([0.08, 0.96, 1.0], device=device)

    crystal_alb = torch.lerp(c_deep_em, c_emerald, 0.40 + 0.60 * torch.clamp(n[..., 1:2], 0.0, 1.0))
    crystal_alb = torch.lerp(crystal_alb, c_gold, gold_vein[..., None] * 0.88)

    is_gold_ring = (py < piston_y + 0.13) & (py > piston_y + 0.03)
    piston_alb = torch.where(
        is_gold_ring[..., None],
        torch.tensor([0.98, 0.76, 0.24], device=device),
        torch.tensor([0.58, 0.63, 0.70], device=device),
    )
    anvil_alb = torch.tensor([0.09, 0.11, 0.14], device=device)

    albedo = torch.where(is_crystal[..., None], crystal_alb, torch.where(is_piston[..., None], piston_alb, anvil_alb))
    roughness = torch.where(is_crystal[..., None], 0.06, torch.where(is_piston[..., None], 0.20, 0.30))
    metallic = torch.where(is_crystal[..., None], gold_vein[..., None] * 0.85, torch.where(is_piston[..., None], 0.94, 0.80))

    emissive = torch.where(
        is_crystal[..., None],
        c_gold * (gold_vein[..., None] * (0.28 + 0.8 * comp))
        + c_cyan * (stress_glow[..., None] * (0.35 + 1.4 * shatter)),
        torch.zeros_like(albedo),
    )

    bg = torch.lerp(
        torch.tensor([0.015, 0.025, 0.04], device=device),
        torch.tensor([0.04, 0.11, 0.14], device=device),
        torch.clamp(0.5 + 0.5 * rd[..., 1:2], 0.0, 1.0),
    )

    hdr = studio_pbr_shade(
        p_hit, n, rd, hit, albedo, roughness, metallic, emissive, total_sdf, bg,
        key_dir_xyz=(0.6, 0.78, 0.42), rim_color_rgb=(0.15, 0.95, 0.75)
    )
    return aces_tonemap_to_u8(hdr)


# ---------------------------------------------------------------------------
# CENA 2: Cápsula Toroidal Iridescente 24 Camadas + Lâmina Aço Damasco
# ---------------------------------------------------------------------------
def render_scene_capsula(
    frame_idx: int, total_frames: int, x: torch.Tensor, y: torch.Tensor, device: torch.device
) -> np.ndarray:
    u = frame_idx / max(1, total_frames - 1)
    cam_x = -1.65 + 3.10 * u
    ro, rd = build_camera_rays((cam_x, 2.05, 2.25), (0.0, -0.05, 0.0), 2.25, x, y, device)

    cut_prog = max(0.0, min(1.0, (u - 0.08) / 0.50))
    blade_y = 1.45 - 1.98 * (cut_prog ** 0.95)
    split = max(0.0, min(1.0, (u - 0.50) / 0.50))
    split_smooth = split * split * (3.0 - 2.0 * split)
    sep_x = 0.010 + 0.62 * split_smooth
    tilt_ang = 0.30 * split_smooth

    def capsule_half_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor, side: float) -> torch.Tensor:
        hx = px - side * sep_x
        hy = py + 0.02
        rx, ry = rot2d_xy(hx, hy, -side * tilt_ang)
        pry, prz = rot2d_xy(ry, pz, 0.35)
        d_torus = sd_torus_z_xyz(rx, pry, prz, 0.46, 0.30)
        d_core = sd_cylinder_y_xyz(rx, pry, prz, 0.58, 0.30, 0.08)
        d_shape = smin(d_torus, d_core, 0.08)
        d_cut_plane = side * (-rx) - 0.002
        gap_cut = torch.where((py > blade_y - 0.02) & (cut_prog > 0.01), 0.025 * cut_prog, 0.0)
        return torch.maximum(d_shape, d_cut_plane + gap_cut)

    def blade_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        by = py - (blade_y + 0.55)
        thickness = torch.clamp(0.004 + (by + 0.55) * 0.042, 0.004, 0.032)
        dx = torch.abs(px) - thickness
        dy = torch.abs(by) - 0.55
        dz = torch.abs(pz) - 1.15
        out = torch.sqrt(
            torch.clamp(dx, min=0.0) ** 2 + torch.clamp(dy, min=0.0) ** 2 + torch.clamp(dz, min=0.0) ** 2 + 1e-12
        )
        ins = torch.clamp(torch.maximum(dx, torch.maximum(dy, dz)), max=0.0)
        return out + ins

    def total_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        d_left = capsule_half_sdf(px, py, pz, -1.0)
        d_right = capsule_half_sdf(px, py, pz, 1.0)
        return torch.minimum(torch.minimum(d_left, d_right), torch.minimum(blade_sdf(px, py, pz), py + 0.62))

    _, p_hit, hit = raymarch_scene(ro, rd, total_sdf, steps=54, step_scale=0.60, max_t=10.5)
    n = compute_normal_tetra(p_hit, total_sdf, eps=0.0018)

    px, py, pz = p_hit[..., 0], p_hit[..., 1], p_hit[..., 2]
    d_left = capsule_half_sdf(px, py, pz, -1.0)
    d_right = capsule_half_sdf(px, py, pz, 1.0)
    d_cap = torch.minimum(d_left, d_right)
    d_blade = blade_sdf(px, py, pz)
    d_tbl = py + 0.62

    is_cap = (d_cap <= d_blade) & (d_cap <= d_tbl)
    is_blade = (d_blade < d_cap) & (d_blade <= d_tbl)

    side_sign = torch.where(d_left < d_right, -1.0, 1.0)
    p_local_x = px - side_sign * sep_x
    r_layer = torch.sqrt(p_local_x * p_local_x + py * py + pz * pz + 1e-12)
    ring_phase = r_layer * 32.0 - split_smooth * 1.4
    ring_idx = torch.floor(ring_phase)
    ring_mod = torch.remainder(ring_idx, 4.0)

    c_amethyst = torch.tensor([0.58, 0.18, 0.92], device=device)
    c_copper = torch.tensor([0.92, 0.44, 0.22], device=device)
    c_sapphire = torch.tensor([0.04, 0.64, 0.98], device=device)
    c_magenta = torch.tensor([0.98, 0.08, 0.56], device=device)

    cap_alb = torch.where(
        (ring_mod < 0.5)[..., None], c_amethyst,
        torch.where((ring_mod < 1.5)[..., None], c_copper,
                    torch.where((ring_mod < 2.5)[..., None], c_sapphire, c_magenta)),
    )
    seam = torch.exp(-torch.abs(torch.frac(ring_phase) - 0.5) * 18.0)[..., None]
    cap_alb = cap_alb * (0.72 + 0.38 * seam)

    damascus = 0.5 + 0.5 * torch.sin(pz * 36.0 + torch.sin(py * 28.0) * 2.2)
    blade_alb = torch.lerp(
        torch.tensor([0.24, 0.27, 0.32], device=device),
        torch.tensor([0.84, 0.87, 0.92], device=device),
        damascus[..., None],
    )

    grid_line = torch.exp(-torch.abs(torch.sin(px * 6.0)) * 22.0) + torch.exp(-torch.abs(torch.sin(pz * 6.0)) * 22.0)
    table_alb = torch.tensor([0.025, 0.03, 0.045], device=device) + grid_line[..., None] * torch.tensor([0.025, 0.035, 0.055], device=device)

    albedo = torch.where(is_cap[..., None], cap_alb, torch.where(is_blade[..., None], blade_alb, table_alb))
    roughness = torch.where(is_cap[..., None], 0.12, torch.where(is_blade[..., None], 0.07, 0.16))
    metallic = torch.where(
        is_cap[..., None],
        torch.where((ring_mod > 0.5) & (ring_mod < 1.5), 0.78, 0.25)[..., None],
        torch.where(is_blade[..., None], 0.96, 0.42),
    )
    emissive = torch.where(is_cap[..., None], cap_alb * 0.08, torch.zeros_like(albedo))

    bg = torch.lerp(
        torch.tensor([0.025, 0.015, 0.05], device=device),
        torch.tensor([0.07, 0.035, 0.12], device=device),
        torch.clamp(0.5 + 0.5 * rd[..., 1:2], 0.0, 1.0),
    )

    hdr = studio_pbr_shade(
        p_hit, n, rd, hit, albedo, roughness, metallic, emissive, total_sdf, bg,
        key_dir_xyz=(0.45, 0.88, 0.35), rim_color_rgb=(0.85, 0.35, 1.0)
    )
    return aces_tonemap_to_u8(hdr)


# ---------------------------------------------------------------------------
# CENA 3: Torno Helicoidal com Crosta Vulcânica -> Cromo Azul-Safira
# ---------------------------------------------------------------------------
def render_scene_torno(
    frame_idx: int, total_frames: int, x: torch.Tensor, y: torch.Tensor, device: torch.device
) -> np.ndarray:
    u = frame_idx / max(1, total_frames - 1)
    cut_x = 0.85 - 1.70 * u
    spin = u * 42.0

    # Câmera lateral oblíqua deslizante (tracking shot a ~60°) mostrando eixo e inserto sem obstrução
    ro, rd = build_camera_rays(
        (cut_x + 1.48, 0.66, 2.25),
        (cut_x - 0.12, 0.04, 0.0),
        2.30,
        x,
        y,
        device,
    )

    # Faíscas e gotas metálicas incandescentes voando da linha de corte
    sparks: list[tuple[float, float, float, float]] = []
    for i in range(10):
        phase = (u * 6.0 + i / 10.0) % 1.0
        ang = phase * 4.0 + i * 1.3
        sx = cut_x - 0.04 + phase * 0.45 * math.cos(ang * 0.4)
        sy = -0.10 + math.sin(ang) * 0.18 - 1.15 * (phase ** 2)
        sz = 0.42 + phase * 0.65 + math.cos(ang) * 0.08
        sparks.append((sx, sy, sz, 0.018 + 0.008 * (1.0 - phase)))

    def shaft_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        # Degrau nítido entre seção bruta (crosta vulcânica) e núcleo usinado espelhado
        step_mask = torch.sigmoid((px - cut_x) * 45.0)
        base_r = 0.40 + 0.070 * step_mask
        theta = torch.atan2(pz, py) + spin
        # Roscas/ranhuras helicoidais afiadas em V no núcleo e crosta facetada na direita
        v_thread = torch.abs(torch.frac(px * 6.0 + theta * 0.477) - 0.5) * 0.028 * (1.0 - step_mask)
        crust_facets = (1.0 - torch.abs(torch.sin(px * 14.0 + theta * 2.0))) * 0.022 * step_mask
        dyz = (torch.sqrt(py * py + pz * pz + 1e-12) - (base_r + v_thread + crust_facets)) * 0.85
        dx = torch.abs(px) - 1.55
        d_bar = torch.maximum(dyz, dx)
        d_flange = sd_cylinder_x_xyz(torch.abs(px) - 1.50, py, pz, 0.66, 0.14, 0.02)
        return torch.minimum(d_bar, d_flange)

    def tool_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        # Pastilha diamante dourada a 45° entrando no degrau de corte + suporte chanfrado
        ix_r, iz_r = rot2d_xy(px - (cut_x + 0.02), pz - 0.47, 0.785)
        d_insert = sd_box_xyz(ix_r, py + 0.06, iz_r, 0.08, 0.028, 0.08, 0.005)
        hx = px - (cut_x + 0.14)
        hy = py + 0.16
        hz = pz - 0.86
        d_holder = sd_box_xyz(hx, hy, hz, 0.095, 0.06, 0.36, 0.015)
        d_out = torch.minimum(d_insert, d_holder)
        for sx, sy, sz, sr in sparks:
            d_sp = torch.sqrt((px - sx) ** 2 + (py - sy) ** 2 + (pz - sz) ** 2 + 1e-12) - sr
            d_out = torch.minimum(d_out, d_sp)
        return d_out

    def total_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        return torch.minimum(shaft_sdf(px, py, pz), tool_sdf(px, py, pz))

    _, p_hit, hit = raymarch_scene(ro, rd, total_sdf, steps=54, step_scale=0.58, max_t=10.5)
    n = compute_normal_tetra(p_hit, total_sdf, eps=0.0014)

    px, py, pz = p_hit[..., 0], p_hit[..., 1], p_hit[..., 2]
    d_sh = shaft_sdf(px, py, pz)
    d_tl = tool_sdf(px, py, pz)
    is_shaft = d_sh <= d_tl

    theta_h = torch.atan2(pz, py) + spin
    machined = torch.sigmoid((cut_x - px) * 65.0)[..., None]
    damascus_raw = 0.5 + 0.5 * torch.sin(theta_h * 6.0 + px * 26.0 + torch.sin(px * 10.0) * 1.8)
    damascus_sharp = torch.sigmoid((damascus_raw - 0.48) * 14.0)

    c_crust = torch.tensor([0.075, 0.065, 0.06], device=device)
    c_sapphire_dark = torch.tensor([0.02, 0.18, 0.52], device=device)
    c_sapphire_bright = torch.tensor([0.24, 0.82, 1.0], device=device)
    sapphire_alb = torch.lerp(c_sapphire_dark, c_sapphire_bright, damascus_sharp[..., None])
    shaft_alb = torch.lerp(c_crust, sapphire_alb, machined)

    is_insert = (pz < 0.57) & (pz > 0.36) & (py > -0.14) & (py < -0.02) & (torch.abs(px - (cut_x + 0.02)) < 0.11)
    is_spark = (py < -0.08) & (pz > 0.40) & (torch.abs(px - (cut_x + 0.14)) > 0.12)
    tool_alb = torch.where(
        is_insert[..., None],
        torch.tensor([1.0, 0.78, 0.22], device=device),
        torch.where(
            is_spark[..., None],
            torch.tensor([1.0, 0.45, 0.15], device=device),
            torch.tensor([0.14, 0.16, 0.20], device=device),
        ),
    )

    albedo = torch.where(is_shaft[..., None], shaft_alb, tool_alb)
    roughness = torch.where(is_shaft[..., None], 0.72 + (0.05 - 0.72) * machined, 0.14)
    metallic = torch.where(is_shaft[..., None], 0.15 + (0.98 - 0.15) * machined, 0.94)

    glow_band = torch.exp(-((px - cut_x) ** 2) * 220.0) * torch.exp(-((py + 0.07) ** 2 + (pz - 0.39) ** 2) * 14.0)
    c_ruby = torch.tensor([1.0, 0.12, 0.28], device=device)
    c_amber = torch.tensor([1.0, 0.58, 0.12], device=device)
    emissive = glow_band[..., None] * (c_ruby * 2.2 + c_amber * 1.1) + torch.where(
        (~is_shaft & is_spark)[..., None],
        torch.tensor([1.4, 0.42, 0.15], device=device),
        torch.zeros_like(albedo),
    )

    bg = torch.lerp(
        torch.tensor([0.015, 0.025, 0.04], device=device),
        torch.tensor([0.045, 0.085, 0.14], device=device),
        torch.clamp(0.5 + 0.5 * rd[..., 1:2], 0.0, 1.0),
    )

    hdr = studio_pbr_shade(
        p_hit, n, rd, hit, albedo, roughness, metallic, emissive, total_sdf, bg,
        key_dir_xyz=(0.45, 0.85, 0.55), rim_color_rgb=(0.25, 0.72, 1.0)
    )
    return aces_tonemap_to_u8(hdr)


# ---------------------------------------------------------------------------
# CENA 4: Rodo Duplo de Titânio + Espuma Prismática sobre Ouro Hexagonal
# ---------------------------------------------------------------------------
def render_scene_rodo(
    frame_idx: int, total_frames: int, x: torch.Tensor, y: torch.Tensor, device: torch.device
) -> np.ndarray:
    u = frame_idx / max(1, total_frames - 1)
    wiper_z = 1.35 - 2.65 * u

    cam_x = -1.15 + 0.55 * math.sin(u * math.pi)
    cam_y = 1.65 - 0.20 * u
    cam_z = wiper_z + 1.55
    ro, rd = build_camera_rays(
        (cam_x, cam_y, cam_z),
        (0.0, 0.02, wiper_z - 0.32),
        2.10,
        x,
        y,
        device,
    )

    def hex_surface_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        edge_dist, _ = hex_grid_metric(px, pz, scale=2.8)
        # Chanfro suave sem salto abrupto para zero artefato de normal
        tile_h = torch.sigmoid((edge_dist - 0.04) * 45.0) * 0.026
        return py - tile_h

    def foam_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        ahead = torch.sigmoid((wiper_z - pz) * 24.0)
        wave_roll = torch.exp(-((pz - (wiper_z - 0.14)) ** 2) * 28.0) * 0.24
        bubbles = (
            torch.sin(px * 9.0) * torch.cos(pz * 9.0) * 0.042
            + torch.sin(px * 19.0 + pz * 14.0) * 0.018
        )
        foam_top = (0.12 + wave_roll + bubbles) * ahead - 0.03 * (1.0 - ahead)
        return torch.maximum((py - foam_top) * 0.72, pz - wiper_z)

    def squeegee_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        d_blade1 = sd_box_xyz(px, py - 0.18, pz - (wiper_z + 0.03), 1.15, 0.17, 0.018, 0.008)
        d_blade2 = sd_box_xyz(px, py - 0.20, pz - (wiper_z + 0.11), 1.15, 0.16, 0.018, 0.008)
        d_head = sd_box_xyz(px, py - 0.34, pz - (wiper_z + 0.07), 1.16, 0.045, 0.075, 0.02)
        hy, hz = rot2d_xy(py - 0.56, pz - (wiper_z + 0.30), -0.68)
        d_handle = sd_cylinder_y_xyz(px, hy, hz, 0.042, 0.42, 0.012)
        return torch.minimum(torch.minimum(d_blade1, d_blade2), torch.minimum(d_head, d_handle))

    def total_sdf(px: torch.Tensor, py: torch.Tensor, pz: torch.Tensor) -> torch.Tensor:
        return torch.minimum(hex_surface_sdf(px, py, pz), torch.minimum(foam_sdf(px, py, pz), squeegee_sdf(px, py, pz)))

    _, p_hit, hit = raymarch_scene(ro, rd, total_sdf, steps=54, step_scale=0.58, max_t=11.0)
    n = compute_normal_tetra(p_hit, total_sdf, eps=0.0020)

    px, py, pz = p_hit[..., 0], p_hit[..., 1], p_hit[..., 2]
    d_hex = hex_surface_sdf(px, py, pz)
    d_fm = foam_sdf(px, py, pz)
    d_sq = squeegee_sdf(px, py, pz)

    is_foam = (d_fm <= d_hex) & (d_fm <= d_sq)
    is_sq = (d_sq < d_hex) & (d_sq < d_fm)

    edge_dist, cell_hash = hex_grid_metric(px, pz, scale=2.8)
    is_gold_tile = (edge_dist > 0.035) & (cell_hash > 0.34)
    hex_alb = torch.where(
        is_gold_tile[..., None],
        torch.tensor([0.92, 0.72, 0.24], device=device),
        torch.where(
            (edge_dist > 0.035)[..., None],
            torch.tensor([0.12, 0.14, 0.18], device=device),
            torch.tensor([0.025, 0.03, 0.04], device=device),
        ),
    )

    swirl = 0.5 + 0.5 * torch.sin(px * 2.6 + pz * 3.1 + torch.sin(px * 5.0) * 0.9)
    c_turq = torch.tensor([0.04, 0.95, 0.84], device=device)
    c_mag = torch.tensor([1.0, 0.16, 0.58], device=device)
    c_viol = torch.tensor([0.55, 0.28, 1.0], device=device)
    foam_alb = torch.where(
        (swirl < 0.5)[..., None],
        torch.lerp(c_turq, c_viol, (swirl * 2.0)[..., None]),
        torch.lerp(c_viol, c_mag, ((swirl - 0.5) * 2.0)[..., None]),
    )
    micro_bub = 0.82 + 0.18 * torch.sin(px * 38.0) * torch.cos(pz * 38.0)
    foam_alb = foam_alb * micro_bub[..., None]

    sq_alb = torch.tensor([0.78, 0.82, 0.88], device=device)

    albedo = torch.where(is_foam[..., None], foam_alb, torch.where(is_sq[..., None], sq_alb, hex_alb))
    roughness = torch.where(is_foam[..., None], 0.42, torch.where(is_sq[..., None], 0.07, 0.12))
    metallic = torch.where(is_foam[..., None], 0.02, torch.where(is_sq[..., None], 0.96, 0.90))
    emissive = torch.where(is_foam[..., None], foam_alb * 0.06, torch.zeros_like(albedo))

    bg = torch.lerp(
        torch.tensor([0.02, 0.03, 0.05], device=device),
        torch.tensor([0.07, 0.04, 0.12], device=device),
        torch.clamp(0.5 + 0.5 * rd[..., 1:2], 0.0, 1.0),
    )

    hdr = studio_pbr_shade(
        p_hit, n, rd, hit, albedo, roughness, metallic, emissive, total_sdf, bg,
        key_dir_xyz=(0.50, 0.82, 0.38), rim_color_rgb=(0.20, 0.92, 0.88)
    )
    return aces_tonemap_to_u8(hdr)


RENDERERS: dict[str, Callable[[int, int, torch.Tensor, torch.Tensor, torch.device], np.ndarray]] = {
    "cristal": render_scene_cristal,
    "capsula": render_scene_capsula,
    "torno": render_scene_torno,
    "rodo": render_scene_rodo,
}


def _extract_highband_texture(ref_shortcode: str, target_len: int, rng: np.random.Generator) -> np.ndarray:
    """Extrai apenas micro-transientes de alta frequência (>2.6kHz) embaralhados sem voz/melodia."""
    wav_path = REF_AUDIO_DIR / f"{ref_shortcode}_raw.wav"
    if not wav_path.exists():
        return np.zeros(target_len, dtype=np.float32)
    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(min(w.getnframes(), sr * 12)), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    if len(raw) < sr:
        return np.zeros(target_len, dtype=np.float32)
    sos = signal.butter(4, 2600.0, btype="highpass", fs=sr, output="sos")
    hf = signal.sosfilt(sos, raw).astype(np.float32)
    grain = int(sr * 0.025)
    n_grains = len(hf) // grain
    if n_grains < 4:
        return np.zeros(target_len, dtype=np.float32)
    blocks = hf[: n_grains * grain].reshape(n_grains, grain)
    win = np.hanning(grain).astype(np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    hop = grain // 2
    for pos in range(0, target_len - grain, hop):
        idx = int(rng.integers(0, n_grains))
        out[pos : pos + grain] += blocks[idx] * win
    peak = float(np.max(np.abs(out))) + 1e-6
    return out / peak


def synthesize_asmr_audio(spec: SceneSpec, wav_out: Path) -> None:
    """Sintetiza trilha ASMR estéreo sincronizada frame a frame com a física 3D."""
    n_samples = int(DURATION_SEC * AUDIO_SR)
    t = np.linspace(0.0, DURATION_SEC, n_samples, endpoint=False, dtype=np.float32)
    u = t / DURATION_SEC
    rng = np.random.default_rng(abs(hash(spec.key)) % (2**32))

    white = rng.standard_normal(n_samples).astype(np.float32)
    tex = _extract_highband_texture(spec.ref_shortcode, n_samples, rng)

    left = np.zeros(n_samples, dtype=np.float32)
    right = np.zeros(n_samples, dtype=np.float32)

    if spec.audio_mode == "crystal_press":
        servo_env = np.where((u < 0.72) | (u > 0.72), 0.14 * np.exp(-((u - 0.35) ** 2) * 3.0), 0.05)
        servo_hum = np.sin(2.0 * np.pi * (58.0 + 18.0 * u) * t) * servo_env
        crack_sos = signal.butter(4, [2800.0, 9500.0], btype="bandpass", fs=AUDIO_SR, output="sos")
        crack_noise = signal.sosfilt(crack_sos, white)
        stress_env = np.clip((u - 0.20) / 0.32, 0.0, 1.0) * (u < 0.56)
        pops = (rng.random(n_samples) > (0.9992 - 0.004 * stress_env)).astype(np.float32)
        pops = signal.lfilter([1.0], [1.0, -0.96], pops) * crack_noise * 2.4
        shatter_env = np.exp(-np.maximum(0.0, t - 5.20) * 3.2) * (t >= 5.20)
        shatter_modes = (
            0.45 * np.sin(2.0 * np.pi * 2420.0 * t)
            + 0.35 * np.sin(2.0 * np.pi * 3890.0 * t)
            + 0.25 * np.sin(2.0 * np.pi * 5620.0 * t)
        ) * shatter_env
        shard_impulses = np.zeros(n_samples, dtype=np.float32)
        for k in range(28):
            tk = 5.25 + k * 0.09 + float(rng.uniform(0.0, 0.04))
            idx = int(tk * AUDIO_SR)
            if idx < n_samples - 2000:
                dt_k = np.arange(2000, dtype=np.float32) / AUDIO_SR
                freq_k = 3200.0 + float(rng.uniform(-900.0, 2200.0))
                shard_impulses[idx : idx + 2000] += np.sin(2.0 * np.pi * freq_k * dt_k) * np.exp(-dt_k * 35.0) * 0.22
        mono = servo_hum + pops + shatter_modes + shard_impulses + tex * (0.08 + 0.35 * shatter_env)
        left = mono + 0.15 * np.roll(shard_impulses, 420)
        right = mono + 0.15 * np.roll(shard_impulses, -420)

    elif spec.audio_mode == "iridescent_slice":
        slice_env = np.exp(-((u - 0.33) ** 2) * 28.0)
        sos_slice = signal.butter(4, [1600.0, 7800.0], btype="bandpass", fs=AUDIO_SR, output="sos")
        blade_fric = signal.sosfilt(sos_slice, white) * slice_env * 0.85
        layer_ticks = np.zeros(n_samples, dtype=np.float32)
        for k in range(24):
            tk = 1.0 + k * (4.6 / 24.0)
            idx = int(tk * AUDIO_SR)
            if idx < n_samples - 1200:
                dt_k = np.arange(1200, dtype=np.float32) / AUDIO_SR
                layer_ticks[idx : idx + 1200] += np.sin(2.0 * np.pi * (1900.0 + k * 85.0) * dt_k) * np.exp(-dt_k * 55.0) * 0.28
        slide_env = np.exp(-((u - 0.72) ** 2) * 35.0)
        sos_low = signal.butter(4, [240.0, 1400.0], btype="bandpass", fs=AUDIO_SR, output="sos")
        slide_snd = signal.sosfilt(sos_low, white) * slide_env * 0.55
        left = blade_fric * 0.9 + layer_ticks + slide_snd + tex * slice_env * 0.22
        right = blade_fric * 1.1 + np.roll(layer_ticks, 180) + slide_snd * 0.85 + tex * slice_env * 0.20

    elif spec.audio_mode == "helical_lathe":
        rpm_hz = 74.0
        lathe_hum = (
            0.22 * np.sin(2.0 * np.pi * rpm_hz * t)
            + 0.14 * np.sin(2.0 * np.pi * rpm_hz * 3.0 * t)
            + 0.08 * np.sin(2.0 * np.pi * 920.0 * t + 0.5 * np.sin(2.0 * np.pi * 12.0 * t))
        )
        sos_chip = signal.butter(4, [2200.0, 10500.0], btype="bandpass", fs=AUDIO_SR, output="sos")
        chip_stream = signal.sosfilt(sos_chip, white)
        mod = 0.65 + 0.35 * np.sin(2.0 * np.pi * 11.5 * t)
        sparks = (rng.random(n_samples) > 0.9985).astype(np.float32)
        sparks = signal.lfilter([1.0], [1.0, -0.94], sparks) * chip_stream * 1.8
        pan = 0.5 + 0.45 * np.sin(2.0 * np.pi * 0.12 * t)
        core = lathe_hum + chip_stream * mod * 0.52 + sparks + tex * 0.24
        left = core * (1.2 - pan)
        right = core * (0.2 + pan)

    else:  # prismatic_squeegee
        sos_foam = signal.butter(4, [1800.0, 11000.0], btype="bandpass", fs=AUDIO_SR, output="sos")
        fizz_base = signal.sosfilt(sos_foam, white)
        bubbles = (rng.random(n_samples) > 0.9972).astype(np.float32)
        bubbles = signal.lfilter([1.0], [1.0, -0.93], bubbles) * fizz_base * 2.2
        sos_glide = signal.butter(4, [320.0, 1900.0], btype="bandpass", fs=AUDIO_SR, output="sos")
        glide_env = 0.55 + 0.45 * np.sin(np.pi * u)
        squeegee_glide = signal.sosfilt(sos_glide, white) * glide_env * 0.65
        hex_rib = 0.08 * np.sin(2.0 * np.pi * 240.0 * t) * glide_env
        left = bubbles * 0.85 + squeegee_glide + hex_rib + tex * 0.26
        right = np.roll(bubbles, 310) * 0.85 + squeegee_glide * 0.95 + hex_rib + tex * 0.24

    stereo = np.stack([left, right], axis=-1)
    peak = float(np.max(np.abs(stereo))) + 1e-6
    stereo = np.clip(stereo / peak * 0.88, -1.0, 1.0)
    pcm16 = (stereo * 32767.0).astype(np.int16)

    with wave.open(str(wav_out), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_SR)
        wf.writeframes(pcm16.tobytes())


def render_video_with_audio(spec: SceneSpec, device: torch.device) -> Path:
    """Renderiza os 600 frames na RX 9070 XT e faz mux/masterização final com FFmpeg."""
    CRIADOS_DIR.mkdir(parents=True, exist_ok=True)
    out_mp4 = CRIADOS_DIR / spec.output_filename
    render_fn = RENDERERS[spec.key]

    x, y = make_ray_grid(RENDER_H, RENDER_W, device)

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_wav = Path(tmpdir) / f"{spec.key}_raw.wav"
        asmr_wav = Path(tmpdir) / f"{spec.key}_asmr.wav"
        synthesize_asmr_audio(spec, raw_wav)

        res_audio = subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(raw_wav),
                "-af",
                "highpass=f=60,treble=g=4:f=4500,acompressor=threshold=-18dB:ratio=3:attack=5:release=50,loudnorm=I=-14:TP=-1.5:LRA=11",
                str(asmr_wav),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res_audio.returncode != 0 or not asmr_wav.exists():
            raise RuntimeError(f"FFmpeg audio master falhou: {res_audio.stderr.strip()}")

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-nostats",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{RENDER_W}x{RENDER_H}",
            "-r",
            str(FPS),
            "-i",
            "-",
            "-i",
            str(asmr_wav),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-vf",
            f"scale={OUT_W}:{OUT_H}:flags=lanczos,unsharp=5:5:0.65:5:5:0.0,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "17",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out_mp4),
        ]

        err_path = Path(tmpdir) / f"{spec.key}_ffmpeg.err"
        with open(err_path, "wb") as err_file:
            proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=err_file,
            )
            assert proc.stdin is not None
            try:
                with torch.no_grad():
                    for idx in range(TOTAL_FRAMES):
                        frame_u8 = render_fn(idx, TOTAL_FRAMES, x, y, device)
                        proc.stdin.write(frame_u8.tobytes())
                        if (idx + 1) % 150 == 0:
                            print(f"    [{spec.key}] {idx + 1}/{TOTAL_FRAMES} frames...", flush=True)
            except BaseException:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
                proc.kill()
                proc.wait()
                raise
            finally:
                if proc.stdin and not proc.stdin.closed:
                    proc.stdin.close()
            ret = proc.wait()

        if ret != 0:
            err_msg = err_path.read_text(encoding="utf-8", errors="replace") if err_path.exists() else ""
            raise RuntimeError(
                f"FFmpeg video encode falhou para {spec.output_filename}: {err_msg}"
            )

    return out_mp4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Renderiza 4 vídeos ASMR 100% originais em GPU (RX 9070 XT) sem logos ou marcas."
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=[s.key for s in SCENES],
        default=[s.key for s in SCENES],
        help="Subconjunto de cenas a renderizar (padrao: todas as 4).",
    )
    return parser.parse_args()


def main() -> int:
    if not torch.cuda.is_available():
        print("[ERRO] GPU ROCm/CUDA nao detectada neste interpretador Python.", file=sys.stderr)
        return 1

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU ativa: {gpu_name} | Renderizando {RENDER_W}x{RENDER_H} -> {OUT_W}x{OUT_H} @ {FPS}fps ({TOTAL_FRAMES} frames/video)")

    args = parse_args()
    selected = [s for s in SCENES if s.key in args.scenes]

    for spec in selected:
        print(f"\n==> Gerando [{spec.key}] -> {spec.output_filename}")
        out_path = render_video_with_audio(spec, device)
        size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"    OK: {out_path.relative_to(BASE_DIR).as_posix()} ({size_mb:.2f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
