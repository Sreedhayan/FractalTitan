#!/usr/bin/env python3
"""
hyperfractrix.py

A deliberately enormous, compute-heavy, "super duper" Mandelbulb-like raymarch renderer in Python.
Now runs infinitely in a "live" loop, evolving fractal and camera over time.

WARNING: Defaults will require huge RAM/CPU/GPU and may crash or hang any normal machine.
This infinite loop is intended to stress test or conceptually visualize fractal evolution.
############!!!!!!!!!!>>>>>>>>run at your own risk<<<<<<<<!!!!!!!!!!############
"""

from __future__ import annotations
import os
import sys
import math
import time
import json
import argparse
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Tuple, Optional
import numpy as np
from PIL import Image
import traceback

# Optional acceleration libraries (import lazily)
try:
    import numba
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False

# ---------------------------
# CONFIG (colossal defaults)
# ---------------------------
@dataclass
class Config:
    width: int = 32768
    height: int = 32768
    tile_size: int = 2048
    num_workers: int = max(1, (mp.cpu_count() - 1))
    max_ray_steps: int = 4000
    max_fractal_iters: int = 2000
    escape_threshold: float = 1e6
    power_base: float = 8.0
    growth_speed: float = 0.15
    samples_per_pixel: int = 8
    min_dist: float = 1e-6
    max_dist: float = 1000.0
    out_dir: str = "colossal_output"
    out_basename: str = "colossal_mandelbulb"
    tile_format: str = "png"
    save_tiles: bool = True
    stitch_tiles: bool = True
    dry_run: bool = False
    verbose: bool = True
    use_numba: bool = NUMBA_AVAILABLE
    use_cupy: bool = CUPY_AVAILABLE
    rng_seed: int = 42

cfg = Config()

# ---------------------------
# Utilities
# ---------------------------
def dbg(*args, **kwargs):
    if cfg.verbose:
        print(*args, **kwargs)

def ensure_out():
    os.makedirs(cfg.out_dir, exist_ok=True)

# ---------------------------
# Math / Fractal functions
# ---------------------------
def sph_coords(zx, zy, zz):
    r = math.sqrt(zx*zx + zy*zy + zz*zz)
    if r == 0.0:
        return 0.0, 0.0, 0.0
    theta = math.acos(zz / r)
    phi = math.atan2(zy, zx)
    return r, theta, phi

def mandelbulb_de_point(px: float, py: float, pz: float, power: float, iters: int) -> float:
    zx, zy, zz = px, py, pz
    dr = 1.0
    r = 0.0
    for i in range(iters):
        r = math.sqrt(zx*zx + zy*zy + zz*zz)
        if r > cfg.escape_threshold:
            break
        if r == 0.0:
            theta = 0.0
            phi = 0.0
        else:
            theta = math.acos(zz / r)
            phi = math.atan2(zy, zx)
        zr = r ** power
        theta *= power
        phi *= power
        sin_t = math.sin(theta)
        zx = zr * sin_t * math.cos(phi) + px
        zy = zr * sin_t * math.sin(phi) + py
        zz = zr * math.cos(theta) + pz
        dr = dr * power * (r ** (power - 1.0)) + 1.0
    if r == 0.0:
        return 0.0
    return 0.5 * math.log(r) * r / dr

# Optional Numba accelerated version
if NUMBA_AVAILABLE:
    @numba.njit(cache=True)
    def mandelbulb_de_point_numba(px, py, pz, power, iters, escape_threshold):
        zx, zy, zz = px, py, pz
        dr = 1.0
        r = 0.0
        for i in range(iters):
            r = math.sqrt(zx*zx + zy*zy + zz*zz)
            if r > escape_threshold:
                break
            if r == 0.0:
                theta = 0.0
                phi = 0.0
            else:
                theta = math.acos(zz / r)
                phi = math.atan2(zy, zx)
            zr = r ** power
            theta *= power
            phi *= power
            sin_t = math.sin(theta)
            zx = zr * sin_t * math.cos(phi) + px
            zy = zr * sin_t * math.sin(phi) + py
            zz = zr * math.cos(theta) + pz
            dr = dr * power * (r ** (power - 1.0)) + 1.0
        if r == 0.0:
            return 0.0
        return 0.5 * math.log(r) * r / dr

# ---------------------------
# Raymarching core
# ---------------------------
def raymarch_pixel(rx, ry, rz, rdx, rdy, rdz, t_time: float, max_steps: int, iters: int, power: float) -> Tuple[float, float, float]:
    total_dist = 0.0
    steps = 0
    growth = 1.0 + cfg.growth_speed * t_time
    dyn_power = power * growth
    for i in range(max_steps):
        steps += 1
        px = rx + rdx * total_dist
        py = ry + rdy * total_dist
        pz = rz + rdz * total_dist
        warp = 0.15 * math.sin(1.5*px) * math.cos(1.2*pz) + 0.07 * math.sin(3.7*py)
        px += warp
        py += 0.6 * math.cos(2.2*px*0.3)
        d = mandelbulb_de_point(px, py, pz, dyn_power, iters)
        total_dist += d
        if d < cfg.min_dist:
            return total_dist, 1.0, float(steps)
        if total_dist > cfg.max_dist:
            return total_dist, 0.0, float(steps)
    return total_dist, 0.0, float(steps)

# ---------------------------
# Tile renderer worker
# ---------------------------
def render_tile_worker(args):
    try:
        (tile_x, tile_y, tile_w, tile_h, canvas_w, canvas_h, camera_params_json, seed, tile_index) = args
        camera_params = json.loads(camera_params_json)
        rng = np.random.RandomState(seed + tile_index * 1009)
        dbg(f"[worker] tile {tile_x},{tile_y} size {tile_w}x{tile_h} seed {seed}")

        tile = np.zeros((tile_h, tile_w, 3), dtype=np.float32)
        fov = camera_params.get("fov", 45.0)
        aspect = canvas_w / canvas_h
        cam_pos = camera_params.get("pos", [3.2, 0.9, 2.5])
        target = camera_params.get("target", [0.0, 0.0, 0.0])
        up = camera_params.get("up", [0.0, 1.0, 0.0])
        cx, cy, cz = cam_pos
        tx, ty, tz = target
        forward = np.array([tx-cx, ty-cy, tz-cz], dtype=np.float64)
        forward /= np.linalg.norm(forward) + 1e-12
        right = np.cross(forward, np.array(up, dtype=np.float64))
        right /= (np.linalg.norm(right) + 1e-12)
        upv = np.cross(right, forward)
        scale = math.tan(math.radians(fov * 0.5))

        for py in range(tile_h):
            for px in range(tile_w):
                gx = tile_x + px
                gy = tile_y + py
                ndc_x = ( (gx + 0.5) / canvas_w ) * 2.0 - 1.0
                ndc_y = ( (gy + 0.5) / canvas_h ) * 2.0 - 1.0
                ndc_x *= aspect * scale
                ndc_y *= scale
                color_accum = np.zeros(3, dtype=np.float64)
                for s in range(cfg.samples_per_pixel):
                    jx = (rng.rand() - 0.5) * (1.0 / canvas_w)
                    jy = (rng.rand() - 0.5) * (1.0 / canvas_h)
                    sx = ndc_x + jx
                    sy = ndc_y + jy
                    rd = forward + sx*right + sy*upv
                    rd = rd / np.linalg.norm(rd)
                    rox, roy, roz = cx, cy, cz
                    dist, hit, steps_used = raymarch_pixel(rox, roy, roz, rd[0], rd[1], rd[2],
                                                           camera_params.get("time", 0.0),
                                                           cfg.max_ray_steps, cfg.max_fractal_iters,
                                                           cfg.power_base)
                    if hit > 0.5:
                        eps = 1e-3
                        p_hit_x = rox + rd[0] * dist
                        p_hit_y = roy + rd[1] * dist
                        p_hit_z = roz + rd[2] * dist
                        dx = mandelbulb_de_point(p_hit_x + eps, p_hit_y, p_hit_z, cfg.power_base, cfg.max_fractal_iters) - \
                             mandelbulb_de_point(p_hit_x - eps, p_hit_y, p_hit_z, cfg.power_base, cfg.max_fractal_iters)
                        dy = mandelbulb_de_point(p_hit_x, p_hit_y + eps, p_hit_z, cfg.power_base, cfg.max_fractal_iters) - \
                             mandelbulb_de_point(p_hit_x, p_hit_y - eps, p_hit_z, cfg.power_base, cfg.max_fractal_iters)
                        dz = mandelbulb_de_point(p_hit_x, p_hit_y, p_hit_z + eps, cfg.power_base, cfg.max_fractal_iters) - \
                             mandelbulb_de_point(p_hit_x, p_hit_y, p_hit_z - eps, cfg.power_base, cfg.max_fractal_iters)
                        normal = np.array([dx, dy, dz], dtype=np.float64)
                        normal /= (np.linalg.norm(normal)+1e-12)
                        l1 = np.array([0.8, 0.6, -0.7])
                        l1 /= np.linalg.norm(l1)
                        diff = max(np.dot(normal, l1), 0.0)
                        spec = pow(max(np.dot(np.array([-rd[0], -rd[1], -rd[2]]), (2*diff*normal - l1)), 0.0), 32.0)
                        ao = 1.0
                        base = np.array([0.5 + 0.5*math.sin(dist*0.01), 0.3 + 0.6*math.cos(dist*0.02), 0.4 + 0.5*math.sin(dist*0.013)])
                        shaded = base * (0.8*diff*ao) + spec * 0.5
                        color_accum += shaded
                    else:
                        fog = math.exp(-0.0005 * dist)
                        basebg = np.array([0.01, 0.02, 0.05]) * fog
                        color_accum += basebg
                tile[py, px, :] = color_accum / float(cfg.samples_per_pixel)
            if py % 256 == 0:
                dbg(f"tile {tile_x},{tile_y} row {py}/{tile_h}")
        tile_clamped = np.clip(tile, 0.0, 1.0)
        tile_img = (tile_clamped * 255.0).astype(np.uint8)
        if cfg.save_tiles:
            ensure_out()
            tile_fname = os.path.join(cfg.out_dir, f"{cfg.out_basename}_tile_{tile_x}_{tile_y}.{cfg.tile_format}")
            img = Image.fromarray(tile_img, mode='RGB')
            img.save(tile_fname)
            dbg(f"Saved tile {tile_fname}")
        return True, (tile_x, tile_y), tile_img
    except Exception as e:
        tb = traceback.format_exc()
        dbg("Worker exception:", e, tb)
        return False, (tile_x, tile_y), None

# ---------------------------
# Orchestrator / infinite live loop
# ---------------------------
def generate_camera_params(time_sec: float = 0.0):
    yaw = time_sec * 0.18
    pitch = 0.35 + 0.15 * math.sin(time_sec * 0.12)
    return {
        "fov": 45.0,
        "pos": [3.2 * (1.0 - 0.2*math.sin(time_sec*0.25)), 0.9 - 0.25*math.sin(time_sec*0.12), 2.5],
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "yaw": yaw,
        "pitch": pitch,
        "time": time_sec
    }

def plan_tiles(width, height, tile_size):
    tiles = []
    y = 0
    idx = 0
    while y < height:
        x = 0
        th = min(tile_size, height - y)
        while x < width:
            tw = min(tile_size, width - x)
            tiles.append((x, y, tw, th, idx))
            idx += 1
            x += tw
        y += th
    return tiles

def main_infinite_render():
    frame = 0
    while True:
        t_start = time.time()
        dbg(f"\n=== FRAME {frame} ===")
        t_time = frame * 0.1
        camera_params = generate_camera_params(time_sec=t_time)
        camera_json = json.dumps(camera_params)
        tiles_info = plan_tiles(cfg.width, cfg.height, cfg.tile_size)
        worker_args = [(x, y, w, h, cfg.width, cfg.height, camera_json, cfg.rng_seed, idx)
                       for (x, y, w, h, idx) in tiles_info]
        dbg(f"Launching {len(worker_args)} tiles with {cfg.num_workers} workers (frame {frame})")
        try:
            with mp.Pool(processes=cfg.num_workers) as pool:
                pool.map(render_tile_worker, worker_args)
        except Exception as e:
            dbg("Worker pool error:", e)
        t_end = time.time()
        dbg(f"Frame {frame} complete in {t_end - t_start:.2f} seconds (impossible to finish!)")
        frame += 1

# ---------------------------
# CLI
# ---------------------------
def parse_cli():
    p = argparse.ArgumentParser(description="Colossal Mandelbulb Live Renderer (impossible specs).")
    p.add_argument("--width", type=int, default=cfg.width)
    p.add_argument("--height", type=int, default=cfg.height)
    p.add_argument("--tile", type=int, default=cfg.tile_size)
    p.add_argument("--workers", type=int, default=cfg.num_workers)
    p.add_argument("--steps", type=int, default=cfg.max_ray_steps)
    p.add_argument("--iters", type=int, default=cfg.max_fractal_iters)
    p.add_argument("--samples", type=int, default=cfg.samples_per_pixel)
    p.add_argument("--dry", action="store_true", help="dry run (don't actually render)")
    p.add_argument("--no-stitch", action="store_true", help="don't stitch tiles")
    p.add_argument("--verbose", action="store_true", help="verbose logging")
    return p.parse_args()

if __name__ == "__main__":
    try:
        args = parse_cli()
        cfg.width = args.width
        cfg.height = args.height
        cfg.tile_size = args.tile
        cfg.num_workers = args.workers
        cfg.max_ray_steps = args.steps
        cfg.max_fractal_iters = args.iters
        cfg.samples_per_pixel = args.samples
        cfg.dry_run = args.dry
        if args.no_stitch:
            cfg.stitch_tiles = False
        if args.verbose:
            cfg.verbose = True
        dbg("Starting colossal infinite live renderer (impossible specs).")
        main_infinite_render()
    except KeyboardInterrupt:
        dbg("Interrupted by user.")
    except Exception as e:
        dbg("Fatal error:", e)
        traceback.print_exc()
