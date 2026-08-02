import argparse
import os
import subprocess
import time

import imageio.v3 as iio
import numpy as np

import func
import main
from disk_cache import DiskLUTCache
from lut_cache import OrbitalLUTCache


def look_at(camera, target):
    """Oriente CAMERA vers target avec la convention sphérique du projet."""
    dx = target[0] - camera["x"]
    dy = target[1] - camera["y"]
    dz = target[2] - camera["z"]

    r_xy = np.sqrt(dx * dx + dy * dy)
    camera["angle_horizontal"] = np.arctan2(dy, dx)
    camera["angle_vertical"] = np.clip(np.arctan2(r_xy, dz), 1e-3, np.pi - 1e-3)


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


def set_camera_resolution(width, height):
    main.CAMERA["width"] = int(width)
    main.CAMERA["height"] = int(height)
    main.CAMERA["distance_from_virtual_screen"] = (
        main.CAMERA["camera_virtual_screen_width"]
        / (2.0 * np.tan(main.CAMERA["FOV"] / 2.0))
    )


def render_frames(args):
    os.makedirs(args.frames_dir, exist_ok=True)

    main.ENABLE_DISK_OVERLAY = not args.no_disk
    main.RAYS_NUMBER = args.rays
    main.MAX_STEPS = args.steps
    main.LUT_DR = args.lut_dr
    set_camera_resolution(args.width, args.height)

    skybox = func.load_skybox(args.skybox)

    lut_cache = OrbitalLUTCache(
        M=main.BLACKHOLE["MASS"],
        rays_number=main.RAYS_NUMBER,
        max_steps=main.MAX_STEPS,
        dr=main.LUT_DR,
    )
    disk_cache = DiskLUTCache(
        M=main.BLACKHOLE["MASS"],
        rays_number=main.RAYS_NUMBER,
        max_steps=main.MAX_STEPS,
        dr=main.LUT_DR,
    )

    target = np.asarray([args.look_x, args.look_y, args.look_z], dtype=np.float64)

    t0 = time.perf_counter()
    for frame_id in range(args.frames):
        u = frame_id / max(args.frames - 1, 1)

        # Smoothstep pour éviter un départ/arrêt trop mécanique.
        if args.smooth_path:
            u_path = u * u * (3.0 - 2.0 * u)
        else:
            u_path = u

        main.CAMERA["x"] = lerp(args.start_x, args.end_x, u_path)
        main.CAMERA["y"] = lerp(args.start_y, args.end_y, u_path)
        main.CAMERA["z"] = lerp(args.start_z, args.end_z, u_path)
        look_at(main.CAMERA, target)
        print("x = ", main.CAMERA["x"], " | y = ", main.CAMERA["y"])

        current_r = func.camera_radius(main.CAMERA, main.BLACKHOLE)
        beta_grid, final_states = lut_cache.get_interpolated(current_r)

        disk_crossing_lut = None
        if main.ENABLE_DISK_OVERLAY:
            disk_crossing_lut = disk_cache.get(current_r)

        frame = main.render_frame(skybox, beta_grid, final_states, disk_crossing_lut)
        path = os.path.join(args.frames_dir, f"frame_{frame_id:04d}.png")
        iio.imwrite(path, frame)

        elapsed = time.perf_counter() - t0
        print(
            f"[{frame_id + 1:04d}/{args.frames:04d}] "
            f"y={main.CAMERA['y']:.2f} r={current_r:.2f} "
            f"saved {path} elapsed={elapsed:.1f}s"
        )


def encode_video(args):
    input_pattern = os.path.join(args.frames_dir, "frame_%04d.png")
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-crf",
        str(args.crf),
        "-preset",
        args.preset,
        "-pix_fmt",
        "yuv420p",
        args.output,
    ]
    print("encoding:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline high-res video renderer for the orbital-LUT black-hole scene."
    )

    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=1440)
    parser.add_argument("--rays", type=int, default=5000)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--lut-dr", type=float, default=1.0)

    # Change ces valeurs en CLI si ce n'était pas ça.
    parser.add_argument("--start-x", type=float, default=150.0)
    parser.add_argument("--start-y", type=float, default=-30.0)
    parser.add_argument("--start-z", type=float, default=8.0)
    parser.add_argument("--end-x", type=float, default=-150.0)
    parser.add_argument("--end-y", type=float, default=-30.0)
    parser.add_argument("--end-z", type=float, default=8.0)

    parser.add_argument("--look-x", type=float, default=0.0)
    parser.add_argument("--look-y", type=float, default=0.0)
    parser.add_argument("--look-z", type=float, default=0.0)
    parser.add_argument("--smooth-path", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--frames-dir", default="video_frames")
    parser.add_argument("--output", default="blackhole_disk_video.mp4")
    parser.add_argument("--skybox", default="source/skybox.png")
    parser.add_argument("--crf", type=int, default=16)
    parser.add_argument("--preset", default="slow")
    parser.add_argument("--no-disk", action="store_true")
    parser.add_argument("--no-encode", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    render_frames(args)
    if not args.no_encode:
        encode_video(args)
