import argparse
import sys
import time

import cupy as cp
import numpy as np

import GR
import func


BLACKHOLE = {
    "MASS": 1.0,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
}


def make_camera(width, height):
    camera = {
        "FOV": np.deg2rad(75),
        "x": 0.0,
        "y": -75.0,
        "z": 0.0,
        "angle_vertical": np.pi / 2,
        "angle_horizontal": np.pi / 2,
        "width": width,
        "height": height,
        "camera_virtual_screen_width": 1.0,
    }
    camera["distance_from_virtual_screen"] = (
        camera["camera_virtual_screen_width"] / (2.0 * np.tan(camera["FOV"] / 2.0))
    )
    return camera


def camera_radius(camera):
    camera_position = np.array([camera["x"], camera["y"], camera["z"]], dtype=np.float64)
    blackhole_position = np.array([BLACKHOLE["x"], BLACKHOLE["y"], BLACKHOLE["z"]], dtype=np.float64)
    return np.linalg.norm(camera_position - blackhole_position)


def compute_lut(camera, rays_number, max_steps):
    r = camera_radius(camera)
    beta_grid = cp.linspace(0.0, cp.pi, rays_number, dtype=cp.float64)

    t0 = time.perf_counter()
    final_states = GR.orbital_geodesic_fast(
        r,
        BLACKHOLE["MASS"],
        RAYS_NUMBER=rays_number,
        MAX_STEPS=max_steps,
    )
    cp.cuda.Stream.null.synchronize()
    print(f"LUT/deviations: {time.perf_counter() - t0:.3f}s")

    return beta_grid, final_states


def generate_frames(camera, skybox, beta_grid, final_states, frames_count, rotation_degrees):
    """Calcule les images du tour de caméra une seule fois.

    Comme ça, le benchmark matplotlib vs OpenGL mesure surtout l'affichage, pas la GR.
    """
    initial_angle = camera["angle_horizontal"]
    start_angle = initial_angle + np.deg2rad(rotation_degrees) / 2.0
    angle_step = np.deg2rad(rotation_degrees) / max(frames_count - 1, 1)

    frames_rgb_u8 = []

    t0 = time.perf_counter()
    render_time = 0.0
    copy_time = 0.0
    for frame_id in range(frames_count):
        camera["angle_horizontal"] = start_angle - frame_id * angle_step

        t_render = time.perf_counter()
        image, _, _ = func.render_skybox_from_orbital_lut(
            camera,
            BLACKHOLE,
            beta_grid,
            final_states,
            skybox,
        )
        cp.cuda.Stream.null.synchronize()
        render_time += time.perf_counter() - t_render

        t_copy = time.perf_counter()
        image_cpu = cp.asnumpy(image)
        frame = np.clip(image_cpu * 255.0, 0.0, 255.0).astype(np.uint8)
        frames_rgb_u8.append(np.ascontiguousarray(frame))
        copy_time += time.perf_counter() - t_copy

    cp.cuda.Stream.null.synchronize()
    print(f"render GPU for {frames_count} frames: {render_time:.3f}s")
    print(f"GPU->CPU copy + uint8 for {frames_count} frames: {copy_time:.3f}s")
    print(f"render + GPU->CPU copy for {frames_count} frames: {time.perf_counter() - t0:.3f}s")

    camera["angle_horizontal"] = initial_angle
    return frames_rgb_u8


def benchmark_matplotlib(frames_rgb_u8, pause=0.001):
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots()
    ax.axis("off")

    artist = ax.imshow(frames_rgb_u8[0])
    plt.show(block=False)
    fig.canvas.draw()
    fig.canvas.flush_events()

    t0 = time.perf_counter()
    for frame in frames_rgb_u8:
        artist.set_data(frame)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(pause)
    dt = time.perf_counter() - t0

    fps = len(frames_rgb_u8) / dt
    print(f"matplotlib display: {dt:.3f}s, {fps:.1f} FPS")

    plt.ioff()
    plt.close(fig)
    return dt


def benchmark_opengl(frames_rgb_u8):
    """Affichage OpenGL très simple : une texture plein écran mise à jour à chaque frame.

    Nécessite : pygame + PyOpenGL.
    Installation si besoin :
        python3.14 -m pip install PyOpenGL
    """
    try:
        import pygame
        from OpenGL.GL import (
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR,
            GL_RGB,
            GL_TEXTURE_2D,
            GL_TEXTURE_MAG_FILTER,
            GL_TEXTURE_MIN_FILTER,
            GL_UNSIGNED_BYTE,
            GL_QUADS,
            glBegin,
            glBindTexture,
            glClear,
            glEnable,
            glEnd,
            glGenTextures,
            glTexCoord2f,
            glTexImage2D,
            glTexParameteri,
            glTexSubImage2D,
            glVertex2f,
        )
    except ModuleNotFoundError as exc:
        print(f"OpenGL display skipped: missing module {exc.name!r}")
        print(f"Current Python: {sys.executable}")
        print("Use the project venv instead:")
        print("  ../.venv/bin/python benchmark_display.py --backend opengl")
        print("or activate it first:")
        print("  source ../.venv/bin/activate")
        return None

    height, width, _ = frames_rgb_u8[0].shape

    pygame.init()
    pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Black hole OpenGL display benchmark")

    glEnable(GL_TEXTURE_2D)
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # Préparation hors chrono : OpenGL a l'origine texture en bas à gauche.
    frames_gl = [np.ascontiguousarray(np.flipud(frame)) for frame in frames_rgb_u8]

    first_frame = frames_gl[0]
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, first_frame)

    def draw_frame(frame_gl):
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, frame_gl)

        glClear(GL_COLOR_BUFFER_BIT)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0)
        glEnd()
        pygame.display.flip()

    # warmup
    draw_frame(frames_gl[0])

    t0 = time.perf_counter()
    for frame in frames_gl:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        draw_frame(frame)
    dt = time.perf_counter() - t0

    fps = len(frames_rgb_u8) / dt
    print(f"OpenGL display: {dt:.3f}s, {fps:.1f} FPS")

    time.sleep(0.5)
    pygame.quit()
    return dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["matplotlib", "opengl", "both"], default="both")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--degrees", type=float, default=30.0)
    parser.add_argument("--rays", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=10_000)
    args = parser.parse_args()

    camera = make_camera(args.width, args.height)
    skybox = func.load_skybox("source/skybox.png")

    beta_grid, final_states = compute_lut(camera, args.rays, args.max_steps)
    frames = generate_frames(camera, skybox, beta_grid, final_states, args.frames, args.degrees)

    print(f"benchmark: {args.frames} frames over {args.degrees} degrees at {args.width}x{args.height}")

    if args.backend in ("matplotlib", "both"):
        benchmark_matplotlib(frames)

    if args.backend in ("opengl", "both"):
        benchmark_opengl(frames)


if __name__ == "__main__":
    main()
