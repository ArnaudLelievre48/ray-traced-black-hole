import time

import cupy as cp
import numpy as np
import pygame
import os

import func
from disk_cache import DiskLUTCache
from display import CudaOpenGLImageDisplay
from gpu_disk_renderer import DiskGpuRenderer
from gpu_renderer import OrbitalGpuRenderer
from lut_cache import OrbitalLUTCache


BLACKHOLE = {
    "MASS": 1.0,
    "x": 0.0,
    "y": 0.0,
    "z": 0.0,
}

CAMERA = {
    "FOV": np.deg2rad(75),
    "x": 0.0,
    "y": -70.0,
    "z": 8.0,
    "angle_vertical": np.pi / 2 + np.deg2rad(7),
    "angle_horizontal": np.pi / 2,
    "width": 1280,
    "height": 720,
    "camera_virtual_screen_width": 1.0,
}
CAMERA["distance_from_virtual_screen"] = (
    CAMERA["camera_virtual_screen_width"] / (2.0 * np.tan(CAMERA["FOV"] / 2.0))
)

RAYS_NUMBER = 1000
MAX_STEPS = 250_000
#LUT_DR = 1.0
LUT_DR = 0.1
STARTUP_PRECOMPUTE_MARGIN = 5.0
IDLE_PREFETCH_MARGIN = 5.0
ENABLE_DISK_OVERLAY = False

ANGLE_SPEED = np.deg2rad(90.0)  # rad/s
MOVE_SPEED = 10.0                # unités de coordonnée / s
MOUSE_SENSITIVITY = 0.0005        # rad/pixel

OUT_DIR = "video_frames"
os.makedirs(OUT_DIR, exist_ok=True)

_GPU_RENDERER = None
_GPU_RENDERER_SKYBOX_POINTER = None
_DISK_GPU_RENDERER = None


def compute_disk_lut_for_current_camera(disk_cache):
    if not ENABLE_DISK_OVERLAY:
        return None

    current_r = func.camera_radius(CAMERA, BLACKHOLE)
    return disk_cache.get(current_r)


def get_gpu_renderer(skybox):
    """Réutilise texture CUDA et kernels compilés tant que la skybox ne change pas."""
    global _GPU_RENDERER, _GPU_RENDERER_SKYBOX_POINTER

    skybox_pointer = int(skybox.data.ptr)
    if _GPU_RENDERER is None or _GPU_RENDERER_SKYBOX_POINTER != skybox_pointer:
        _GPU_RENDERER = OrbitalGpuRenderer(skybox)
        _GPU_RENDERER_SKYBOX_POINTER = skybox_pointer
    return _GPU_RENDERER


def get_disk_gpu_renderer():
    global _DISK_GPU_RENDERER
    if _DISK_GPU_RENDERER is None:
        _DISK_GPU_RENDERER = DiskGpuRenderer()
    return _DISK_GPU_RENDERER


def prepare_disk_render_lut(disk_renderer, disk_crossing_lut):
    if disk_crossing_lut is None:
        return None
    return disk_renderer.prepare_lut(*disk_crossing_lut)


def render_frame_gpu(
    skybox,
    beta_grid,
    final_states,
    disk_crossing_lut=None,
    renderer=None,
    render_lut=None,
    disk_renderer=None,
    disk_render_lut=None,
):
    """Rend l'image dans la VRAM, sans recalculer la géodésique."""
    if renderer is None:
        renderer = get_gpu_renderer(skybox)
    if render_lut is None:
        render_lut = renderer.prepare_lut(beta_grid, final_states, BLACKHOLE["MASS"])

    image_gpu = renderer.render_rgb(
        CAMERA,
        BLACKHOLE,
        render_lut,
    )
    if ENABLE_DISK_OVERLAY and disk_crossing_lut is not None:
        if disk_renderer is None:
            disk_renderer = get_disk_gpu_renderer()
        if disk_render_lut is None:
            disk_render_lut = prepare_disk_render_lut(
                disk_renderer,
                disk_crossing_lut,
            )
        overlay = disk_renderer.render(
            CAMERA,
            BLACKHOLE,
            disk_render_lut,
        )
        cp.add(image_gpu, overlay, out=image_gpu)
        cp.clip(image_gpu, 0.0, 1.0, out=image_gpu)

    # Pas de synchronize() ici : le display consomme ce tableau sur le même
    # stream CUDA, puis l'interop CUDA/OpenGL assure l'ordre des opérations.
    return image_gpu


def update_interactive_frame(
    display,
    renderer,
    render_lut,
    disk_renderer,
    disk_render_lut,
    skybox,
    beta_grid,
    final_states,
    disk_crossing_lut,
):
    """Chemin direct PBO sans image RGB intermédiaire quand le disque est coupé."""
    if not ENABLE_DISK_OVERLAY or disk_render_lut is None:
        display.update_from_cuda(
            lambda output, stream: renderer.render_rgba8(
                CAMERA,
                BLACKHOLE,
                render_lut,
                output,
                stream,
            )
        )
        return

    image_gpu = render_frame_gpu(
        skybox,
        beta_grid,
        final_states,
        disk_crossing_lut,
        renderer=renderer,
        render_lut=render_lut,
        disk_renderer=disk_renderer,
        disk_render_lut=disk_render_lut,
    )
    display.update_image(image_gpu)


def download_frame(image_gpu):
    """Conversion explicite réservée aux exports PNG/vidéo hors ligne."""
    frame_gpu = cp.clip(image_gpu * 255.0, 0.0, 255.0).astype(cp.uint8)
    return cp.asnumpy(frame_gpu)


def render_frame(skybox, beta_grid, final_states, disk_crossing_lut=None):
    """Compatibilité pour le renderer hors ligne, qui doit écrire sur disque."""
    return download_frame(
        render_frame_gpu(skybox, beta_grid, final_states, disk_crossing_lut)
    )


def move_camera(delta):
    CAMERA["x"] += delta[0]
    CAMERA["y"] += delta[1]
    CAMERA["z"] += delta[2]


def handle_continuous_input(dt):
    """Lit l'état clavier courant.

    Contrairement aux KEYDOWN ponctuels, pygame.key.get_pressed() permet :
        - maintenir une touche enfoncée ;
        - combiner plusieurs touches, ex: z+d, z+space, left+up.

    Retourne (orientation_changed, position_changed, should_quit).
    """
    keys = pygame.key.get_pressed()

    orientation_changed = False
    position_changed = False
    should_quit = keys[pygame.K_ESCAPE]
    toggle_disk = keys[pygame.K_m]

    # Rotation continue.
    d_angle_h = 0.0
    d_angle_v = 0.0

    if keys[pygame.K_LEFT] or keys[pygame.K_j]:
        d_angle_h -= ANGLE_SPEED * dt
    if keys[pygame.K_RIGHT] or keys[pygame.K_l]:
        d_angle_h += ANGLE_SPEED * dt
    if keys[pygame.K_UP] or keys[pygame.K_i]:
        d_angle_v -= ANGLE_SPEED * dt
    if keys[pygame.K_DOWN] or keys[pygame.K_k]:
        d_angle_v += ANGLE_SPEED * dt

    if d_angle_h != 0.0 or d_angle_v != 0.0:
        CAMERA["angle_horizontal"] += d_angle_h
        CAMERA["angle_vertical"] = np.clip(
            CAMERA["angle_vertical"] + d_angle_v,
            1e-3,
            np.pi - 1e-3,
        )
        orientation_changed = True

    # Déplacement continu AZERTY : zqsd + espace/shift.
    forward, right, up = func.camera_basis(CAMERA)
    move = np.zeros(3, dtype=np.float64)

    if keys[pygame.K_z]:
        move += forward
    if keys[pygame.K_s]:
        move -= forward
    if keys[pygame.K_d]:
        move += right
    if keys[pygame.K_q]:
        move -= right
    if keys[pygame.K_SPACE]:
        move += up
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        move -= up

    move_norm = np.linalg.norm(move)
    if move_norm > 0.0:
        # Normalisation : z+d ne va pas sqrt(2) fois plus vite que z seul.
        move_camera(MOVE_SPEED * dt * move / move_norm)
        position_changed = True

    return orientation_changed, position_changed, should_quit, toggle_disk


def capture_mouse():
    """Capture la souris dans la fenêtre pour pouvoir regarder autour."""
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(False)
    pygame.mouse.get_rel()  # flush, évite un saut initial


def handle_mouse_look():
    """Tourne la caméra avec le mouvement relatif de la souris."""
    dx, dy = pygame.mouse.get_rel()

    if dx == 0 and dy == 0:
        return False

    CAMERA["angle_horizontal"] += dx * MOUSE_SENSITIVITY

    # dy positif = souris vers le bas. Le signe ci-dessous donne un feeling FPS :
    # souris vers le haut -> regarde vers le haut.
    CAMERA["angle_vertical"] = np.clip(
        CAMERA["angle_vertical"] + dy * MOUSE_SENSITIVITY,
        1e-3,
        np.pi - 1e-3,
    )

    return True


def main():
    global ENABLE_DISK_OVERLAY

    # Le contexte OpenGL doit exister avant la première allocation CUDA afin de
    # sélectionner le GPU commun aux deux API et d'enregistrer les PBO partagés.
    display = CudaOpenGLImageDisplay(
        CAMERA["width"],
        CAMERA["height"],
        title="Ray-traced black hole",
    )
    try:
        skybox = func.load_skybox("source/skybox.png")
        renderer = get_gpu_renderer(skybox)
        disk_renderer = get_disk_gpu_renderer()

        lut_cache = OrbitalLUTCache(
            M=BLACKHOLE["MASS"],
            rays_number=RAYS_NUMBER,
            max_steps=MAX_STEPS,
            dr=LUT_DR,
        )
        disk_cache = DiskLUTCache(
            M=BLACKHOLE["MASS"],
            rays_number=RAYS_NUMBER,
            max_steps=MAX_STEPS,
            dr=LUT_DR,
        )

        # Démarrage volontairement un peu plus long : on prend un peu d'avance.
        # Le reste de la marge +/-20 sera rempli pendant les temps morts.
        current_r = func.camera_radius(CAMERA, BLACKHOLE)
        lut_cache.precompute_margin_around(current_r, margin=STARTUP_PRECOMPUTE_MARGIN)
        beta_grid, final_states = lut_cache.get_interpolated(current_r)
        render_lut = renderer.prepare_lut(beta_grid, final_states, BLACKHOLE["MASS"])
        disk_crossing_lut = compute_disk_lut_for_current_camera(disk_cache)
        disk_render_lut = prepare_disk_render_lut(disk_renderer, disk_crossing_lut)
        update_interactive_frame(
            display,
            renderer,
            render_lut,
            disk_renderer,
            disk_render_lut,
            skybox,
            beta_grid,
            final_states,
            disk_crossing_lut,
        )
        capture_mouse()

        running = True
        previous_toggle_disk = False
        last_time = time.perf_counter()
        while running:
            now = time.perf_counter()
            dt = min(now - last_time, 0.05)  # évite un gros saut après un calcul bloquant
            last_time = now

            image_dirty = False
            geodesic_dirty = False
            had_input = False

            for event in display.poll_events():
                if event.type == pygame.QUIT:
                    running = False

            orientation_changed, position_changed, should_quit, toggle_disk = handle_continuous_input(dt)
            mouse_changed = handle_mouse_look()
            if should_quit:
                running = False
            if toggle_disk and not previous_toggle_disk:
                ENABLE_DISK_OVERLAY = not ENABLE_DISK_OVERLAY
                print(f"disk overlay: {'on' if ENABLE_DISK_OVERLAY else 'off'}")
                if ENABLE_DISK_OVERLAY and disk_crossing_lut is None:
                    disk_crossing_lut = compute_disk_lut_for_current_camera(disk_cache)
                    disk_render_lut = prepare_disk_render_lut(
                        disk_renderer,
                        disk_crossing_lut,
                    )
                image_dirty = True
                had_input = True
            previous_toggle_disk = toggle_disk

            if orientation_changed or mouse_changed:
                image_dirty = True
                had_input = True
            if position_changed:
                geodesic_dirty = True
                image_dirty = True
                had_input = True

            if geodesic_dirty:
                current_r = func.camera_radius(CAMERA, BLACKHOLE)
                beta_grid, final_states = lut_cache.get_interpolated(current_r)
                render_lut = renderer.prepare_lut(
                    beta_grid,
                    final_states,
                    BLACKHOLE["MASS"],
                )
                disk_crossing_lut = compute_disk_lut_for_current_camera(disk_cache)
                disk_render_lut = prepare_disk_render_lut(
                    disk_renderer,
                    disk_crossing_lut,
                )

            if image_dirty:
                update_interactive_frame(
                    display,
                    renderer,
                    render_lut,
                    disk_renderer,
                    disk_render_lut,
                    skybox,
                    beta_grid,
                    final_states,
                    disk_crossing_lut,
                )

            # Si rien n'a changé, on ne recalcule pas l'image : on redessine juste
            # la texture déjà en VRAM + le FPS.
            display.draw()
            display.tick(240)

            # Temps mort : prépare une seule LUT voisine. Si ça prend ~1s, ce n'est
            # pas grave : ça arrive seulement quand l'utilisateur ne touche à rien.
            if not had_input and not image_dirty:
                current_r = func.camera_radius(CAMERA, BLACKHOLE)
                lut_cache.prefetch_one_missing_with_margin(
                    current_r,
                    margin=IDLE_PREFETCH_MARGIN,
                )
    finally:
        display.close()


if __name__ == "__main__":
    main()
