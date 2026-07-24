import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# local imports
import func
import assets


# Variables globales lues par les workers multiprocessing.
# Sur Linux, ProcessPoolExecutor utilise fork par défaut : les workers héritent
# de ces objets en lecture, donc on n'a pas besoin de tout repasser à chaque ligne.
MAP = None
SKYBOX = None
BLACKHOLE = None
camera = None
width = None
height = None
center_x = None
center_strip_half_width = None
center_subpixel_samples_x = None


def init_worker(MAP_, SKYBOX_, BLACKHOLE_, camera_, width_, height_, center_x_, center_strip_half_width_, center_subpixel_samples_x_):
    """Initialise les globales dans chaque process worker.

    render_row(pixel_y) reste simple, mais on ne dépend pas du mode exact de
    multiprocessing. Les gros objets sont passés une fois par worker, pas à
    chaque ligne rendue.
    """
    global MAP, SKYBOX, BLACKHOLE, camera
    global width, height, center_x, center_strip_half_width, center_subpixel_samples_x

    MAP = MAP_
    SKYBOX = SKYBOX_
    BLACKHOLE = BLACKHOLE_
    camera = camera_
    width = width_
    height = height_
    center_x = center_x_
    center_strip_half_width = center_strip_half_width_
    center_subpixel_samples_x = center_subpixel_samples_x_


def render_row(pixel_y):
    global MAP, SKYBOX, BLACKHOLE, camera
    global width, height, center_x, center_strip_half_width, center_subpixel_samples_x

    row = np.zeros((width, 3), dtype=np.float32)

    for pixel_x in range(width):
        if abs(pixel_x - center_x) <= center_strip_half_width:
            color = np.zeros(3, dtype=np.float32)
            for subpixel_x in center_subpixel_samples_x:
                ray = assets.LightRay(camera, pixel_x, pixel_y, BLACKHOLE, subpixel_x=subpixel_x)
                color += ray.render_light_ray(MAP, SKYBOX, BLACKHOLE)
            row[pixel_x] = color / len(center_subpixel_samples_x)
        else:
            ray = assets.LightRay(camera, pixel_x, pixel_y, BLACKHOLE)
            row[pixel_x] = ray.render_light_ray(MAP, SKYBOX, BLACKHOLE)
    return pixel_y, row


def generate_img():
    global MAP, SKYBOX, BLACKHOLE, camera
    global width, height, center_x, center_strip_half_width, center_subpixel_samples_x

    image = np.zeros((height, width, 3), dtype=np.float32)

    with ProcessPoolExecutor(
        max_workers=min(12,os.cpu_count()),
        initializer=init_worker,
        initargs=(MAP, SKYBOX, BLACKHOLE, camera, width, height, center_x, center_strip_half_width, center_subpixel_samples_x),
    ) as executor:
        futures = [executor.submit(render_row, pixel_y) for pixel_y in range(height)]

        done = 0
        last_percent = -1
        for future in as_completed(futures):
            pixel_y, row = future.result()
            image[pixel_y] = row

            done += 1
            percent = int(100 * done / height)
            if percent != last_percent:
                print(f"render progress: {percent}%", flush=True)
                last_percent = percent

    return image.copy()




def main():
    global MAP, SKYBOX, BLACKHOLE, camera
    global width, height, center_x, center_strip_half_width, center_subpixel_samples_x

    MAP_size = 400

    # MAP[z, y, x] = [R, G, B, brightness]
    MAP = np.zeros((MAP_size, MAP_size, MAP_size, 4), dtype=np.float32)
    SKYBOX = func.load_skybox("source/skybox.png")
    BLACKHOLE = assets.BlackHole(0,0,0,1)


    # preview camera
    camera = assets.Camera(x=0, y=-75, z=0, angle_vertical=np.pi/2, angle_horizontal=np.pi/2, fov=75, resolution=(600,450))
    width, height = camera.resolution

    # Autour de la colonne centrale, les rayons ont uphi ~ 0 : ils restent presque
    # tous dans le même plan méridien et échantillonnent une bande très fine de la
    # skybox. On anti-aliase seulement cette zone pour garder un rendu preview rapide.
    center_x = width / 2 - 0.5
    center_strip_half_width = max(3, int(0.03 * width))
    center_subpixel_samples_x = (0.2, 0.5, 0.8)


    # video

    # video 1
    camera.set_position(0,-75,0)
    BLACKHOLE.set_position(-75, 0 ,0)

    image = np.zeros((height, width, 3), dtype=np.float32)
    for frame in range(48):
        image = generate_img()
        plt.imshow(image)
        plt.savefig(f"renders1/render_{frame:04d}.png", dpi=150)
        BLACKHOLE.move(150/24,0,0)
        image = np.zeros((height, width, 3), dtype=np.float32)



    quit()

    # video 2

    camera.set_position(0,-100,0)
    BLACKHOLE.set_position(0, 0 ,0)

    image = np.zeros((height, width, 3), dtype=np.float32)
    for frame in range(24):
        image = generate_img()
        plt.imshow(image)
        plt.savefig(f"renders2/render_{frame:04d}.png", dpi=150)
        camera.move(0,(100-4)/24,0,0,0)
        image = np.zeros((height, width, 3), dtype=np.float32)




if __name__ == "__main__":
    main()
