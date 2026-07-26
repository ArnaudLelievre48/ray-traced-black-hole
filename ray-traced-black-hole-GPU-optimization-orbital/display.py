import time

import numpy as np
import pygame
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_RGB,
    GL_RGBA,
    GL_SRC_ALPHA,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    GL_QUADS,
    glBegin,
    glBindTexture,
    glBlendFunc,
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


class OpenGLImageDisplay:
    """Affiche une image RGB uint8 dans une texture OpenGL plein écran.

    Le but est de cacher le code OpenGL moche hors de main.py.
    main.py doit seulement appeler :
        display.update_image(frame_rgb)
        display.draw()
    """

    def __init__(self, width, height, title="OpenGL display"):
        self.width = width
        self.height = height

        pygame.init()
        pygame.font.init()
        pygame.display.set_mode((width, height), pygame.OPENGL | pygame.DOUBLEBUF)
        pygame.display.set_caption(title)

        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.image_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.image_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # Texture vide au démarrage. Elle sera remplie par update_image().
        empty = np.zeros((height, width, 3), dtype=np.uint8)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGB,
            width,
            height,
            0,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            empty,
        )

        self.font = pygame.font.SysFont("monospace", 22)
        self.fps_texture = glGenTextures(1)
        self.fps_width = 1
        self.fps_height = 1

        self.clock = pygame.time.Clock()
        self.fps = 0.0
        self.fps_counter = 0
        self.fps_last_update = time.perf_counter()
        self._update_fps_texture(0.0)

    def poll_events(self):
        return pygame.event.get()

    def tick(self, fps_limit=240):
        self.clock.tick(fps_limit)

    def close(self):
        pygame.quit()

    def update_image(self, frame_rgb):
        """Update la texture image. frame_rgb.shape=(H,W,3), dtype=uint8."""
        frame_gl = np.ascontiguousarray(np.flipud(frame_rgb))

        glBindTexture(GL_TEXTURE_2D, self.image_texture)
        glTexSubImage2D(
            GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            frame_gl,
        )

    def draw(self):
        """Dessine l'image courante + FPS."""
        self._update_fps_counter()

        glClear(GL_COLOR_BUFFER_BIT)
        self._draw_fullscreen_texture()
        self._draw_fps_text()
        pygame.display.flip()

    def _update_fps_counter(self):
        self.fps_counter += 1
        now = time.perf_counter()

        if now - self.fps_last_update >= 0.25:
            self.fps = self.fps_counter / (now - self.fps_last_update)
            self.fps_counter = 0
            self.fps_last_update = now
            self._update_fps_texture(self.fps)

    def _update_fps_texture(self, fps):
        surface = self.font.render(f"FPS: {fps:5.1f}", True, (255, 255, 255))
        self.fps_width, self.fps_height = surface.get_size()
        text_rgba = pygame.image.tostring(surface, "RGBA", True)

        glBindTexture(GL_TEXTURE_2D, self.fps_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            self.fps_width,
            self.fps_height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            text_rgba,
        )

    def _draw_fullscreen_texture(self):
        glBindTexture(GL_TEXTURE_2D, self.image_texture)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(-1.0, -1.0)

        glTexCoord2f(1.0, 0.0)
        glVertex2f(1.0, -1.0)

        glTexCoord2f(1.0, 1.0)
        glVertex2f(1.0, 1.0)

        glTexCoord2f(0.0, 1.0)
        glVertex2f(-1.0, 1.0)
        glEnd()

    def _draw_fps_text(self):
        margin_x = 12
        margin_y = 10

        x0 = -1.0 + 2.0 * margin_x / self.width
        y1 = 1.0 - 2.0 * margin_y / self.height
        x1 = x0 + 2.0 * self.fps_width / self.width
        y0 = y1 - 2.0 * self.fps_height / self.height

        glBindTexture(GL_TEXTURE_2D, self.fps_texture)

        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 0.0)
        glVertex2f(x0, y0)

        glTexCoord2f(1.0, 0.0)
        glVertex2f(x1, y0)

        glTexCoord2f(1.0, 1.0)
        glVertex2f(x1, y1)

        glTexCoord2f(0.0, 1.0)
        glVertex2f(x0, y1)
        glEnd()
