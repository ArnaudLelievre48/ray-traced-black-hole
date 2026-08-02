import ctypes
import ctypes.util
import os
import time

# Sur un portable hybride, le contexte OpenGL doit être créé sur le GPU NVIDIA
# qui exécute CUDA. Ces variables sont l'équivalent de `prime-run` sous Linux et
# doivent être définies avant l'initialisation SDL/GLX. setdefault laisse à
# l'utilisateur la possibilité de choisir explicitement une autre configuration.
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")

import cupy as cp
import pygame
from OpenGL.GL import (
    GL_BLEND,
    GL_COLOR_BUFFER_BIT,
    GL_LINEAR,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_PIXEL_UNPACK_BUFFER,
    GL_QUADS,
    GL_RENDERER,
    GL_RGBA,
    GL_RGBA8,
    GL_SRC_ALPHA,
    GL_STREAM_DRAW,
    GL_TEXTURE_2D,
    GL_TEXTURE_MAG_FILTER,
    GL_TEXTURE_MIN_FILTER,
    GL_UNSIGNED_BYTE,
    glBegin,
    glBindBuffer,
    glBindTexture,
    glBlendFunc,
    glBufferData,
    glClear,
    glDeleteBuffers,
    glDeleteTextures,
    glEnable,
    glEnd,
    glFinish,
    glGenBuffers,
    glGenTextures,
    glGetString,
    glTexCoord2f,
    glTexImage2D,
    glTexParameteri,
    glTexSubImage2D,
    glVertex2f,
)


class CudaOpenGLError(RuntimeError):
    """Erreur d'initialisation ou d'utilisation de l'interop CUDA/OpenGL."""


class _CudaOpenGLInterop:
    """Petit binding ctypes des fonctions d'interop absentes de l'API CuPy."""

    CUDA_GL_DEVICE_LIST_ALL = 1
    CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2

    def __init__(self):
        library_name = ctypes.util.find_library("cudart")
        if library_name is None:
            raise CudaOpenGLError("libcudart est introuvable")

        try:
            self.lib = ctypes.CDLL(library_name)
        except OSError as exc:
            raise CudaOpenGLError(f"impossible de charger {library_name}: {exc}") from exc

        resource_pointer = ctypes.POINTER(ctypes.c_void_p)
        self.lib.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.lib.cudaGetErrorString.restype = ctypes.c_char_p
        self.lib.cudaGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaGetDeviceCount.restype = ctypes.c_int
        self.lib.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        self.lib.cudaGetDevice.restype = ctypes.c_int
        self.lib.cudaSetDevice.argtypes = [ctypes.c_int]
        self.lib.cudaSetDevice.restype = ctypes.c_int
        self.lib.cudaGLGetDevices.argtypes = [
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.c_int,
        ]
        self.lib.cudaGLGetDevices.restype = ctypes.c_int
        self.lib.cudaGraphicsGLRegisterBuffer.argtypes = [
            resource_pointer,
            ctypes.c_uint,
            ctypes.c_uint,
        ]
        self.lib.cudaGraphicsGLRegisterBuffer.restype = ctypes.c_int
        self.lib.cudaGraphicsMapResources.argtypes = [
            ctypes.c_int,
            resource_pointer,
            ctypes.c_void_p,
        ]
        self.lib.cudaGraphicsMapResources.restype = ctypes.c_int
        self.lib.cudaGraphicsResourceGetMappedPointer.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self.lib.cudaGraphicsResourceGetMappedPointer.restype = ctypes.c_int
        self.lib.cudaGraphicsUnmapResources.argtypes = [
            ctypes.c_int,
            resource_pointer,
            ctypes.c_void_p,
        ]
        self.lib.cudaGraphicsUnmapResources.restype = ctypes.c_int
        self.lib.cudaGraphicsUnregisterResource.argtypes = [ctypes.c_void_p]
        self.lib.cudaGraphicsUnregisterResource.restype = ctypes.c_int

    def _check(self, result, operation):
        if result == 0:
            return
        message = self.lib.cudaGetErrorString(result)
        detail = message.decode("utf-8", errors="replace") if message else "erreur inconnue"
        raise CudaOpenGLError(f"{operation}: CUDA {result} ({detail})")

    def select_device_for_current_gl_context(self):
        device_count = ctypes.c_int()
        self._check(self.lib.cudaGetDeviceCount(ctypes.byref(device_count)), "cudaGetDeviceCount")
        if device_count.value <= 0:
            raise CudaOpenGLError("aucun GPU CUDA détecté")

        devices = (ctypes.c_int * device_count.value)()
        gl_device_count = ctypes.c_uint()
        query_result = self.lib.cudaGLGetDevices(
            ctypes.byref(gl_device_count),
            devices,
            device_count.value,
            self.CUDA_GL_DEVICE_LIST_ALL,
        )
        if query_result == 0 and gl_device_count.value > 0:
            device_id = int(devices[0])
        else:
            # Certains couples SDL/GLX + pilotes récents renvoient CUDA 999 sur
            # cudaGLGetDevices alors que l'enregistrement du PBO fonctionne. On
            # tente le périphérique CUDA courant ; register_buffer reste le test
            # définitif et échouera clairement si les GPU sont incompatibles.
            current_device = ctypes.c_int()
            self._check(self.lib.cudaGetDevice(ctypes.byref(current_device)), "cudaGetDevice")
            device_id = int(current_device.value)
            message = self.lib.cudaGetErrorString(query_result)
            detail = message.decode("utf-8", errors="replace") if message else "erreur inconnue"
            print(
                "warning: cudaGLGetDevices indisponible "
                f"(CUDA {query_result}: {detail}), tentative sur CUDA device {device_id}"
            )

        self._check(self.lib.cudaSetDevice(device_id), "cudaSetDevice")
        # Maintient également l'état de périphérique vu par CuPy sur ce thread.
        cp.cuda.Device(device_id).use()
        return device_id

    def register_buffer(self, pbo):
        resource = ctypes.c_void_p()
        self._check(
            self.lib.cudaGraphicsGLRegisterBuffer(
                ctypes.byref(resource),
                int(pbo),
                self.CUDA_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD,
            ),
            "cudaGraphicsGLRegisterBuffer",
        )
        return resource

    def unregister(self, resource):
        self._check(
            self.lib.cudaGraphicsUnregisterResource(resource),
            "cudaGraphicsUnregisterResource",
        )

    @staticmethod
    def _resource_array(resource):
        return (ctypes.c_void_p * 1)(resource.value)

    def map(self, resource, stream_ptr):
        resources = self._resource_array(resource)
        self._check(
            self.lib.cudaGraphicsMapResources(
                1,
                resources,
                ctypes.c_void_p(stream_ptr),
            ),
            "cudaGraphicsMapResources",
        )

        device_pointer = ctypes.c_void_p()
        size = ctypes.c_size_t()
        self._check(
            self.lib.cudaGraphicsResourceGetMappedPointer(
                ctypes.byref(device_pointer),
                ctypes.byref(size),
                resource,
            ),
            "cudaGraphicsResourceGetMappedPointer",
        )
        return int(device_pointer.value), int(size.value)

    def unmap(self, resource, stream_ptr):
        resources = self._resource_array(resource)
        self._check(
            self.lib.cudaGraphicsUnmapResources(
                1,
                resources,
                ctypes.c_void_p(stream_ptr),
            ),
            "cudaGraphicsUnmapResources",
        )


class CudaOpenGLImageDisplay:
    """Affiche directement une image CuPy grâce à deux PBO CUDA/OpenGL partagés.

    Le chemin des pixels est entièrement GPU :
        image float32 CuPy -> kernel CUDA RGBA8 -> PBO OpenGL -> texture OpenGL.

    Les deux PBO alternés évitent que CUDA attende le transfert OpenGL du PBO
    précédent. Seule la petite texture de texte FPS est encore créée côté CPU.
    """

    _FLOAT_RGB_TO_RGBA8 = r"""
    extern "C" __global__
    void float_rgb_to_rgba8_flipped(
        const float* image,
        unsigned char* output,
        const int width,
        const int height
    ) {
        const int pixel = blockDim.x * blockIdx.x + threadIdx.x;
        const int pixel_count = width * height;
        if (pixel >= pixel_count) return;

        const int x = pixel % width;
        const int y = pixel / width;
        const int source = ((height - 1 - y) * width + x) * 3;
        const int destination = pixel * 4;

        float r = image[source];
        float g = image[source + 1];
        float b = image[source + 2];
        r = fminf(fmaxf(r, 0.0f), 1.0f);
        g = fminf(fmaxf(g, 0.0f), 1.0f);
        b = fminf(fmaxf(b, 0.0f), 1.0f);

        output[destination] = (unsigned char)(r * 255.0f);
        output[destination + 1] = (unsigned char)(g * 255.0f);
        output[destination + 2] = (unsigned char)(b * 255.0f);
        output[destination + 3] = 255;
    }
    """

    def __init__(self, width, height, title="CUDA/OpenGL display"):
        self.width = int(width)
        self.height = int(height)
        self._buffer_size = self.width * self.height * 4
        self._closed = False
        self._interop = None
        self._cuda_resources = []
        self._pbos = []
        self._next_pbo = 0
        self.image_texture = None
        self.fps_texture = None

        pygame.init()
        pygame.font.init()
        try:
            pygame.display.set_mode(
                (self.width, self.height),
                pygame.OPENGL | pygame.DOUBLEBUF,
            )
            pygame.display.set_caption(title)

            glEnable(GL_TEXTURE_2D)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            self.image_texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.image_texture)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            # Allocation VRAM uniquement : aucune image noire NumPy n'est envoyée.
            glTexImage2D(
                GL_TEXTURE_2D,
                0,
                GL_RGBA8,
                self.width,
                self.height,
                0,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                None,
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

            self._interop = _CudaOpenGLInterop()
            self.cuda_device_id = self._interop.select_device_for_current_gl_context()
            self._create_interop_pbos()
            self._convert_kernel = cp.RawKernel(
                self._FLOAT_RGB_TO_RGBA8,
                "float_rgb_to_rgba8_flipped",
            )

            renderer = glGetString(GL_RENDERER)
            renderer_name = renderer.decode("utf-8", errors="replace") if renderer else "inconnu"
            print(f"CUDA/OpenGL interop actif: CUDA device {self.cuda_device_id}, {renderer_name}")
        except Exception:
            self.close()
            raise

    def _create_interop_pbos(self):
        generated = glGenBuffers(2)
        self._pbos = [int(pbo) for pbo in generated]
        try:
            for pbo in self._pbos:
                glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
                glBufferData(
                    GL_PIXEL_UNPACK_BUFFER,
                    self._buffer_size,
                    None,
                    GL_STREAM_DRAW,
                )
                self._cuda_resources.append(self._interop.register_buffer(pbo))
        finally:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)

    def poll_events(self):
        return pygame.event.get()

    def tick(self, fps_limit=240):
        self.clock.tick(fps_limit)

    def close(self):
        if self._closed:
            return
        self._closed = True

        # Le contexte doit rester vivant jusqu'à la libération des ressources.
        try:
            if pygame.display.get_surface() is not None:
                glFinish()
            if self._interop is not None:
                for resource in self._cuda_resources:
                    try:
                        self._interop.unregister(resource)
                    except CudaOpenGLError as exc:
                        print(f"warning: {exc}")
            self._cuda_resources.clear()

            if self._pbos and pygame.display.get_surface() is not None:
                glDeleteBuffers(len(self._pbos), self._pbos)
            self._pbos.clear()

            textures = [texture for texture in (self.image_texture, self.fps_texture) if texture]
            if textures and pygame.display.get_surface() is not None:
                glDeleteTextures(textures)
        finally:
            pygame.quit()

    def update_image(self, image_gpu):
        """Convertit et publie une image CuPy (H, W, 3) sans copie vers le CPU."""
        if not isinstance(image_gpu, cp.ndarray):
            raise TypeError("update_image attend un cupy.ndarray situé sur le GPU")
        if image_gpu.shape != (self.height, self.width, 3):
            raise ValueError(
                f"forme attendue {(self.height, self.width, 3)}, reçue {image_gpu.shape}"
            )
        if image_gpu.dtype != cp.float32:
            image_gpu = image_gpu.astype(cp.float32, copy=False)
        if not image_gpu.flags.c_contiguous:
            image_gpu = cp.ascontiguousarray(image_gpu)

        def convert_to_rgba(output, stream):
            threads = 256
            pixels = self.width * self.height
            self._convert_kernel(
                ((pixels + threads - 1) // threads,),
                (threads,),
                (image_gpu, output, self.width, self.height),
                stream=stream,
            )

        self.update_from_cuda(convert_to_rgba)

    def update_from_cuda(self, producer):
        """Fait écrire ``producer(output_rgba8, stream)`` directement dans le PBO.

        Cette variante supprime même le buffer RGB intermédiaire : le kernel de
        rendu peut produire les pixels finaux dans la mémoire partagée OpenGL.
        """
        index = self._next_pbo
        resource = self._cuda_resources[index]
        stream = cp.cuda.get_current_stream()
        mapped = False
        try:
            pointer, mapped_size = self._interop.map(resource, stream.ptr)
            mapped = True
            if mapped_size < self._buffer_size:
                raise CudaOpenGLError(
                    f"PBO trop petit: {mapped_size} octets, {self._buffer_size} requis"
                )

            memory = cp.cuda.UnownedMemory(pointer, mapped_size, self)
            output = cp.ndarray(
                (self.height, self.width, 4),
                dtype=cp.uint8,
                memptr=cp.cuda.MemoryPointer(memory, 0),
            )
            producer(output, stream)
        finally:
            if mapped:
                self._interop.unmap(resource, stream.ptr)

        # Avec un PBO lié, None signifie offset 0 dans le buffer GPU partagé.
        glBindTexture(GL_TEXTURE_2D, self.image_texture)
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, self._pbos[index])
        try:
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                self.width,
                self.height,
                GL_RGBA,
                GL_UNSIGNED_BYTE,
                ctypes.c_void_p(0),
            )
        finally:
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
        self._next_pbo = (index + 1) % len(self._pbos)

    def draw(self):
        """Dessine la texture GPU courante et le compteur FPS."""
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
        text_rgba = pygame.image.tobytes(surface, "RGBA", True)

        # Important : un PBO lié transformerait text_rgba en offset GPU.
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)
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


# Nom historique conservé pour les imports externes ; l'implémentation est full GPU.
OpenGLImageDisplay = CudaOpenGLImageDisplay
