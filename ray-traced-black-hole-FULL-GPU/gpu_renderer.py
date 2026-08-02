"""Renderer orbital fusionné, optimisé pour le chemin interactif CUDA/OpenGL."""

import cupy as cp
import numpy as np
from cupy.cuda import runtime, texture


_ORBITAL_RENDER_KERNEL = r"""
#include <cuda_runtime.h>

__device__ __forceinline__ float clamp_unit(float value) {
    return fminf(fmaxf(value, -1.0f), 1.0f);
}

__device__ __forceinline__ float3 normalize3(float3 value) {
    const float norm2 = value.x * value.x + value.y * value.y + value.z * value.z;
    if (norm2 <= 1.0e-20f) return make_float3(0.0f, 0.0f, 0.0f);
    const float inverse_norm = rsqrtf(norm2);
    return make_float3(
        value.x * inverse_norm,
        value.y * inverse_norm,
        value.z * inverse_norm
    );
}

__device__ __forceinline__ float3 add3(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __forceinline__ float3 scale3(float value, float3 vector) {
    return make_float3(value * vector.x, value * vector.y, value * vector.z);
}

__device__ __forceinline__ float dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __forceinline__ float3 render_pixel(
    const int x,
    const int y,
    const int width,
    const int height,
    const float* __restrict__ lut,
    const int lut_size,
    const cudaTextureObject_t skybox,
    const float skybox_width,
    const float skybox_height,
    const float screen_width,
    const float screen_height,
    const float screen_distance,
    const float3 forward,
    const float3 right,
    const float3 up,
    const float3 center,
    const float3 fallback
) {
    const float offset_x = ((float)x + 0.5f - 0.5f * (float)width) / (float)width;
    const float offset_y = ((float)y + 0.5f - 0.5f * (float)height) / (float)height;
    float3 direction = add3(
        add3(
            scale3(screen_distance, forward),
            scale3(offset_x * screen_width, right)
        ),
        scale3(-offset_y * screen_height, up)
    );
    direction = normalize3(direction);

    const float cos_beta = clamp_unit(dot3(direction, center));
    const float beta = acosf(cos_beta);
    const float sin_beta = sqrtf(fmaxf(1.0f - cos_beta * cos_beta, 0.0f));

    float3 side;
    if (sin_beta > 1.0e-6f) {
        const float inverse_sin_beta = 1.0f / sin_beta;
        side = make_float3(
            (direction.x - cos_beta * center.x) * inverse_sin_beta,
            (direction.y - cos_beta * center.y) * inverse_sin_beta,
            (direction.z - cos_beta * center.z) * inverse_sin_beta
        );
        side = normalize3(side);
    } else {
        side = fallback;
    }

    // beta_grid vaut linspace(0, pi, N) : index direct au lieu de cinq
    // recherches binaires cp.interp indépendantes par pixel.
    const float lut_position = beta * (float)(lut_size - 1) * 0.3183098861837907f;
    const int index0 = min((int)lut_position, lut_size - 1);
    const int index1 = min(index0 + 1, lut_size - 1);
    const float fraction = lut_position - (float)index0;
    const float inverse_fraction = 1.0f - fraction;

    const float* state0 = lut + index0 * 5;
    const float* state1 = lut + index1 * 5;
    const float delta = inverse_fraction * state0[0] + fraction * state1[0];
    const float radius = inverse_fraction * state0[1] + fraction * state1[1];
    const float radial_velocity = inverse_fraction * state0[2] + fraction * state1[2];
    const float angular_velocity = inverse_fraction * state0[3] + fraction * state1[3];
    const float captured = inverse_fraction * state0[4] + fraction * state1[4];
    if (captured > 0.5f) return make_float3(0.0f, 0.0f, 0.0f);

    float sin_delta;
    float cos_delta;
    sincosf(delta, &sin_delta, &cos_delta);

    // e_r0 = -center. On développe directement les deux bases finales pour
    // limiter registres, instructions et écritures temporaires.
    const float3 radial_basis = make_float3(
        -cos_delta * center.x + sin_delta * side.x,
        -cos_delta * center.y + sin_delta * side.y,
        -cos_delta * center.z + sin_delta * side.z
    );
    const float3 angular_basis = make_float3(
        sin_delta * center.x + cos_delta * side.x,
        sin_delta * center.y + cos_delta * side.y,
        sin_delta * center.z + cos_delta * side.z
    );
    const float tangential_velocity = radius * angular_velocity;
    float3 final_direction = add3(
        scale3(radial_velocity, radial_basis),
        scale3(tangential_velocity, angular_basis)
    );
    final_direction = normalize3(final_direction);

    const float phi = atan2f(final_direction.y, final_direction.x);
    const float theta = acosf(clamp_unit(final_direction.z));
    float u = (phi + 3.14159265358979323846f) * 0.15915494309189535f;
    if (u >= 1.0f) u -= 1.0f;
    if (u < 0.0f) u += 1.0f;
    const float v = theta * 0.3183098861837907f;

    // Le +0.5 reproduit les centres de texels du sampler bilinéaire manuel.
    const float texture_u = u + 0.5f / skybox_width;
    const float texture_v = (v * (skybox_height - 1.0f) + 0.5f) / skybox_height;
    const float4 color = tex2D<float4>(skybox, texture_u, texture_v);
    return make_float3(color.x, color.y, color.z);
}

extern "C" __global__
void render_orbital_rgb(
    float* __restrict__ output,
    const float* __restrict__ lut,
    const int lut_size,
    const cudaTextureObject_t skybox,
    const int skybox_width,
    const int skybox_height,
    const int width,
    const int height,
    const float screen_width,
    const float screen_height,
    const float screen_distance,
    const float forward_x,
    const float forward_y,
    const float forward_z,
    const float right_x,
    const float right_y,
    const float right_z,
    const float up_x,
    const float up_y,
    const float up_z,
    const float center_x,
    const float center_y,
    const float center_z,
    const float fallback_x,
    const float fallback_y,
    const float fallback_z
) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= width * height) return;
    const int x = pixel % width;
    const int y = pixel / width;
    const float3 color = render_pixel(
        x, y, width, height, lut, lut_size, skybox,
        (float)skybox_width, (float)skybox_height,
        screen_width, screen_height, screen_distance,
        make_float3(forward_x, forward_y, forward_z),
        make_float3(right_x, right_y, right_z),
        make_float3(up_x, up_y, up_z),
        make_float3(center_x, center_y, center_z),
        make_float3(fallback_x, fallback_y, fallback_z)
    );
    const int destination = pixel * 3;
    output[destination] = color.x;
    output[destination + 1] = color.y;
    output[destination + 2] = color.z;
}

extern "C" __global__
void render_orbital_rgba8(
    uchar4* __restrict__ output,
    const float* __restrict__ lut,
    const int lut_size,
    const cudaTextureObject_t skybox,
    const int skybox_width,
    const int skybox_height,
    const int width,
    const int height,
    const float screen_width,
    const float screen_height,
    const float screen_distance,
    const float forward_x,
    const float forward_y,
    const float forward_z,
    const float right_x,
    const float right_y,
    const float right_z,
    const float up_x,
    const float up_y,
    const float up_z,
    const float center_x,
    const float center_y,
    const float center_z,
    const float fallback_x,
    const float fallback_y,
    const float fallback_z
) {
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= width * height) return;
    const int x = pixel % width;
    const int y = pixel / width;
    const float3 color = render_pixel(
        x, y, width, height, lut, lut_size, skybox,
        (float)skybox_width, (float)skybox_height,
        screen_width, screen_height, screen_distance,
        make_float3(forward_x, forward_y, forward_z),
        make_float3(right_x, right_y, right_z),
        make_float3(up_x, up_y, up_z),
        make_float3(center_x, center_y, center_z),
        make_float3(fallback_x, fallback_y, fallback_z)
    );
    const int destination = (height - 1 - y) * width + x;
    output[destination] = make_uchar4(
        (unsigned char)(fminf(fmaxf(color.x, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(color.y, 0.0f), 1.0f) * 255.0f),
        (unsigned char)(fminf(fmaxf(color.z, 0.0f), 1.0f) * 255.0f),
        255
    );
}
"""


class _SkyboxTexture:
    """Copie RGBA alignée de la skybox dans une texture CUDA bilinéaire."""

    def __init__(self, skybox):
        if skybox.dtype != cp.float32 or skybox.ndim != 3 or skybox.shape[2] != 3:
            raise ValueError("la skybox doit être un tableau CuPy float32 (H, W, 3)")

        self.height, self.width, _ = skybox.shape
        rgba = cp.empty((self.height, self.width, 4), dtype=cp.float32)
        rgba[:, :, :3] = skybox
        rgba[:, :, 3] = 1.0

        channel = texture.ChannelFormatDescriptor(
            32,
            32,
            32,
            32,
            runtime.cudaChannelFormatKindFloat,
        )
        self.array = texture.CUDAarray(channel, self.width, self.height)
        self.array.copy_from(rgba.reshape(self.height, self.width * 4))
        self.resource = texture.ResourceDescriptor(
            runtime.cudaResourceTypeArray,
            cuArr=self.array,
        )
        descriptor = texture.TextureDescriptor(
            addressModes=(runtime.cudaAddressModeWrap, runtime.cudaAddressModeClamp),
            filterMode=runtime.cudaFilterModeLinear,
            readMode=runtime.cudaReadModeElementType,
            normalizedCoords=1,
        )
        self.object = texture.TextureObject(self.resource, descriptor)


class OrbitalGpuRenderer:
    """Renderer persistant : texture, kernels et buffer RGB sont réutilisés."""

    def __init__(self, skybox):
        self.source_skybox = skybox
        self.skybox = _SkyboxTexture(skybox)
        options = ("--use_fast_math", "--std=c++11")
        self._rgb_kernel = cp.RawKernel(
            _ORBITAL_RENDER_KERNEL,
            "render_orbital_rgb",
            options=options,
        )
        self._rgba_kernel = cp.RawKernel(
            _ORBITAL_RENDER_KERNEL,
            "render_orbital_rgba8",
            options=options,
        )
        self._rgb_output = None

    @staticmethod
    def prepare_lut(beta_grid, final_states, mass):
        """Compacte les colonnes utiles de la LUT en float32, une fois par position."""
        beta_grid = cp.asarray(beta_grid)
        final_states = cp.asarray(final_states)
        if beta_grid.ndim != 1 or final_states.shape != (beta_grid.size, 6):
            raise ValueError("dimensions de LUT orbitale invalides")
        if beta_grid.size < 2:
            raise ValueError("la LUT orbitale doit contenir au moins deux rayons")

        packed = cp.empty((beta_grid.size, 5), dtype=cp.float32)
        packed[:, 0] = final_states[:, 2] - beta_grid
        packed[:, 1] = final_states[:, 1]
        packed[:, 2] = final_states[:, 4]
        packed[:, 3] = final_states[:, 5]
        packed[:, 4] = final_states[:, 1] <= 2.0 * mass
        return packed

    @staticmethod
    def _camera_arguments(camera, blackhole):
        width = int(camera["width"])
        height = int(camera["height"])
        aspect_ratio = width / height

        angle_vertical = float(camera["angle_vertical"])
        angle_horizontal = float(camera["angle_horizontal"])
        sin_vertical = np.sin(angle_vertical)
        cos_vertical = np.cos(angle_vertical)
        sin_horizontal = np.sin(angle_horizontal)
        cos_horizontal = np.cos(angle_horizontal)

        forward = np.asarray(
            [
                sin_vertical * cos_horizontal,
                sin_vertical * sin_horizontal,
                cos_vertical,
            ],
            dtype=np.float32,
        )
        forward /= np.linalg.norm(forward)
        right = np.asarray([-sin_horizontal, cos_horizontal, 0.0], dtype=np.float32)
        up = np.asarray(
            [
                -cos_vertical * cos_horizontal,
                -cos_vertical * sin_horizontal,
                sin_vertical,
            ],
            dtype=np.float32,
        )

        camera_position = np.asarray(
            [camera["x"], camera["y"], camera["z"]],
            dtype=np.float64,
        )
        blackhole_position = np.asarray(
            [blackhole["x"], blackhole["y"], blackhole["z"]],
            dtype=np.float64,
        )
        center = blackhole_position - camera_position
        center /= np.linalg.norm(center)
        center = center.astype(np.float32)

        fallback = np.cross(center, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))
        if np.linalg.norm(fallback) < 1.0e-6:
            fallback = np.cross(center, np.asarray([0.0, 1.0, 0.0], dtype=np.float32))
        fallback /= np.linalg.norm(fallback)

        screen_width = float(camera.get("camera_virtual_screen_width", 1.0))
        screen_height = screen_width / aspect_ratio
        screen_distance = float(
            camera.get(
                "distance_from_virtual_screen",
                screen_width / (2.0 * np.tan(float(camera["FOV"]) / 2.0)),
            )
        )
        floats = (
            screen_width,
            screen_height,
            screen_distance,
            *forward,
            *right,
            *up,
            *center,
            *fallback,
        )
        return width, height, tuple(np.float32(value) for value in floats)

    def _launch(self, kernel, output, camera, blackhole, render_lut, stream=None):
        if render_lut.dtype != cp.float32 or not render_lut.flags.c_contiguous:
            render_lut = cp.ascontiguousarray(render_lut, dtype=cp.float32)
        if render_lut.ndim != 2 or render_lut.shape[1] != 5:
            raise ValueError("render_lut doit avoir la forme (N, 5)")

        width, height, camera_arguments = self._camera_arguments(camera, blackhole)
        if output.shape[:2] != (height, width):
            raise ValueError(f"sortie incompatible avec la résolution {width}x{height}")
        threads = 256
        pixels = width * height
        kernel(
            ((pixels + threads - 1) // threads,),
            (threads,),
            (
                output,
                render_lut,
                np.int32(render_lut.shape[0]),
                self.skybox.object,
                np.int32(self.skybox.width),
                np.int32(self.skybox.height),
                np.int32(width),
                np.int32(height),
                *camera_arguments,
            ),
            stream=stream,
        )
        return output

    def render_rgb(self, camera, blackhole, render_lut, output=None, stream=None):
        """Produit un RGB float32 GPU, notamment pour disque et export."""
        shape = (int(camera["height"]), int(camera["width"]), 3)
        if output is None:
            if self._rgb_output is None or self._rgb_output.shape != shape:
                self._rgb_output = cp.empty(shape, dtype=cp.float32)
            output = self._rgb_output
        if output.dtype != cp.float32 or output.shape != shape:
            raise ValueError(f"la sortie RGB doit être un float32 de forme {shape}")
        return self._launch(
            self._rgb_kernel,
            output,
            camera,
            blackhole,
            render_lut,
            stream,
        )

    def render_rgba8(self, camera, blackhole, render_lut, output, stream=None):
        """Écrit directement le rendu final dans le PBO CUDA/OpenGL partagé."""
        shape = (int(camera["height"]), int(camera["width"]), 4)
        if output.dtype != cp.uint8 or output.shape != shape:
            raise ValueError(f"la sortie RGBA doit être un uint8 de forme {shape}")
        return self._launch(
            self._rgba_kernel,
            output,
            camera,
            blackhole,
            render_lut,
            stream,
        )
