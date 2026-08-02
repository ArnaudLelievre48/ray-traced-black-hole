"""Overlay du disque d'accrétion fusionné en un kernel CUDA par frame."""

from dataclasses import dataclass

import cupy as cp
import numpy as np

from gpu_renderer import OrbitalGpuRenderer


_DISK_OVERLAY_KERNEL = r"""
#include <cuda_runtime.h>

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

__device__ __forceinline__ float positive_mod_pi(float value) {
    const float pi = 3.14159265358979323846f;
    value = fmodf(value, pi);
    return value < 0.0f ? value + pi : value;
}

__device__ __forceinline__ float interpolate_sample(
    const float first,
    const float second,
    const float fraction
) {
    if (fraction <= 1.0e-7f) return first;
    if (fraction >= 0.9999999f) return second;
    return first + fraction * (second - first);
}

__device__ __forceinline__ float3 blackbody_disk_emission(
    const float radius,
    const float mass,
    const float inner_radius,
    const float outer_radius,
    const float closeness,
    const float emission
) {
    if (radius < inner_radius || radius > outer_radius) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }

    const float safe_radius = fmaxf(radius, 1.0e-6f);
    float flux = 1.0f / (safe_radius * safe_radius * safe_radius);
    flux *= fmaxf(1.0f - sqrtf(inner_radius / safe_radius), 0.0f);

    const float fade_width = 0.25f * fmaxf(outer_radius - inner_radius, 1.0e-6f);
    const float outer_x = fminf(fmaxf((outer_radius - safe_radius) / fade_width, 0.0f), 1.0f);
    flux *= outer_x * outer_x * (3.0f - 2.0f * outer_x);

    const float peak_radius = (49.0f / 36.0f) * inner_radius;
    float peak_flux = 1.0f / (peak_radius * peak_radius * peak_radius);
    peak_flux *= fmaxf(1.0f - sqrtf(inner_radius / peak_radius), 1.0e-12f);
    const float normalized_flux = fminf(fmaxf(flux / (peak_flux + 1.0e-30f), 0.0f), 1.0f);

    const float temperature_factor = powf(normalized_flux, 0.25f);
    const float redshift = sqrtf(fminf(fmaxf(1.0f - 2.0f * mass / safe_radius, 0.05f), 1.0f));
    float temperature = (1300.0f + (7200.0f - 1300.0f) * temperature_factor) * redshift;
    temperature = fminf(fmaxf(temperature, 1000.0f), 40000.0f) / 100.0f;

    float red;
    float green;
    float blue;
    if (temperature <= 66.0f) {
        red = 255.0f;
        green = 99.4708025861f * logf(temperature) - 161.1195681661f;
    } else {
        red = 329.698727446f * powf(temperature - 60.0f, -0.1332047592f);
        green = 288.1221695283f * powf(temperature - 60.0f, -0.0755148492f);
    }
    if (temperature >= 66.0f) {
        blue = 255.0f;
    } else if (temperature <= 19.0f) {
        blue = 0.0f;
    } else {
        blue = 138.5177312231f * logf(temperature - 10.0f) - 305.0447927307f;
    }

    red = fminf(fmaxf(red / 255.0f * 1.10f, 0.0f), 1.0f);
    green = fminf(fmaxf(green / 255.0f * 0.92f, 0.0f), 1.0f);
    blue = fminf(fmaxf(blue / 255.0f * 0.72f, 0.0f), 1.0f);
    const float intensity = emission * closeness * 1.65f * normalized_flux;
    return make_float3(intensity * red, intensity * green, intensity * blue);
}

extern "C" __global__
void render_disk_overlay(
    float* __restrict__ output,
    const float* __restrict__ beta_samples,
    const float* __restrict__ radius_samples,
    const int ray_count,
    const int sample_count,
    const int width,
    const int height,
    const float mass,
    const float camera_radius,
    const float inner_radius,
    const float outer_radius,
    const float beta_tolerance,
    const float max_beta,
    const float emission,
    const float minimum_impact,
    const float maximum_impact,
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
    const float3 forward = make_float3(forward_x, forward_y, forward_z);
    const float3 right = make_float3(right_x, right_y, right_z);
    const float3 up = make_float3(up_x, up_y, up_z);
    const float3 center = make_float3(center_x, center_y, center_z);

    const float offset_x = ((float)x + 0.5f - 0.5f * (float)width) / (float)width;
    const float offset_y = ((float)y + 0.5f - 0.5f * (float)height) / (float)height;
    float3 direction = make_float3(
        screen_distance * forward.x + offset_x * screen_width * right.x - offset_y * screen_height * up.x,
        screen_distance * forward.y + offset_x * screen_width * right.y - offset_y * screen_height * up.y,
        screen_distance * forward.z + offset_x * screen_width * right.z - offset_y * screen_height * up.z
    );
    direction = normalize3(direction);

    const float cos_beta = fminf(fmaxf(
        direction.x * center.x + direction.y * center.y + direction.z * center.z,
        -1.0f
    ), 1.0f);
    const float beta = acosf(cos_beta);
    const float sin_beta = sqrtf(fmaxf(1.0f - cos_beta * cos_beta, 0.0f));

    float3 side;
    if (sin_beta > 1.0e-6f) {
        const float inverse_sin = 1.0f / sin_beta;
        side = normalize3(make_float3(
            (direction.x - cos_beta * center.x) * inverse_sin,
            (direction.y - cos_beta * center.y) * inverse_sin,
            (direction.z - cos_beta * center.z) * inverse_sin
        ));
    } else {
        side = make_float3(fallback_x, fallback_y, fallback_z);
    }

    const float metric = 1.0f - 2.0f * mass / camera_radius;
    const float impact = camera_radius * sin_beta / sqrtf(metric);
    const bool candidate = impact >= minimum_impact && impact <= maximum_impact;
    float3 best = make_float3(0.0f, 0.0f, 0.0f);
    if (candidate) {
        const float beta_disk = positive_mod_pi(atan2f(center.z, side.z));
        const float lut_position = beta * (float)(ray_count - 1) * 0.3183098861837907f;
        const int index0 = min((int)lut_position, ray_count - 1);
        const int index1 = min(index0 + 1, ray_count - 1);
        const float fraction = lut_position - (float)index0;
        const int row0 = index0 * sample_count;
        const int row1 = index1 * sample_count;
        const float half_pi = 1.57079632679489661923f;

        for (int sample = 0; sample < sample_count; ++sample) {
            const float path_beta = interpolate_sample(
                beta_samples[row0 + sample],
                beta_samples[row1 + sample],
                fraction
            );
            const float path_radius = interpolate_sample(
                radius_samples[row0 + sample],
                radius_samples[row1 + sample],
                fraction
            );
            if (!isfinite(path_beta) || !isfinite(path_radius)
                    || path_beta < 0.0f || path_beta > max_beta) {
                continue;
            }
            const float difference = fabsf(
                positive_mod_pi(path_beta - beta_disk + half_pi) - half_pi
            );
            if (difference >= beta_tolerance) continue;

            const float closeness = fminf(fmaxf(
                1.0f - difference / beta_tolerance,
                0.0f
            ), 1.0f);
            const float3 value = blackbody_disk_emission(
                path_radius,
                mass,
                inner_radius,
                outer_radius,
                closeness,
                emission
            );
            best.x = fmaxf(best.x, value.x);
            best.y = fmaxf(best.y, value.y);
            best.z = fmaxf(best.z, value.z);
        }
    }

    const int destination = pixel * 3;
    output[destination] = best.x;
    output[destination + 1] = best.y;
    output[destination + 2] = best.z;
}
"""


@dataclass
class PreparedDiskLut:
    beta_samples: cp.ndarray
    radius_samples: cp.ndarray

    @property
    def ray_count(self):
        return self.beta_samples.shape[0]

    @property
    def sample_count(self):
        return self.beta_samples.shape[1]


class DiskGpuRenderer:
    """Version persistante de l'overlay : aucun cp.interp ni temporaire par sample."""

    def __init__(self):
        self._kernel = cp.RawKernel(
            _DISK_OVERLAY_KERNEL,
            "render_disk_overlay",
            options=("--use_fast_math", "--std=c++11"),
        )
        self._output = None
        self._prepared_luts = {}

    def prepare_lut(self, beta_grid, beta_samples, radius_samples):
        beta_grid = cp.asarray(beta_grid)
        beta_samples = cp.asarray(beta_samples)
        radius_samples = cp.asarray(radius_samples)
        if beta_grid.ndim != 1 or beta_grid.size < 2:
            raise ValueError("beta_grid du disque invalide")
        expected = (beta_grid.size, beta_samples.shape[1])
        if beta_samples.ndim != 2 or radius_samples.shape != expected:
            raise ValueError("dimensions des samples du disque invalides")

        cache_key = (
            int(beta_samples.data.ptr),
            int(radius_samples.data.ptr),
            beta_samples.shape,
        )
        cached = self._prepared_luts.get(cache_key)
        if cached is not None:
            return cached

        # Les colonnes finales entièrement NaN ne transportent aucune donnée.
        # On les retire une fois lors d'un changement de position.
        used_columns = cp.any(cp.isfinite(beta_samples) & cp.isfinite(radius_samples), axis=0)
        indices = cp.flatnonzero(used_columns)
        sample_count = int(indices[-1].get()) + 1 if indices.size else 1
        prepared = PreparedDiskLut(
            cp.ascontiguousarray(beta_samples[:, :sample_count], dtype=cp.float32),
            cp.ascontiguousarray(radius_samples[:, :sample_count], dtype=cp.float32),
        )
        self._prepared_luts[cache_key] = prepared
        return prepared

    def render(
        self,
        camera,
        blackhole,
        disk_lut,
        beta_tolerance=0.1,
        max_beta_turns=8,
        emission=1.25,
        b_min_factor=0.0,
        b_max_factor=40.0,
        stream=None,
    ):
        width, height, camera_arguments = OrbitalGpuRenderer._camera_arguments(
            camera,
            blackhole,
        )
        shape = (height, width, 3)
        if self._output is None or self._output.shape != shape:
            self._output = cp.empty(shape, dtype=cp.float32)

        mass = float(blackhole["MASS"])
        factor = 1.7
        angular_momentum = factor * 2.0 * np.sqrt(3.0) * mass
        discriminant = np.sqrt(1.0 - 12.0 * mass * mass / (angular_momentum**2))
        inner_radius = (angular_momentum**2 / (2.0 * mass)) * (1.0 - discriminant)
        outer_radius = (angular_momentum**2 / (2.0 * mass)) * (1.0 + discriminant)
        inner_radius = max(inner_radius, 6.0 * mass)

        camera_position = np.asarray(
            [camera["x"], camera["y"], camera["z"]],
            dtype=np.float64,
        )
        blackhole_position = np.asarray(
            [blackhole["x"], blackhole["y"], blackhole["z"]],
            dtype=np.float64,
        )
        camera_radius = np.linalg.norm(camera_position - blackhole_position)

        threads = 256
        pixels = width * height
        self._kernel(
            ((pixels + threads - 1) // threads,),
            (threads,),
            (
                self._output,
                disk_lut.beta_samples,
                disk_lut.radius_samples,
                np.int32(disk_lut.ray_count),
                np.int32(disk_lut.sample_count),
                np.int32(width),
                np.int32(height),
                np.float32(mass),
                np.float32(camera_radius),
                np.float32(inner_radius),
                np.float32(outer_radius),
                np.float32(beta_tolerance),
                np.float32(max_beta_turns * 2.0 * np.pi),
                np.float32(emission),
                np.float32(b_min_factor * mass),
                np.float32(b_max_factor * mass),
                *camera_arguments,
            ),
            stream=stream,
        )
        return self._output
