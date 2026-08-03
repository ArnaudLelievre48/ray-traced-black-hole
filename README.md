# Ray-traced black hole

A black hole visualization made with ray tracing. The project started in Python and now includes CPU, GPU, optimized GPU, full GPU, and Rust versions.

## Demo

![black hole moving](ray-traced-black-hole-FULL-GPU/video_frames/blackhole_disk.gif)

## Versions

- `ray-traced-black-hole-CPU-multiprocessing`
  - Python CPU version using multiprocessing to compute several light rays at once.
  - Solves the geodesic deviation with `(t, r, theta, phi)`.
- `ray-traced-black-hole-GPU`
  - Python GPU version using CuPy to compute the light rays on the GPU.
  - Solves the geodesic deviation with `(t, r, theta, phi)`.
- `ray-traced-black-hole-GPU-optimization-orbital`
  - Optimized Python GPU version.
  - Solves the geodesic deviation with `(t, r, beta)` in the orbital plane of the light ray.
  - Uses CuPy for computation, with pygame and OpenGL for display.
  - Supports an accretion disk overlay.
- `ray-traced-black-hole-FULL-GPU`
  - Fastest Python version.
  - Uses multiple computation and display optimizations, including GPU-only display.
  - Runs in real time, reaching about 100 FPS on average at 720p.
- `ray-traced-black-hole-RUST`
  - Rust version made as a learning project.
  - Written to be simple and readable.

## Dependencies

Python dependencies are listed in `requirements-raytracing.txt`.

Install them with:

```bash
pip install -r requirements-raytracing.txt
```

The GPU versions need a CUDA-compatible GPU and a working CuPy installation.

The video renderer also needs `ffmpeg` installed.

## Run

Run the full GPU version:

```bash
cd ray-traced-black-hole-FULL-GPU
python main.py
```

Render a video:

```bash
cd ray-traced-black-hole-FULL-GPU
python render_video.py
```

Run the CPU version:

```bash
cd ray-traced-black-hole-CPU-multiprocessing
python ray-traced-black-hole.py
```

## Controls

- `Z`, `Q`, `S`, `D`: move the camera
- `Space` / `Shift`: move up and down
- Arrow keys or mouse: look around
- `M`: toggle the accretion disk
- `Esc`: quit
