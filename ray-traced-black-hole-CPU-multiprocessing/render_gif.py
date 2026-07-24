#!/usr/bin/env python3
"""Crée un GIF depuis les images d'un répertoire, triées lexicalement.

Usage :
    python render_gif.py renders
    python render_gif.py renders -o blackhole.gif --duration 60
"""

from __future__ import annotations

import argparse
from pathlib import Path
from PIL import Image


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def list_images(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_gif(input_dir: Path, output_path: Path, duration_ms: int, loop: int) -> None:
    image_paths = list_images(input_dir)

    if not image_paths:
        raise ValueError(f"Aucune image trouvée dans {input_dir}")

    frames = []
    expected_size = None

    for path in image_paths:
        frame = Image.open(path).convert("RGB")

        if expected_size is None:
            expected_size = frame.size
        elif frame.size != expected_size:
            raise ValueError(
                f"Toutes les images doivent avoir la même taille. "
                f"{path.name} a la taille {frame.size}, attendu {expected_size}."
            )

        frames.append(frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
        optimize=False,
    )

    print(f"GIF écrit : {output_path}")
    print(f"Images utilisées : {len(frames)}")
    print("Première image :", image_paths[0].name)
    print("Dernière image :", image_paths[-1].name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crée un GIF depuis les images d'un répertoire, triées dans l'ordre lexical."
    )
    parser.add_argument("input_dir", type=Path, help="Répertoire contenant les images")
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Chemin du GIF de sortie. Défaut : <input_dir>/render.gif",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=80,
        help="Durée d'une frame en millisecondes. Défaut : 80",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Nombre de boucles. 0 = boucle infinie. Défaut : 0",
    )

    args = parser.parse_args()
    input_dir = args.input_dir

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Répertoire invalide : {input_dir}")

    output_path = args.output if args.output is not None else input_dir / "render.gif"
    make_gif(input_dir, output_path, args.duration, args.loop)


if __name__ == "__main__":
    main()
