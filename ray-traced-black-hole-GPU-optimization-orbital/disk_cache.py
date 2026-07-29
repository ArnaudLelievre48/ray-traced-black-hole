import math

import disk_lut


class DiskLUTCache:
    """Cache des LUT disque en rayon caméra.

    Même idée que OrbitalLUTCache, mais pour la LUT auxiliaire du disque :
        r_camera -> (beta_grid, beta_samples, r_samples)

    Pour l'instant on fait du nearest-neighbor sur une grille de r. Interpoler
    les samples disque entre deux rayons caméra est possible, mais moins propre
    que pour la LUT de déviation car les samples contiennent des branches/NaN.
    """

    def __init__(
        self,
        M,
        rays_number,
        max_steps,
        dr=1.0,
        min_r=None,
    ):
        self.M = M
        self.rays_number = rays_number
        self.max_steps = max_steps
        self.dr = float(dr)
        self.min_r = 2.05 * M if min_r is None else float(min_r)
        self.cache = {}

    def key(self, r):
        r_key = math.floor(r / self.dr + 0.5) * self.dr
        return max(self.min_r, float(r_key))

    def has(self, r):
        return self.key(r) in self.cache

    def get(self, r):
        r_key = self.key(r)
        if r_key not in self.cache:
            self.cache[r_key] = disk_lut.compute_disk_crossing_lut(
                r_key,
                self.M,
                rays_number=self.rays_number,
                max_steps=self.max_steps,
            )
        return self.cache[r_key]

    def keys(self):
        return sorted(self.cache.keys())
