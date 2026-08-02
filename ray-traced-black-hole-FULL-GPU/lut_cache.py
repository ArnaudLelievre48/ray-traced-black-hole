import time
import math

import cupy as cp

import GR


class OrbitalLUTCache:
    """Cache de LUT GR en distance caméra-trou noir.

    On garde main.py simple :
        lut_cache = OrbitalLUTCache(...)
        beta_grid, final_states = lut_cache.get_interpolated(r)

    Stratégie :
        - r est quantifié sur une grille régulière de pas dr ;
        - au démarrage on peut pré-calculer plusieurs voisins ;
        - pour un r quelconque, on interpole entre les deux LUTs voisines ;
        - si une LUT manque, on la calcule à la demande ;
        - pendant les temps morts, main.py peut appeler prefetch_around(...).
    """

    def __init__(
        self,
        M,
        rays_number,
        max_steps,
        dr=2.0,
        h_step=0.5,
        h_update_interval=32,
        min_r_margin=2.2,
        verbose=False,
    ):
        self.M = M
        self.rays_number = rays_number
        self.max_steps = max_steps
        self.dr = dr
        self.h_step = h_step
        self.h_update_interval = h_update_interval
        self.min_r = min_r_margin * M
        self.verbose = verbose

        self.beta_grid = cp.linspace(0.0, cp.pi, rays_number, dtype=cp.float64)
        self.cache = {}

    def key(self, r):
        # Pas le round Python natif : round(32.5) -> 32 à cause du bankers rounding.
        # Ici on veut vraiment la grille la plus proche.
        r_key = math.floor(r / self.dr + 0.5) * self.dr
        return max(self.min_r, float(r_key))

    def radius_count_from_margin(self, margin):
        return int(math.ceil(margin / self.dr))

    def bracket_keys(self, r):
        r0 = max(self.min_r, self.dr * int(r // self.dr))
        r1 = r0 + self.dr

        # Si r tombe pile sur la grille, une seule LUT suffit.
        if abs(r - r0) < 1e-9:
            return float(r0), float(r0)

        return float(r0), float(r1)

    def compute_at(self, r_key):
        r_key = float(max(self.min_r, r_key))
        if r_key in self.cache:
            return self.cache[r_key]

        if self.verbose:
            t0 = time.perf_counter()
            print(f"computing LUT r={r_key:.3f}...")

        final_states = GR.orbital_geodesic_fast(
            r_key,
            self.M,
            RAYS_NUMBER=self.rays_number,
            MAX_STEPS=self.max_steps,
            h_step=self.h_step,
            h_update_interval=self.h_update_interval,
        )
        self.cache[r_key] = final_states
        if self.verbose:
            cp.cuda.get_current_stream().synchronize()
            print(f"LUT r={r_key:.3f} done in {time.perf_counter() - t0:.3f}s")
        return final_states

    def precompute_around(self, r, radius_count=2):
        """Pré-calcule r_key, puis les voisins ±dr, ±2dr, ..."""
        center = self.key(r)
        keys = [center]

        for i in range(1, radius_count + 1):
            keys.append(max(self.min_r, center - i * self.dr))
            keys.append(center + i * self.dr)

        # Déduplique en gardant l'ordre.
        seen = set()
        ordered_keys = []
        for key in keys:
            if key not in seen:
                seen.add(key)
                ordered_keys.append(key)

        for key in ordered_keys:
            self.compute_at(key)

    def prefetch_one_missing_around(self, r, radius_count=3):
        """Calcule une seule LUT manquante autour de r, utile pendant les temps morts."""
        center = self.key(r)
        candidates = []

        for i in range(0, radius_count + 1):
            if i == 0:
                candidates.append(center)
            else:
                candidates.append(max(self.min_r, center - i * self.dr))
                candidates.append(center + i * self.dr)

        for key in candidates:
            if key not in self.cache:
                self.compute_at(key)
                return True

        return False

    def prefetch_one_missing_with_margin(self, r, margin):
        """Calcule une seule LUT manquante dans [r-margin, r+margin].

        Exemple avec dr=2, r=50, margin=20 : remplit progressivement
        30, 32, ..., 50, ..., 68, 70 autour de la position courante.
        """
        return self.prefetch_one_missing_around(
            r,
            radius_count=self.radius_count_from_margin(margin),
        )

    def precompute_margin_around(self, r, margin):
        """Pré-calcule tout [r-margin, r+margin]. Bloquant, donc plutôt démarrage."""
        return self.precompute_around(
            r,
            radius_count=self.radius_count_from_margin(margin),
        )

    def get_interpolated(self, r):
        """Retourne (beta_grid, final_states) pour r, avec interpolation en r.

        captured est conservateur : si une des deux LUTs capture un rayon, le rayon
        interpolé est forcé capturé/noir.
        """
        r0, r1 = self.bracket_keys(r)
        X0 = self.compute_at(r0)

        if r0 == r1:
            return self.beta_grid, X0

        X1 = self.compute_at(r1)
        alpha = (r - r0) / (r1 - r0)

        X = (1.0 - alpha) * X0 + alpha * X1

        captured0 = X0[:, 1] <= 2.0 * self.M
        captured1 = X1[:, 1] <= 2.0 * self.M
        captured = captured0 | captured1

        # Force noir côté rendu : render_skybox_from_orbital_lut teste r <= 2M.
        X[captured, 1] = 2.0 * self.M
        X[captured, 3:6] = 0.0

        return self.beta_grid, X
