import numpy as np

# local imports
import func
import GR



# --------------------
# Camera object
# --------------------

class Camera:
    def __init__(self, x=0, y=-100, z=0, angle_vertical=np.pi/2, angle_horizontal=0, fov=70, resolution=(400,300)):
        self.x = x
        self.y = y
        self.z = z
        self.angle_vertical = angle_vertical
        self.angle_horizontal= angle_horizontal
        self.fov = np.deg2rad(fov)
        self.resolution = resolution
        self.aspect_ratio = self.resolution[0] / self.resolution[1]
        self.camera_virtual_screen_width = 5
        self.distance_from_virtual_screen = self.camera_virtual_screen_width/(2*np.tan(self.fov/2)) + 0.001
        print(self.distance_from_virtual_screen)

    def move(self, dx, dy, dz, d_angle_vertical, d_angle_horizontal):
        self.x += dx
        self.y += dy
        self.z += dz
        self.angle_vertical += d_angle_vertical
        self.angle_horizontal += d_angle_horizontal

    def set_position(self, x, y, z, angle_vertical=None, angle_horizontal=None):
        self.x = x
        self.y = y
        self.z = z

        # Si on veut juste déplacer la caméra, on garde l'orientation actuelle.
        # Sinon, passer explicitement angle_vertical ET angle_horizontal.
        if angle_vertical is not None:
            self.angle_vertical = angle_vertical
        if angle_horizontal is not None:
            self.angle_horizontal = angle_horizontal




# --------------------
# Black Hole
# --------------------

class BlackHole:
    def __init__(self, x=0, y=0, z=0, mass=100):
        self.x = x
        self.y = y
        self.z = z
        self.mass = mass

    def Schwarzchild_radius(self):
        return 2*self.mass

    def Schwarzchild_shadow_radius(self):
        return 3*np.sqrt(3)*self.mass

    def move(self, dx, dy, dz):
        self.x += dx
        self.y += dy
        self.z += dz

    def set_position(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z



# --------------------
# Light Ray
# --------------------

class LightRay:
    def __init__(self, camera, pixel_x, pixel_y, blackhole=None, subpixel_x=0.5, subpixel_y=0.5):
        self.origin_x = camera.x
        self.origin_y = camera.y
        self.origin_z = camera.z

        # subpixel_x/subpixel_y décrivent la position à l'intérieur du pixel :
        # 0.5 = centre du pixel, 0.25/0.75 = petits décalages pour anti-aliasing.
        pixel_offset_x = (pixel_x + subpixel_x - camera.resolution[0]/2) / camera.resolution[0]
        pixel_offset_y = (pixel_y + subpixel_y - camera.resolution[1]/2) / camera.resolution[1]

        self.direction_angle_horizontal = camera.angle_horizontal + camera.fov * pixel_offset_x
        self.direction_angle_vertical = camera.angle_vertical + (camera.fov / camera.aspect_ratio) * pixel_offset_y

        forward = np.array([
            np.sin(camera.angle_vertical) * np.cos(camera.angle_horizontal),
            np.sin(camera.angle_vertical) * np.sin(camera.angle_horizontal),
            np.cos(camera.angle_vertical),
        ], dtype=np.float64)
        forward = forward / np.linalg.norm(forward)

        _, e_theta, e_phi = func.spherical_basis(camera.angle_vertical, camera.angle_horizontal)
        right = e_phi
        up = -e_theta

        screen_width = camera.camera_virtual_screen_width
        screen_height = screen_width / camera.aspect_ratio
        screen_x = pixel_offset_x * screen_width
        screen_y = pixel_offset_y * screen_height

        direction_vector = (
            camera.distance_from_virtual_screen * forward
            + screen_x * right
            - screen_y * up
        )
        self.direction_vector = (direction_vector / np.linalg.norm(direction_vector)).astype(np.float64)

        self.size_x_0 = camera.camera_virtual_screen_width*camera.distance_from_virtual_screen/camera.resolution[0]
        self.size_y_0 = camera.camera_virtual_screen_width*camera.distance_from_virtual_screen/camera.resolution[1]
        self.distance_traveled = camera.distance_from_virtual_screen

        self.size_x = self.size_x_0
        self.size_y = self.size_y_0
        direction = self.direction()
        self.x = camera.x + direction[0]*self.distance_traveled
        self.y = camera.y + direction[1]*self.distance_traveled
        self.z = camera.z + direction[2]*self.distance_traveled
        if blackhole is not None:
            self.state = self.photon_initial_parameters_black_hole(camera, blackhole)
        else:
            self.state = None

    def photon_initial_parameters_black_hole(self, camera, blackhole):
        """
        Construit l'état initial GR du photon dans les coordonnées de Schwarzschild
        centrées sur le trou noir.

        Retourne :
            state = [t, r, theta, phi, ut, ur, utheta, uphi]

        Idée :
        - position caméra cartésienne relative au trou noir -> (r, theta, phi)
        - direction cartésienne du rayon -> (ur, utheta, uphi)
        - ut est fixé par la condition photon ds² = 0.

        Approximation : la direction caméra est traitée comme direction spatiale
        localement euclidienne. C'est OK pour commencer si la caméra est loin de 2M.
        """
        # Position initiale = caméra, exprimée relativement au centre du trou noir.
        pos = np.array([
            camera.x - blackhole.x,
            camera.y - blackhole.y,
            camera.z - blackhole.z,
        ], dtype=np.float64)

        x, y, z = pos
        r = np.linalg.norm(pos)

        if r <= 2.0 * blackhole.mass:
            raise ValueError("Camera is inside or on the black hole horizon")

        # Direction cartésienne du rayon, dans le repère monde.
        direction_world = self.direction().astype(np.float64)
        direction_world = direction_world / np.linalg.norm(direction_world)

        # Astuce importante : Schwarzschild est sphériquement symétrique, donc
        # chaque rayon peut être intégré dans SON repère local. On choisit l'axe
        # z local aligné avec L = r x v. Le rayon reste alors dans le plan
        # équatorial theta=pi/2, au lieu de passer près d'un pôle sphérique où
        # cot(theta) explose numériquement.
        angular_momentum_axis = np.cross(pos, direction_world)
        angular_momentum_norm = np.linalg.norm(angular_momentum_axis)
        if angular_momentum_norm > 1e-12:
            e_z_local_world = angular_momentum_axis / angular_momentum_norm
            e_x_local_world = pos / r
            e_y_local_world = np.cross(e_z_local_world, e_x_local_world)
            e_y_local_world = e_y_local_world / np.linalg.norm(e_y_local_world)
            self.gr_basis = np.column_stack((e_x_local_world, e_y_local_world, e_z_local_world))
        else:
            self.gr_basis = np.eye(3, dtype=np.float64)

        pos_gr = self.gr_basis.T @ pos
        direction = self.gr_basis.T @ direction_world

        _, theta, phi = func.cartesian_to_spherical(pos_gr)

        sin_theta = np.sin(theta)

        # Base sphérique orthonormée locale exprimée en cartésien.
        e_r, e_theta, e_phi = func.spherical_basis(theta, phi)

        # Composantes physiques de la direction sur la base sphérique locale.
        direction_r = np.dot(direction, e_r)
        direction_theta = np.dot(direction, e_theta)
        direction_phi = np.dot(direction, e_phi)

        # Conversion vers vitesses coordonnées :
        # ds_spatial² = dr² + r² dtheta² + r² sin²(theta) dphi² localement.
        ur = direction_r
        utheta = direction_theta / r

        if abs(sin_theta) < 1e-12:
            # Aux pôles, phi est mal défini. On évite la division par zéro.
            uphi = 0.0
        else:
            uphi = direction_phi / (r * sin_theta)

        M = blackhole.mass
        f = 1.0 - 2.0 * M / r

        # Condition photon : g_mu_nu u^mu u^nu = 0
        # 0 = -f ut² + ur²/f + r² utheta² + r² sin²(theta) uphi²
        spatial_norm = (
            ur**2 / f
            + r**2 * utheta**2
            + r**2 * sin_theta**2 * uphi**2
        )
        ut = np.sqrt(spatial_norm / f)

        return np.array([
            0.0,    # t
            r,
            theta,
            phi,
            ut,
            ur,
            utheta,
            uphi,
        ], dtype=np.float64)


    def move_light(self, h=0.05, blackhole=None):
        direction = self.direction()

        # Mode ray-tracing plat : pas d'état GR à intégrer.
        if self.state is None or blackhole is None:
            self.x += direction[0]*self.distance_traveled
            self.y += direction[1]*self.distance_traveled
            self.z += direction[2]*self.distance_traveled
            return

        # Mode GR : self.state = [t, r, theta, phi, ut, ur, utheta, uphi]
        # Pas adaptatif très simple : gros pas loin du trou noir, petit pas
        # près de l'horizon/la sphère photonique pour éviter les anneaux pixellisés.
        if not np.all(np.isfinite(self.state)):
            return

        r = self.state[1]
        M = blackhole.mass
        if r <= 2.05 * M:
            return

        if r < 8.0 * M:
            # Près de la sphère photonique, h=0.5 est encore trop gros :
            # certains rayons quasi centraux font un saut numérique énorme puis
            # sont faussement classés comme "échappés", ce qui crée la bande verticale.
            h_effective = min(h, max(0.05, 0.15 * (r - 2.0 * M)))
        else:
            h_effective = min(h, max(0.5, 0.4 * (r - 2.0 * M)))

        self.state = GR.rk4_step(self.state.copy(), h_effective, blackhole.mass)

        if not np.all(np.isfinite(self.state)):
            return

        # On reconvertit la position Schwarzschild (r,theta,phi) en coordonnées
        # cartésiennes monde pour garder collision()/skybox compatibles.
        local_pos = func.spherical_to_cartesian(self.state[1:4])
        if hasattr(self, "gr_basis"):
            x_rel, y_rel, z_rel = self.gr_basis @ local_pos
        else:
            x_rel, y_rel, z_rel = local_pos
        self.x = blackhole.x + x_rel
        self.y = blackhole.y + y_rel
        self.z = blackhole.z + z_rel

    def direction(self):
        # Direction caméra calculée avec un écran virtuel local.
        # Les angles sphériques restent l'orientation CENTRALE de la caméra,
        # mais les pixels ne sont plus générés par theta+=dy, phi+=dx.
        if hasattr(self, "direction_vector"):
            return self.direction_vector

        # Fallback : coordonnées sphériques classiques.
        dx = np.sin(self.direction_angle_vertical) * np.cos(self.direction_angle_horizontal)
        dy = np.sin(self.direction_angle_vertical) * np.sin(self.direction_angle_horizontal)
        dz = np.cos(self.direction_angle_vertical)
        direction = np.array([dx, dy, dz], dtype=np.float64)
        return direction / np.linalg.norm(direction)

    def collision(self, MAP, skybox=None, blackhole=None):
        if blackhole is not None:
            if self.state is not None:
                M = blackhole.mass
                r = self.state[1]
                ur = self.state[5]

                # Sécurité numérique : si RK4 part en NaN/inf, on ne laisse pas
                # le rayon ressortir coloré depuis le centre de l'ombre.
                if not np.all(np.isfinite(self.state)):
                    return np.array([0, 0, 0], dtype=np.float32)

                # Critère analytique de capture : si le photon arrive depuis loin
                # vers le trou noir avec b < b_crit = 3 sqrt(3) M, il appartient
                # à l'ombre. Ça évite d'intégrer des rayons centraux qui peuvent
                # numériquement "traverser" l'horizon et ressortir colorés.
                b = GR.photon_impact_parameter(self.state, M)
                b_crit = 3.0 * np.sqrt(3.0) * M
                if ur < 0 and b <= 1.002 * b_crit:
                    return np.array([0, 0, 0], dtype=np.float32)

                # Capture certaine avant la zone numériquement violente.
                # Si r<3M et ur<0, le photon est à l'intérieur de la sphère
                # photonique et se dirige vers le trou noir : il ne ressortira pas.
                # Ça évite les fuites numériques qui créent la bande verticale au centre.
                if r <= 3.0 * M and ur < 0:
                    return np.array([0, 0, 0], dtype=np.float32)

                # En mode GR, on capture un peu AVANT r=2M pour éviter la
                # singularité de coordonnées Schwarzschild.
                if r <= 2.05 * M:
                    return np.array([0, 0, 0], dtype=np.float32)

                # Si le photon est loin et s'éloigne, verdict sûr : skybox.
                # Ne pas attendre le bord de la MAP : en GR, dès qu'il est
                # suffisamment loin du trou noir ET ur>0, il part vers l'infini.
                origin_r = func.distance(
                    (self.origin_x, self.origin_y, self.origin_z),
                    (blackhole.x, blackhole.y, blackhole.z),
                )
                r_escape = min(
                    min(MAP.shape[0], MAP.shape[1], MAP.shape[2]) / 2,
                    max(origin_r + 5 * M, 20 * M),
                )
                if r > r_escape and ur > 0:
                    if skybox is not None:
                        final_direction = func.spatial_direction_from_gr_state(self.state, self.gr_basis)
                        return func.skybox_color(final_direction, skybox)
                    return np.array([0, 0, 0], dtype=np.float32)
            else:
                # En ray-tracing plat, shadow_radius est seulement un hack visuel.
                if func.distance((self.x, self.y, self.z), (blackhole.x, blackhole.y, blackhole.z)) < blackhole.Schwarzchild_shadow_radius():
                    return np.array([0, 0, 0], dtype=np.float32)

        if (self.z > MAP.shape[0]/2) or (self.z < -MAP.shape[0]/2) or (self.y > MAP.shape[1]/2) or (self.y < -MAP.shape[1]/2) or (self.x > MAP.shape[2]/2) or (self.x < -MAP.shape[2]/2):
            if skybox is not None:
                if blackhole is not None and self.state is not None:
                    # En GR, la bonne direction de skybox est la direction finale,
                    # pas self.direction() qui est seulement la direction initiale.
                    final_direction = func.spatial_direction_from_gr_state(self.state, self.gr_basis)
                    return func.skybox_color(final_direction, skybox)

                origin = (self.origin_x, self.origin_y, self.origin_z)
                sphere_radius = min(MAP.shape[0], MAP.shape[1], MAP.shape[2]) / 2
                return func.skybox_color_from_position(origin, self.direction(), skybox, sphere_radius)
            return np.array([0, 0, 0], dtype=np.float32)

        return None

    def render_light_ray(self, MAP, skybox=None, blackhole=None, max_steps=200):
        pixel = None
        steps = 0
        while pixel is None and steps < max_steps:
            self.move_light(h=2.0, blackhole = blackhole)
            pixel = self.collision(MAP, skybox, blackhole)
            steps += 1

        if pixel is None:
            # En mode GR, un rayon non résolu ne doit pas créer un pixel coloré
            # au milieu de l'ombre : on le classe noir.
            if blackhole is not None and self.state is not None:
                return np.array([0, 0, 0], dtype=np.float32)
            return np.array([1, 0, 1], dtype=np.float32)

        return pixel


