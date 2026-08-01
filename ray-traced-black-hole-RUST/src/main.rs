use std::f64::consts::PI;
use std::path::Path;



// ---------------------
// ---- DATA STRUCT ----
// ---------------------

struct Simulation {
    max_steps: usize,
    h: f64,
}

struct Camera {
    fov: f64,
    x: f64,
    y: f64,
    z: f64,
    angle_vertical: f64,
    angle_horizontal: f64,
    width: u32,
    height: u32,
}

struct BlackHole {
    mass: f64,
    x: f64,
    y: f64,
    z: f64,
}




// ---------------------
// --- GENERAL FUNCS ---
// ---------------------


fn deg2rad(angle: f64) -> f64 {
    angle * PI / 180.0
}

fn normalize(vector: [f64; 3]) -> [f64; 3] {
    let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();

    [vector[0] / norm, vector[1] / norm, vector[2] / norm]
}

fn direction_to_angles(direction: [f64; 3]) -> (f64, f64) {
    let direction = normalize(direction);
    let theta = direction[2].clamp(-1.0, 1.0).acos();
    let phi = direction[1].atan2(direction[0]).rem_euclid(2.0 * PI);
    (theta, phi)
}




// ---------------------
// --- DISPLAY FUNC  ---
// ---------------------


fn initial_direction(camera: &Camera, pixel_x: u32, pixel_y: u32) -> [f64; 3] {
    let theta = camera.angle_vertical;
    let phi = camera.angle_horizontal;

    let forward = [
        theta.sin() * phi.cos(),
        theta.sin() * phi.sin(),
        theta.cos(),
    ];
    let right = [-phi.sin(), phi.cos(), 0.0];
    let up = [
        -theta.cos() * phi.cos(),
        -theta.cos() * phi.sin(),
        theta.sin(),
    ];

    // Coordonnees du centre du pixel sur un ecran compris entre -1 et 1.
    let screen_x = 2.0 * (pixel_x as f64 + 0.5) / camera.width as f64 - 1.0;
    let screen_y = 1.0 - 2.0 * (pixel_y as f64 + 0.5) / camera.height as f64;
    let tan_half_fov = (camera.fov / 2.0).tan();
    let aspect_ratio = camera.width as f64 / camera.height as f64;
    let x = screen_x * tan_half_fov;
    let y = screen_y * tan_half_fov / aspect_ratio;

    normalize([
        forward[0] + x * right[0] + y * up[0],
        forward[1] + x * right[1] + y * up[1],
        forward[2] + x * right[2] + y * up[2],
    ])
}

fn skybox_pixel(skybox: &image::RgbImage, theta: f64, phi: f64) -> image::Rgb<u8> {
    let (width, height) = skybox.dimensions();
    let x = (phi.rem_euclid(2.0 * PI) / (2.0 * PI) * width as f64) as u32;
    let y = (theta.clamp(0.0, PI) / PI * height as f64) as u32;

    *skybox.get_pixel(x.min(width - 1), y.min(height - 1))
}





// ---------------------
// ------ GR FUNCS -----
// ---------------------

// L'etat contient : t, r, beta, dt/dlambda, dr/dlambda, dbeta/dlambda.
fn initialize_ray(camera: &Camera, black_hole: &BlackHole, direction: [f64; 3]) -> (Vec<f64>, [f64; 3], [f64; 3]) {
    let radial_direction = normalize([camera.x - black_hole.x, camera.y - black_hole.y, camera.z - black_hole.z]);
    let radius = ( (camera.x - black_hole.x).powi(2) + (camera.y - black_hole.y).powi(2) + (camera.z - black_hole.z).powi(2) ).sqrt();

    let cos_alpha = (direction[0] * radial_direction[0] + direction[1] * radial_direction[1] + direction[2] * radial_direction[2]).clamp(-1.0, 1.0); // produit scalaire, clamp pour éviter les erreurs de précision float
    let sin_alpha = (1.0 - cos_alpha * cos_alpha).sqrt();

    // Direction tangentielle du plan dans lequel se deplace le photon.
    let tangent_direction = if sin_alpha > 1e-12 {
        [ (direction[0] - cos_alpha * radial_direction[0]) / sin_alpha, (direction[1] - cos_alpha * radial_direction[1]) / sin_alpha, (direction[2] - cos_alpha * radial_direction[2]) / sin_alpha]
    } else {
        // Le choix du plan n'a pas d'effet pour un rayon parfaitement radial.
        if radial_direction[0].abs() < 0.9 {
            normalize([0.0, radial_direction[2], -radial_direction[1]])
        } else {
            normalize([-radial_direction[2], 0.0, radial_direction[0]])
        }
    };

    let f = 1.0 - 2.0 * black_hole.mass / radius;

    let state = vec![
        0.0, radius, 0.0,
        1.0 / f.sqrt(), f.sqrt() * cos_alpha, sin_alpha / radius,
    ];

    (state, radial_direction, tangent_direction)
}

fn geodesic_dx(state: &[f64], black_hole: &BlackHole) -> Vec<f64> {
    let mut derivative = vec![0.0; state.len()];
    let radius = state[1];
    let f = 1.0 - 2.0 * black_hole.mass / radius;

    derivative[0] = state[3];
    derivative[1] = state[4];
    derivative[2] = state[5];
    derivative[3] = -2.0 * black_hole.mass / (radius * radius * f) * state[3] * state[4];
    derivative[4] = -black_hole.mass * f / (radius * radius) * state[3] * state[3]
        + black_hole.mass / (radius * radius * f) * state[4] * state[4]
        + radius * f * state[5] * state[5];
    derivative[5] = -2.0 / radius * state[4] * state[5];

    return derivative;
}

fn rk4(state: &[f64], h: f64, black_hole: &BlackHole) -> Vec<f64> {
    let k1 = geodesic_dx(state, black_hole);
    let state2: Vec<f64> = state
        .iter()
        .zip(&k1)
        .map(|(value, slope)| value + 0.5 * h * slope)
        .collect();
    let k2 = geodesic_dx(&state2, black_hole);
    let state3: Vec<f64> = state
        .iter()
        .zip(&k2)
        .map(|(value, slope)| value + 0.5 * h * slope)
        .collect();
    let k3 = geodesic_dx(&state3, black_hole);
    let state4: Vec<f64> = state
        .iter()
        .zip(&k3)
        .map(|(value, slope)| value + h * slope)
        .collect();
    let k4 = geodesic_dx(&state4, black_hole);

    (0..state.len())
        .map(|i| state[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

fn final_direction( state: &[f64], initial_radial: [f64; 3], initial_tangent: [f64; 3], black_hole: &BlackHole ) -> [f64; 3] {
    let beta = state[2];
    let radial = [beta.cos() * initial_radial[0] + beta.sin() * initial_tangent[0], beta.cos() * initial_radial[1] + beta.sin() * initial_tangent[1], beta.cos() * initial_radial[2] + beta.sin() * initial_tangent[2]];
    let tangent = [-beta.sin() * initial_radial[0] + beta.cos() * initial_tangent[0], -beta.sin() * initial_radial[1] + beta.cos() * initial_tangent[1], -beta.sin() * initial_radial[2] + beta.cos() * initial_tangent[2]];

    let f = 1.0 - 2.0 * black_hole.mass / state[1];
    let radial_speed = state[4] / f.sqrt();
    let tangent_speed = state[1] * state[5];

    normalize([radial_speed * radial[0] + tangent_speed * tangent[0], radial_speed * radial[1] + tangent_speed * tangent[1], radial_speed * radial[2] + tangent_speed * tangent[2]])
}

// None signifie que le photon a franchi l'horizon et doit etre affiche en noir.
fn trace_ray( camera: &Camera, black_hole: &BlackHole, simulation: &Simulation, direction: [f64; 3] ) -> Option<[f64; 3]> {

    let (mut state, initial_radial, initial_tangent) = initialize_ray(camera, black_hole, direction);
    let initial_radius = state[1];
    let schwarzschild_radius = 2.0 * black_hole.mass;

    for _step in 0..simulation.max_steps {
        state = rk4(&state, simulation.h, black_hole);

        if !state[1].is_finite() || state[1] <= schwarzschild_radius * 1.001 {
            return None;
        }

        // Une fois revenu a la distance de la camera, le rayon ne sera plus sensiblement devié : sa direction permet d'echantillonner la skybox.
        if state[1] >= initial_radius && state[4] > 0.0 {
            break;
        }
    }

    Some(final_direction(
        &state,
        initial_radial,
        initial_tangent,
        black_hole,
    ))
}

fn render_image( camera: &Camera, black_hole: &BlackHole, simulation: &Simulation, skybox: &image::RgbImage ) -> image::RgbImage {

    let mut render = image::RgbImage::new(camera.width, camera.height);

    for y in 0..camera.height {
        println!("{y} / {0}", camera.height);
        for x in 0..camera.width {
            let direction = initial_direction(camera, x, y);
            let pixel = match trace_ray(camera, black_hole, simulation, direction) {
                Some(direction) => {
                    let (theta, phi) = direction_to_angles(direction);
                    skybox_pixel(skybox, theta, phi)
                }
                None => image::Rgb([0, 0, 0]),
            };
            render.put_pixel(x, y, pixel);
        }
    }

    return render;
}




// ---------------------
// ----- MAIN FUNC -----
// ---------------------

fn main() {
    println!("\n---------- RAY-TRACED-BLACK-HOLE ----------\n");

    let simulation = Simulation { max_steps: 100_000, h: 0.05 };
    let camera = Camera { fov: deg2rad(75.0), x: 0.0, y: -50.0, z: 10.0, angle_vertical: PI / 2.0 + deg2rad(10.0), angle_horizontal: PI / 2.0, width: 200, height: 150 };
    let black_hole = BlackHole { mass: 1.0, x: 0.0, y: 0.0, z: 0.0 };

    let skybox_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/skybox.png");
    let render_path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/render.png");
    let skybox = image::open(skybox_path)
        .expect("Impossible de charger la skybox")
        .to_rgb8();

    let render = render_image(&camera, &black_hole, &simulation, &skybox);
    render
        .save(&render_path)
        .expect("Impossible d'enregistrer le rendu");
    println!("Image enregistree dans {}", render_path.display());
}
