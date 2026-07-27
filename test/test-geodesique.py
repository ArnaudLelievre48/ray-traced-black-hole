import numpy as np
import matplotlib.pyplot as plt

def init_massive_particle(r, phi, u_phi, M):
    f = 1.0 - 2.0 * M / r
    u_t = np.sqrt( (1 + r**2 * u_phi**2) / f )
    X = np.stack([0,r,phi,
                  u_t, 0, u_phi], dtype=np.float64)
    return X


def geodesic_dX(X, M):
    """Membre de droite de la géodésique plane.

    X[ray_id] = [t, r, phi, ut, ur, uphi]
    dX[ray_id] = [dt, dr, dphi, dut, dur, duphi]
    """
    r = X[1]
    ut = X[3]
    ur = X[4]
    uphi = X[5]

    # Schwarzschild est singulier en r=2M.
    # Si un rayon est déjà capturé / non fini, on met dX=0 pour éviter de
    # calculer des termes explosifs. orbital_geodesic le désactivera ensuite.
    capture_radius = 2.05 * M

    dX = np.zeros_like(X)

    f = 1.0 - 2.0 * M / r

    # dx^alpha/dlambda = u^alpha
    dX[0] = ut
    dX[1] = ur
    dX[2] = uphi

    # du^alpha/dlambda = -Gamma^alpha_{mu nu} u^mu u^nu
    # Version explicite du même calcul que Gamma + einsum, mais sans allouer
    # Gamma[ray_id,3,3,3] à chaque sous-pas RK4.
    dX[3] = -2.0 * M / (r**2 * f) * ut * ur
    dX[4] = (
        -M * f / r**2 * ut**2
        + M / (r**2 * f) * ur**2
        + r * f * uphi**2
    )
    dX[5] = -2.0 / r * ur * uphi

    return dX




def rk4_step(X, M, h=0.1):
    """Avance X d'un pas RK4.

    h peut être :
        - scalaire ;
        - tableau (RAYS_NUMBER,) ;
        - tableau (RAYS_NUMBER, 1).

    En interne on veut h.shape == (RAYS_NUMBER, 1), pour que chaque rayon ait
    son propre pas mais que ce pas multiplie les 6 composantes de son état.
    """

    k1 = geodesic_dX(X, M)
    k2 = geodesic_dX(X + 0.5 * h * k1, M)
    k3 = geodesic_dX(X + 0.5 * h * k2, M)
    k4 = geodesic_dX(X + h * k3, M)

    return X + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


def position_radiale2cartesian(r, phi):
    x = r*np.cos(phi)
    y = r*np.sin(phi)
    return np.array( [x,y] )

def black_hole_schwarzchild_radius(M):
    r = 2.0 * M
    circle = np.zeros( (500, 2) )

    PHI = np.linspace(0,2*np.pi,circle.shape[0])
    for i in range(circle.shape[0]):
        phi = PHI[i]
        x = r*np.cos(phi)
        y = r*np.sin(phi)
        circle[i] = np.array([x,y])
    return circle

r = 50
phi = -np.pi/2
uphi = 0.003
M = 2

STEPS = 200_000
PARTICLES = 10

POSITIONS = np.zeros( (PARTICLES, STEPS, 2) )

plt.plot(np.array(black_hole_schwarzchild_radius(M)[:,0]), np.array(black_hole_schwarzchild_radius(M)[:,1]), color="black")

for i in range(POSITIONS.shape[0]):
    uphi_rand = uphi + (uphi/10)*(np.random.rand()-0.5)
    X = init_massive_particle(r, phi, uphi_rand, M)
    POSITIONS[i, 0] = position_radiale2cartesian(X[1],X[2])

    for j in range(POSITIONS.shape[1]):
        X = rk4_step(X, M)
        if X[1] <= 2.05*M:
            break
        POSITIONS[i, j] = position_radiale2cartesian(X[1], X[2])

    plt.plot(np.array(POSITIONS[i, :,0]), np.array(POSITIONS[i, :,1]))

plt.show()
