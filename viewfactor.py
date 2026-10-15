# Analytic view factor between two arbitrarily oriented planar polygons.
# Python port of viewFactor.m v1.5 by Jacob Kerkhoff (UW-Madison Solar Energy Laboratory),
# which implements Narayanaswamy (2015), Int. J. Heat Mass Transfer 91:841-847.
# https://github.com/uw-esolab/docs/blob/main/tools/viewfactor/viewfactor.md
# Polygons must be planar, simple and fully visible to each other (true in a convex enclosure).

import numpy as np

ALMOST_ZERO = 1e-7

# Chebyshev coefficients for Clausen's integral, paper Table 1
CHEB_B = np.array([1.865555351433979e-1, 6.269948963579612e-2, 3.139559104552675e-4,
                   3.916780537368088e-6, 6.499672439854756e-8, 1.238143696612060e-9,
                   5.586505893753557e-13])


def clausen(theta):
    # Clausen's integral Cl2(theta), paper Eq. (26): the part of the dilogarithm we need
    theta = np.mod(theta, 2 * np.pi)
    x = theta / np.pi - 1
    # Chebyshev polynomials T_{2n+1}(x), n = 0..6
    T = [x,
         4*x**3 - 3*x,
         16*x**5 - 20*x**3 + 5*x,
         64*x**7 - 112*x**5 + 56*x**3 - 7*x,
         256*x**9 - 576*x**7 + 432*x**5 - 120*x**3 + 9*x,
         1024*x**11 - 2816*x**9 + 2816*x**7 - 1232*x**5 + 220*x**3 - 11*x,
         4096*x**13 - 13312*x**11 + 16640*x**9 - 9984*x**7 + 2912*x**5 - 364*x**3 + 13*x]
    return ((theta - np.pi) * (2 + np.log(np.pi**2 / 2))
            + (2*np.pi - theta) * np.log((2*np.pi - theta) * (1 - ALMOST_ZERO) + ALMOST_ZERO)
            - theta * np.log(theta * (1 - ALMOST_ZERO) + ALMOST_ZERO)
            + float(np.dot(CHEB_B, T)))


def imag_li2(mag, angle):
    # Imaginary part of the dilogarithm Li2(1 - mag*e^{i*angle}), paper Eq. (24)
    if mag > ALMOST_ZERO:
        omega = np.arctan2(mag * np.sin(angle), 1 - mag * np.cos(angle))
        return (0.5 * clausen(2*angle) + 0.5 * clausen(2*omega)
                - 0.5 * clausen(2*omega + 2*angle) + np.log(mag) * omega)
    return mag * np.sin(angle)


def f_skew(s, l, alpha, cos_a, sin_a, d):
    # Paper Eq. (22b): edge-pair contribution when the two edges are skew
    s2, l2, d2, sin2 = s*s, l*l, d*d, sin_a*sin_a
    wsqrt = np.sqrt(s2 + d2 / sin2)
    psqrt = np.sqrt(l2 + d2 / sin2)
    wdim = s + wsqrt if abs(s + wsqrt) > 0 else ALMOST_ZERO
    pdim = l + psqrt if abs(l + psqrt) > 0 else ALMOST_ZERO
    return ((0.5 * cos_a * (s2 + l2) - s*l) * np.log(s2 + l2 - 2*s*l*cos_a + d2)
            + s * sin_a * wsqrt * np.arctan2(np.sqrt(s2*sin2 + d2), l - s*cos_a)
            + l * sin_a * psqrt * np.arctan2(np.sqrt(l2*sin2 + d2), s - l*cos_a) + s*l
            + 0.5 * (d2 / sin_a) * (imag_li2(wdim/pdim, alpha) + imag_li2(pdim/wdim, alpha)
                                    - 2 * imag_li2((wdim - 2*s)/pdim, np.pi - alpha)))


def f_parallel(s, l, d):
    # Paper Eq. (23): the limit when the two edges are parallel
    if d == 0:
        d = ALMOST_ZERO
    sl = s - l
    return (0.5 * (sl*sl - d*d) * np.log(sl*sl + d*d)
            - 2 * sl * d * np.arccos(np.clip(sl / np.sqrt(s*s + l*l - 2*s*l + d*d), -1, 1))
            + s*l)


def edge_pair_parameters(Po, Pf, Qo, Qf):
    # Shortest distance between the infinite lines through the two edges, and the
    # parameterization (origins and directions) the paper's Fig. 5 defines
    u = Pf - Po
    v = Qf - Qo
    w = Po - Qo
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    b = np.dot(u, v)
    d_ = np.dot(u, w)
    e = np.dot(v, w)
    den = 1 - b*b
    if den > ALMOST_ZERO:          # skew (not parallel) edges
        skew = True
        s = (b*e - d_) / den
        l = (e - b*d_) / den
        D = np.linalg.norm(w + s*u - l*v)
    else:                           # parallel edges
        skew = False
        s = 0.0
        l = e
        D = np.linalg.norm(w - e*v)
    s_origin = Po + u*s
    l_origin = Qo + v*l
    # unit vectors point from the origin toward the farther of the two vertices
    if abs(s) < np.linalg.norm(Pf - s_origin):
        s_hat = (Pf - s_origin) / np.linalg.norm(Pf - s_origin)
    else:
        s_hat = (Po - s_origin) / np.linalg.norm(Po - s_origin)
    if abs(l) < np.linalg.norm(Qf - l_origin):
        l_hat = (Qf - l_origin) / np.linalg.norm(Qf - l_origin)
    else:
        l_hat = (Qo - l_origin) / np.linalg.norm(Qo - l_origin)
    return D, s_origin, s_hat, l_hat, l_origin, skew


def clean_polygon(poly, tol=1e-9):
    # Drop repeated consecutive vertices (a clip can produce them when a corner
    # lands exactly on the clipping edge); they create zero-length edges
    poly = [np.asarray(p, dtype=float) for p in poly]
    kept = [p for k, p in enumerate(poly) if np.linalg.norm(p - poly[k - 1]) > tol]
    return kept


def polygon_area(poly):
    # Area of a planar polygon from its 3D vertices (Newell's method: robust
    # even when the first vertices are collinear)
    poly = np.asarray(poly, dtype=float)
    total = np.zeros(3)
    for i in range(len(poly)):
        total += np.cross(poly[i], poly[(i + 1) % len(poly)])
    return np.linalg.norm(total) / 2


def view_factor(poly_a, poly_b):
    # F from polygon A to polygon B and back: (F_ab, F_ba).
    # Each polygon: (N,3) array of vertices in order around the perimeter.
    A = np.asarray(clean_polygon(poly_a), dtype=float)
    B = np.asarray(clean_polygon(poly_b), dtype=float)
    area_a = polygon_area(A)
    area_b = polygon_area(B)
    total = 0.0
    for p in range(len(B)):                 # every edge of B...
        for i in range(len(A)):             # ...against every edge of A, Eq. (22a)
            r_i, r_j = A[i], A[(i + 1) % len(A)]
            r_p, r_q = B[p].copy(), B[(p + 1) % len(B)].copy()
            # coincident vertices (shared corner) break the parameterization: nudge B
            if (np.linalg.norm(r_i - r_p) < ALMOST_ZERO/2 or np.linalg.norm(r_j - r_p) < ALMOST_ZERO/2):
                r_p = r_p + ALMOST_ZERO
            elif (np.linalg.norm(r_i - r_q) < ALMOST_ZERO/2 or np.linalg.norm(r_j - r_q) < ALMOST_ZERO/2):
                r_q = r_q + ALMOST_ZERO
            dmin, s_origin, s_hat, l_hat, l_origin, skew = edge_pair_parameters(r_i, r_j, r_p, r_q)
            if skew:
                s_i = np.dot(r_i - s_origin, s_hat)
                s_j = np.dot(r_j - s_origin, s_hat)
                l_p = np.dot(r_p - l_origin, l_hat)
                l_q = np.dot(r_q - l_origin, l_hat)
                cos_a = np.clip(np.dot(s_hat, l_hat), -1, 1)
                alpha = np.arccos(cos_a)
                sin_a = np.sin(alpha)
                total += cos_a * (f_skew(s_j, l_q, alpha, cos_a, sin_a, dmin)
                                  - f_skew(s_i, l_q, alpha, cos_a, sin_a, dmin)
                                  - f_skew(s_j, l_p, alpha, cos_a, sin_a, dmin)
                                  + f_skew(s_i, l_p, alpha, cos_a, sin_a, dmin))
            else:
                l_hat = s_hat                # the parallel case measures l along s_hat
                s_i = np.dot(r_i - s_origin, s_hat)
                s_j = np.dot(r_j - s_origin, s_hat)
                l_p = np.dot(r_p - l_origin, l_hat)
                l_q = np.dot(r_q - l_origin, l_hat)
                total += (f_parallel(s_j, l_q, dmin) - f_parallel(s_i, l_q, dmin)
                          - f_parallel(s_j, l_p, dmin) + f_parallel(s_i, l_p, dmin))
    rad_ua = abs(total) / (4 * np.pi)        # A_a * F_ab = A_b * F_ba, reciprocity built in
    return rad_ua / area_a, rad_ua / area_b