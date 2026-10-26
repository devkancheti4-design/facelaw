"""THE RECOGNISER, end to end, nothing given: one dark RAW image -> known person / unknown / too dark to tell.
Law 1 inverse -> Law 3 pose (photon centroid, coarse grid on the mean face, then per-candidate refit on the NOISY
image, validity: >= 85% of rigid pixels kept) -> Laws 2+4 subspace residual -> Law 5 z -> Law 6 lambda between the
best two -> the decision, executed by the machine-authored lanes (lanes_face.json) and the masks-and-ORs glue.
    python3 recognize.py PROBES [break]      break in {none, shine, shadow, glasses, saturate}"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, sys, time, os
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from lane_eval import E
from world import World, render, photons, sensor, light, pu, pv, IMG, Z0, F_PX, SIG_R, G_DN, B_DN
from laws import inverse, basis, fit, score, SIG2
from check_noise import GAL, W

LN = {l["name"]: l["expr"] for l in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lanes_face.json")))}
MEAN = dict(a=np.zeros(12), b=np.zeros(16), t=0.0)
_A0, _n0, _r0, _i0 = render(W, MEAN, (0, 0, 0, 0, 0, 1.0))
_m0 = (_i0["cnt"] > 0).reshape(IMG, IMG); _yy, _xx = np.mgrid[0:IMG, 0:IMG]
CX0, CY0 = (_m0 * _xx).sum() / _m0.sum(), (_m0 * _yy).sum() / _m0.sum()     # the mean face's FOOTPRINT centroid at t = 0


def footprint(Nh):
    """where the face is: photons above the black background, smoothed -- independent of how the face is lit"""
    from scipy.ndimage import uniform_filter
    S = uniform_filter(np.maximum(Nh, 0).reshape(IMG, IMG), 7)
    return S > 0.2 * np.percentile(S, 90)
NBARS = (2, 5, 10, 20, 50, 100, 200)


def lane(name, v):
    return E(LN[name], int(v))


def residual(Nh, ok, P, pose, n0=None):
    yaw, pitch, roll, tx, ty, s = pose
    if abs(yaw) > 0.6 or abs(pitch) > 0.4 or abs(roll) > 0.3 or not 0.8 <= s <= 1.25:
        return 1e12, None                              # valid pose: a physical head pose and distance...
    pix, B = basis(W, P, pose)
    if basis.cut > 0 or len(pix) < 50:                 # ...with the whole face in frame
        return 1e12, None
    z, info = score(Nh, ok, pix, B)
    return z, (pix, B, info)


def recognise(x):
    Nh, var, ok = inverse(x)                                                        # LAW 1
    m = footprint(Nh); yy, xx = np.mgrid[0:IMG, 0:IMG]
    cx, cy = (m * xx).sum() / m.sum(), (m * yy).sum() / m.sum()
    tx, ty = (cx - CX0) * Z0 / F_PX, -(cy - CY0) * Z0 / F_PX                            # LAW 3: footprint -> t
    n0 = None
    grid = lambda poses: min(((residual(Nh, ok, MEAN, q)[0], q) for q in poses), key=lambda t: t[0])[1]
    best = grid([(0, 0, 0, tx + a, ty + b, 1.0) for a in np.linspace(-0.2, 0.2, 5) for b in np.linspace(-0.2, 0.2, 5)])
    best = grid([(y, p, r, best[3], best[4], 1.0) for y in np.linspace(-0.35, 0.35, 7)
                 for p in np.linspace(-0.2, 0.2, 5) for r in (-0.1, 0, 0.1)])
    best = grid([(best[0], best[1], best[2], best[3] + a, best[4] + b, s) for a in (-0.06, 0, 0.06)
                 for b in (-0.06, 0, 0.06) for s in (0.95, 1.0, 1.05)])
    cand = []
    for j, P in enumerate(GAL):                                                     # per-candidate refit
        n0j = len(basis(W, P, (0, 0, 0, 0, 0, 1))[0])
        f = lambda q: residual(Nh, ok, P, tuple(q), n0j)[0]
        r = minimize(f, np.array(best), method="Nelder-Mead",
                     options=dict(initial_simplex=np.array(best) + np.vstack([np.zeros(6), np.diag([.06, .05, .05, .05, .05, .03])]),
                                  maxiter=220, xatol=1e-3, fatol=1e-2))
        z, got = residual(Nh, ok, P, tuple(r.x), n0j)
        if got is not None:
            cand.append((z, j, tuple(r.x)) + got)
    cand.sort(key=lambda c: c[0])
    (z1, j1, p1, pix1, B1, i1), (z2, j2, p2, pix2, B2, i2) = cand[0], cand[1]
    # LAW 6: lambda between the best's fitted image and the runner-up's subspace (an ideal observer's separation)
    A1, n1, rho1, inf1 = render(W, GAL[j1], p1)
    from world import sh9
    mu1 = A1 @ (rho1[:, None] * sh9(n1)) @ i1["c"]
    mu, w, c = fit(mu1[pix2], B2)
    lam = float((w * (mu1[pix2] - mu) ** 2).sum())
    # break detectors on the best candidate's rigid pixels
    keep = ok[pix1]
    y = Nh[pix1][keep]; m1 = B1[keep] @ i1["c"]
    wr = (y - m1) / np.sqrt(np.maximum(m1, 0) + SIG2)
    sat = int((~ok[pix1]).mean() > 0.05)
    out = int((np.abs(wr) > 4).mean() > 0.02)
    # LAW 5 open set: lambda_open = the smallest lambda at which Law 5's own FAR falls to eps (from this image's dof)
    from scipy.optimize import brentq
    from scipy.stats import norm
    sd0 = i1["sd0"]
    lam_open = brentq(lambda L: norm.cdf((2.326 - L / sd0) / np.sqrt(1 + 4 * L / sd0 ** 2)) - 0.01, 0, 1e7)
    u = min(4095, int(256 * lam / lam_open))
    dark = lane("L6_dark", min(4095, int(16 * lam))) | lane("L5_open", u)
    a1 = lane("L5_accept", int(np.clip(np.floor(16 * z1), -1024, 1023)))
    a2 = lane("L5_accept", int(np.clip(np.floor(16 * z2), -1024, 1023)))
    s = sat | out << 1 | dark << 2 | a1 << 3 | a2 << 4
    A = lane("OUT_any3", s) | lane("OUT_both", s)
    d = (A << 1) | (~A & ~(s >> 3) & 1)
    return dict(decision=("known", "unknown", "too dark")[d], who=j1 if d == 0 else None, z1=z1, z2=z2, lam=lam,
                lam_open=lam_open, limit=lane("L6_dark", min(4095, int(16 * lam))), method=lane("L5_open", u),
                sat=sat, out=out, dark=dark, a1=a1, a2=a2, best=j1)


def add_break(kind, Phi, info, R, unit_scale):
    if kind == "none":
        return Phi
    P = Phi.reshape(IMG, IMG).copy()
    ys, xs = np.nonzero(info["full"].reshape(IMG, IMG))
    cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[0:IMG, 0:IMG]
    if kind == "shine":                       # specular lobe on the nose bridge / forehead: not Lambertian
        P += 6 * unit_scale * np.exp(-((yy - cy + 4) ** 2 + (xx - cx) ** 2) / 6.0)
    elif kind == "shadow":                    # cast shadow of the nose across one cheek: not attached
        P[(xx > cx + 1) & (xx < cx + 9) & (yy > cy - 2) & (yy < cy + 6)] *= 0.15
    elif kind == "glasses":                   # dark frames around both eyes
        for ex in (-6, 6):
            r = np.sqrt((yy - cy + 5) ** 2 + (xx - cx - ex) ** 2)
            P[(r > 3.0) & (r < 4.5)] *= 0.2
    elif kind == "saturate":                  # a lamp in frame / overexposure: many pixels clip at white
        P *= 400.0 / max(P.max(), 1e-9)
    return P.ravel()


def probe(args):
    t, Nbar, brk = args
    R = np.random.default_rng(7000 + t)
    inside = t % 2 == 0
    P = GAL[int(R.integers(len(GAL)))] if inside else W.person(np.random.default_rng(90000 + t))
    truth = next(i for i, g in enumerate(GAL) if g is P) if inside else None
    pose = (R.uniform(-0.3, 0.3), R.uniform(-0.18, 0.18), R.uniform(-0.1, 0.1), R.uniform(-0.15, 0.15),
            R.uniform(-0.15, 0.15), R.uniform(0.94, 1.06))
    Ex = R.uniform(-1, 1, 3)
    Am, n, rho, info = render(W, P, pose, Ex, S=(pu, pv))
    L = light("point", R, n[np.unique(Am.indices)])
    Phi, info, k = photons(W, P, pose, L, Nbar, E=Ex)
    Phi = add_break(brk, Phi, info, R, Nbar)
    x = sensor(Phi, R)
    r = recognise(x)
    r.update(t=t, Nbar=Nbar, inside=inside, truth=truth, brk=brk,
             correct=(r["decision"] == "known" and r["who"] == truth) if inside else (r["decision"] == "unknown"),
             wrong=(r["decision"] == "known" and r["who"] != truth))
    return r


if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    brk = sys.argv[2] if len(sys.argv) > 2 else "none"
    nb = NBARS if brk == "none" else (20, 50)
    t0 = time.time()
    with Pool(12) as p:
        res = p.map(probe, [(t, N, brk) for N in nb for t in range(T)])
    json.dump(res, open("recognize_%s.json" % brk, "w"), indent=1)
    print("%d probes in %.0f s   (half enrolled, half strangers; pose, light, expression, exposure unknown)" % (len(res), time.time() - t0))
    print(" Nbar   enrolled: right  WRONG  unknown  too-dark    strangers: unknown  WRONG  too-dark   gates: L6 L5open a1&a2 out sat")
    for N in nb:
        rr = [r for r in res if r["Nbar"] == N]
        g = [r for r in rr if r["inside"]]; s = [r for r in rr if not r["inside"]]
        f = lambda L, k: sum(1 for r in L if k(r)) / max(len(L), 1)
        print(" %4d          %5.2f  %5.2f    %5.2f     %5.2f                %5.2f  %5.2f     %5.2f       %4.2f %5.2f %5.2f %4.2f %4.2f" % (
            N, f(g, lambda r: r["correct"]), f(g, lambda r: r["wrong"]), f(g, lambda r: r["decision"] == "unknown"),
            f(g, lambda r: r["decision"] == "too dark"), f(s, lambda r: r["decision"] == "unknown"), f(s, lambda r: r["wrong"]),
            f(s, lambda r: r["decision"] == "too dark"), f(rr, lambda r: r["limit"]), f(rr, lambda r: r["method"]), f(rr, lambda r: r["a1"] and r["a2"]),
            f(rr, lambda r: r["out"]), f(rr, lambda r: r["sat"])))
