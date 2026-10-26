"""THE OPTIMAL DECISION LAW -- Bayes with a reject option (Chow): name the most probable hypothesis only if its
posterior is >= 1 - eps; no rule has lower error at the same abstention rate.  Hypotheses: each enrolled person i,
and 'stranger' (any face of the population law).  All are scored on ONE evidence set E (the probe's rigid pixels),
so their likelihoods are comparable.
    log p(x | i)        = -R_i / 2                     (pose, light, exposure, expression profiled; Laws 1-4)
    log p(x | stranger) = -R_s / 2 - psi'P psi / 2 + log det P / 2 - log det H / 2     (Laplace over identity psi,
                          P = population precision, H = J'W(I - P_B)J + P, light profiled out)
    posterior = prior x likelihood, normalised;  pi_stranger = 1/2, pi_i = 1/(2K)
    python3 optimal.py PROBES"""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, sys, time, os
from multiprocessing import Pool
import numpy as np
from scipy.optimize import minimize
from scipy.ndimage import binary_erosion
from lane_eval import E as EV
from world import (World, render, photons, sensor, light, pu, pv, IMG, Z0, F_PX, SIG_R, sh9, sh18, EXPR_V, du, dv)
from laws import inverse, fit, SIG2
from check_noise import GAL, W
import recognize as RZ

EPS = 0.01
ALPHA = 0.06                                          # Law 2's measured worst leftover (5.7%), rounded up: model error
SA, SB, ST = 0.05, 0.07, 0.08                         # the population law (measured over many faces in practice)
PREC = np.concatenate([np.full(12, 1 / SA ** 2), np.full(16, 1 / SB ** 2), [1 / ST ** 2]])
LN = {l["name"]: l["expr"] for l in json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lanes_face.json")))}


def rigid(info):
    return info["full"] & (info["minv"] > EXPR_V + 0.03)


def evidence_set(ok, pose0):
    A, n, rho, info = render(W, RZ.MEAN, pose0)
    r = binary_erosion(rigid(info).reshape(IMG, IMG), np.ones((3, 3), bool)).ravel()
    return np.nonzero(r & ok)[0]


def predict(P, pose, Eset):
    """the candidate's basis on E; pixels of E it does not cover as rigid face predict black background"""
    yaw, pitch, roll, tx, ty, s = pose
    if abs(yaw) > 0.6 or abs(pitch) > 0.4 or abs(roll) > 0.3 or not 0.8 <= s <= 1.25:
        return None
    A, n, rho, info = render(W, P, pose)
    if info["cut"] > 0:
        return None
    B = (A @ (rho[:, None] * LIGHT(n)))[Eset]
    B[~rigid(info)[Eset]] = 0
    return B, (A, n, rho, info)


LIGHT = sh18                                          # every hypothesis gets the same, physically complete light law


def R_of(y, B):
    mu, w, c = fit(y, B, alpha=ALPHA)
    return float((w * (y - mu) ** 2).sum()), mu, w, c


def person_vec(P):
    return np.concatenate([P["a"], P["b"], [P["t"]]])


def vec_person(v):
    return dict(a=v[:12], b=v[12:28], t=float(v[28]))


def stranger(y, Eset, pose, start):
    """the stranger hypothesis profiles EVERYTHING the enrolled ones do, pose included: identity, then pose, twice"""
    ls, Rv, psi = stranger_at(y, Eset, pose, start)
    for rnd in range(2):
        def f(q):
            got = predict(vec_person(psi), tuple(q), Eset)
            return 1e12 if got is None else R_of(y, got[0])[0] + float(psi @ (PREC * psi))
        r = minimize(f, np.array(pose), method="Nelder-Mead",
                     options=dict(initial_simplex=np.array(pose) + np.vstack([np.zeros(6), np.diag([.03, .03, .03, .03, .03, .015])]),
                                  maxiter=150, xatol=1e-3, fatol=1e-2))
        pose = tuple(r.x)
        ls, Rv, psi = stranger_at(y, Eset, pose, vec_person(psi))
    return ls, Rv, psi, pose


def stranger_at(y, Eset, pose, start):
    """MAP identity under the population law (Levenberg-Marquardt, a step is kept only if it lowers R + psi'P psi,
    light profiled), started at the best enrolled person -- so the stranger fit can never end worse than that person
    -- then the Laplace evidence"""
    psi = person_vec(start).astype(float)
    h = 1e-3

    def F(v):
        got = predict(vec_person(v), pose, Eset)
        if got is None:
            return np.inf, None
        Rv, mu, w, c = R_of(y, got[0])
        return Rv + float(v @ (PREC * v)), (got[0], Rv, mu, w, c)

    Fc, cur = F(psi)
    damp = 1e-3
    for it in range(12):
        B, Rv, mu, w, c = cur
        J = np.empty((len(y), 29))
        for k in range(29):
            e = psi.copy(); e[k] += h
            Bk, _ = predict(vec_person(e), pose, Eset)
            J[:, k] = (Bk @ c - mu) / h
        sw = np.sqrt(w); Bw = B * sw[:, None]
        Pb = Bw @ np.linalg.pinv(Bw)                                      # light directions, profiled out
        Jw = (np.eye(len(y)) - Pb) @ (J * sw[:, None])
        rw = (np.eye(len(y)) - Pb) @ (sw * (y - mu))
        H = Jw.T @ Jw + np.diag(PREC)
        g = Jw.T @ rw - PREC * psi
        while damp < 1e6:
            step = np.linalg.solve(H + damp * np.diag(np.diag(H)), g)
            Fn, nxt = F(psi + step)
            if Fn < Fc:
                psi, Fc, cur, damp = psi + step, Fn, nxt, max(damp / 10, 1e-6)
                break
            damp *= 10
        else:
            break
        if np.abs(step).max() < 1e-5:
            break
    B, Rv, mu, w, c = cur
    J = np.empty((len(y), 29))
    for k in range(29):
        e = psi.copy(); e[k] += h
        Bk, _ = predict(vec_person(e), pose, Eset)
        J[:, k] = (Bk @ c - mu) / h
    sw = np.sqrt(w); Bw = B * sw[:, None]
    Jw = (np.eye(len(y)) - Bw @ np.linalg.pinv(Bw)) @ (J * sw[:, None])
    H = Jw.T @ Jw + np.diag(PREC)
    logdetH = np.linalg.slogdet(H)[1]
    return -Rv / 2 - 0.5 * float(psi @ (PREC * psi)) + 0.5 * float(np.log(PREC).sum()) - 0.5 * logdetH, Rv, psi


def recognise(x):
    Nh, var, ok = inverse(x)                                                          # LAW 1
    m = RZ.footprint(Nh); yy, xx = np.mgrid[0:IMG, 0:IMG]
    cx, cy = (m * xx).sum() / m.sum(), (m * yy).sum() / m.sum()
    tx, ty = (cx - RZ.CX0) * Z0 / F_PX, -(cy - RZ.CY0) * Z0 / F_PX                     # LAW 3
    grid = lambda poses: min(((RZ.residual(Nh, ok, RZ.MEAN, q)[0], q) for q in poses), key=lambda t: t[0])[1]
    p0 = grid([(0, 0, 0, tx + a, ty + b, 1.0) for a in np.linspace(-0.2, 0.2, 5) for b in np.linspace(-0.2, 0.2, 5)])
    p0 = grid([(yw, pt, rl, p0[3], p0[4], 1.0) for yw in np.linspace(-0.35, 0.35, 7) for pt in np.linspace(-0.2, 0.2, 5)
               for rl in (-0.1, 0, 0.1)])
    p0 = grid([(p0[0], p0[1], p0[2], p0[3] + a, p0[4] + b, s) for a in (-0.06, 0, 0.06) for b in (-0.06, 0, 0.06)
               for s in (0.95, 1.0, 1.05)])
    Eset = evidence_set(ok, p0)
    y = Nh[Eset]
    logp, fits = [], []
    for j, P in enumerate(GAL):                                                       # profile each enrolled person
        def f(q):
            got = predict(P, tuple(q), Eset)
            return 1e12 if got is None else R_of(y, got[0])[0]
        r = minimize(f, np.array(p0), method="Nelder-Mead",
                     options=dict(initial_simplex=np.array(p0) + np.vstack([np.zeros(6), np.diag([.06, .05, .05, .05, .05, .03])]),
                                  maxiter=220, xatol=1e-3, fatol=1e-2))
        logp.append(-r.fun / 2); fits.append(tuple(r.x))
    jb = int(np.argmax(logp))
    ls, Rs, psi, pose_s = stranger(y, Eset, fits[jb], GAL[jb])                        # the stranger hypothesis
    K = len(GAL)
    lp = np.array(logp + [ls]) + np.log(np.array([0.5 / K] * K + [0.5]))
    post = np.exp(lp - lp.max()); post /= post.sum()
    w = int(np.argmax(post))
    pw = float(min(post[w], 1 - 1e-12))
    lo = np.log(pw / (1 - pw))
    # break detectors on the winner's fit (Law 2's assumptions)
    if w < K:
        B, _ = predict(GAL[w], fits[w], Eset)
    else:
        B, _ = predict(vec_person(psi), pose_s, Eset)
    Rv, mu, wt, c = R_of(y, B)
    out = int((np.abs((y - mu) * np.sqrt(wt)) > 4).mean() > 0.02)
    sat = int((~ok[Eset]).mean() > 0.05) if len(Eset) else 1
    conf = EV(LN["L5_conf"], int(np.clip(np.floor(16 * lo), -2048, 2047)))
    s = sat | out << 1 | conf << 2 | int(w == K) << 3
    abst = EV(LN["OUT_abst"], s)
    d = (abst << 1) | (~abst & (s >> 3) & 1)
    return dict(decision=("known", "unknown", "too dark")[d], who=w if d == 0 else None, winner=w, post=pw, logodds=lo,
                logp=[float(v) for v in lp], E=len(Eset), sat=sat, out=out, conf=conf)


def probe(args):
    t, Nbar = args[:2]
    brk = args[2] if len(args) > 2 else "none"
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
    Phi = RZ.add_break(brk, Phi, info, R, Nbar)
    x = sensor(Phi, R)
    r = recognise(x)
    r.update(t=t, Nbar=Nbar, inside=inside, truth=truth, brk=brk,
             correct=(r["decision"] == "known" and r["who"] == truth) if inside else (r["decision"] == "unknown"),
             wrong=(r["decision"] == "known" and r["who"] != truth) or (inside and r["decision"] == "unknown"))
    return r


NBARS = (2, 5, 10, 20, 50, 100, 200)
if __name__ == "__main__":
    T = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    brk = sys.argv[2] if len(sys.argv) > 2 else "none"
    if brk != "none":
        NBARS = (20, 50, 100)
    t0 = time.time()
    with Pool(12) as p:
        res = p.map(probe, [(t, N, brk) for N in NBARS for t in range(T)])
    json.dump(res, open("optimal_results%s.json" % ("" if brk == "none" else "_" + brk), "w"), indent=1)
    print("%d probes in %.0f s   (half enrolled, half strangers; pose, light, expression, exposure unknown)" % (len(res), time.time() - t0))
    print(" Nbar   answered   WRONG among answered (law: <= %.0f%%)   enrolled: right too-dark   strangers: unknown too-dark" % (100 * EPS))
    for N in NBARS:
        rr = [r for r in res if r["Nbar"] == N]
        ans = [r for r in rr if r["decision"] != "too dark"]
        g = [r for r in rr if r["inside"]]; s = [r for r in rr if not r["inside"]]
        f = lambda L, k: sum(1 for r in L if k(r)) / max(len(L), 1)
        print(" %4d     %5.2f      %5.3f (%d of %d)                        %5.2f    %5.2f              %5.2f   %5.2f" % (
            N, len(ans) / len(rr), f(ans, lambda r: r["wrong"]), sum(r["wrong"] for r in ans), len(ans),
            f(g, lambda r: r["correct"]), f(g, lambda r: r["decision"] == "too dark"),
            f(s, lambda r: r["correct"]), f(s, lambda r: r["decision"] == "too dark")))
