"""SUPERSET: the current law C (joint light law) keeps every answer; the local law L may take over ONLY where the data of
the DESIGN people shows L wins the disagreements.  State per probe: agree (C and L name the same person), C's margin
(R2 - R1) / R1 and L's margin (d2 - d1) / d1, each in 4 levels -> s = agree | cbin << 1 | lbin << 3 (32 states).
Gate truth on the design fold: 1 iff  #(L right, C wrong) > #(C right, L wrong)  in that state;  0 otherwise and for any
state the design data never shows (the current law is unaffected).  The engine authors each fold's gate (fresh engine state),
proven on all 32 states; the law is then scored on the OTHER fold's people (cross-fitted)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, os, pickle, sys, time
import numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yale as Y
from yale_baselines import load_full

LS = json.load(open("lanes_sector.json"))
CAST = json.load(open("lane_cast.json"))["expr"]
EDGES = (0.02, 0.08, 0.3)


def ev(expr, x):
    return eval(expr, {"x": x})


def sectors(I, sigma=0.75):
    """the engine's lanes only: gradient -> packed 12-bit fields -> sector 0..7 (angular order)"""
    G = gaussian_filter(I, sigma)
    gx, gy = np.gradient(G, axis=1), -np.gradient(G, axis=0)
    q = lambda g: (np.clip(np.rint(16 * g), -2048, 2047) + 2048).astype(np.int64)
    x = q(gx) | (q(gy) << 12)
    sy, sx = ev(LS["O_sy"]["expr"], x), ev(LS["O_sx"]["expr"], x)
    ax, ay = ev(LS["O_ax"]["expr"], x), ev(LS["O_ay"]["expr"], x)
    d = ev(CAST, ay | (ax << 12))                       # [ay > ax]: the proven cast lane, reused
    return ev(LS["O_perm"]["expr"], 4 * sy + 2 * sx + d).ravel().astype(np.int8)


def local_law(full):
    gal, lab = [], []
    for p in Y.IDS:
        for sb, I in full[p]:
            if sb == 1:
                gal.append(sectors(I)); lab.append(p)
    gal = np.array(gal, np.int16); lab = np.array(lab)
    probes, PD = [], []
    for p in Y.IDS:
        for sb, I in full[p]:
            if sb >= 2:
                k = sectors(I).astype(np.int16)
                dd = np.abs(gal - k[None]); dd = np.minimum(dd, 8 - dd).sum(1).astype(float)
                PD.append([dd[lab == q].min() for q in Y.IDS]); probes.append((p, sb))
    return probes, np.array(PD)


def margin(v):
    s = np.sort(v)
    return (s[1] - s[0]) / max(s[0], 1e-9)


def binof(m):
    return int(np.searchsorted(EDGES, m))


if __name__ == "__main__":
    t0 = time.time()
    from lane_eval import E
    try:
        from engine import author, header        # the authoring engine (not included in this repository)
    except ImportError:
        author = header = None
    full = load_full()
    probesL, PD = local_law(full)
    J = pickle.load(open("yale_joint_res.pkl", "rb"))
    assert J["probes"] == probesL, "probe order differs"
    RC = J["res"].T                                                     # probes x 38
    ids = np.array(Y.IDS)
    rows = []
    for j, (p, sb) in enumerate(probesL):
        c, l = ids[RC[j].argmin()], ids[PD[j].argmin()]
        s = int(c == l) | binof(margin(RC[j])) << 1 | binof(margin(PD[j])) << 3
        rows.append(dict(p=p, sb=sb, c=c, l=l, s=s))
    folds = [set(p for i, p in enumerate(Y.IDS) if i % 2 == k) for k in (0, 1)]
    header("SUPERSET GATE -- the engine authors when the local law may take over")
    gates = {}
    for k in (0, 1):
        design = [r for r in rows if r["p"] in folds[1 - k]]
        # lane law: agreement only masks (glue: gate = g(s >> 1) & ~s & 1); the engine authors g over y = cbin | lbin << 2
        truth = []
        for y in range(16):
            rs = [r for r in design if (r["s"] >> 1) == y and not (r["s"] & 1)]
            win = sum(r["l"] == r["p"] and r["c"] != r["p"] for r in rs)
            loss = sum(r["c"] == r["p"] and r["l"] != r["p"] for r in rs)
            truth.append((y, int(win > loss)))
        FAM = dict(SHF=(">>1", ">>2", ">>3"), BIT=("&", "|", "~", "^"), LIN=("-", "+1"))
        expr, info = author("GATE_f%d" % k, truth, FAM, (1, 3))
        if expr is None:                                    # cut again: the local law's margin level 0 vs >= 1
            e0, i0 = author("GATE_f%d_l0" % k, [(y, t) for y, t in truth if y < 4], FAM, (1, 3))
            e1, i1 = author("GATE_f%d_l1" % k, [(y, t) for y, t in truth if y >= 4], FAM, (1, 3, 4))
            expr = "((((x >> 2) + 3) >> 2) & (%s)) | ((((x >> 2) + 3) >> 2) ^ 1) & (%s)" % (e1, e0)  # glue: lbin >= 1 ? e1 : e0
            info = dict(gen=(i0.get("gen"), i1.get("gen")))
        assert all(eval(expr, {"x": y}) == t for y, t in truth), "gate not exact"
        gates[k] = dict(expr=expr, truth=truth, gen=info.get("gen"))
    acc = {"current": {}, "local": {}, "superset": {}}
    changed = regress = fixed = 0
    for r in rows:
        k = 0 if r["p"] in folds[0] else 1
        use_l = (eval(gates[k]["expr"], {"x": r["s"] >> 1}) & ~r["s"] & 1) == 1
        final = r["l"] if use_l else r["c"]
        for name, ans in (("current", r["c"]), ("local", r["l"]), ("superset", final)):
            acc[name].setdefault(r["sb"], []).append(ans == r["p"])
        if final != r["c"]:
            changed += 1; regress += r["c"] == r["p"]; fixed += final == r["p"]
    print("\nPROTOCOL A, 38 people, cross-fitted by people -- recognition %")
    print("  subset   current (joint law)   local law (engine sectors, 192x168)   SUPERSET (gate)")
    for sb in (2, 3, 4, 5):
        print("     %d          %6.2f                     %6.2f                        %6.2f" % (
            sb, 100 * np.mean(acc["current"][sb]), 100 * np.mean(acc["local"][sb]), 100 * np.mean(acc["superset"][sb])))
    print("\n  answers the gate changed: %d   of them fixed: %d   current-correct answers overturned: %d" % (changed, fixed, regress))
    json.dump(dict(gates={k: dict(expr=v["expr"], truth=v["truth"]) for k, v in gates.items()}, edges=EDGES,
                   acc={n: {sb: float(np.mean(v)) for sb, v in a.items()} for n, a in acc.items()},
                   changed=changed, fixed=fixed, regress=regress), open("yale_gate.json", "w"), indent=1)
    print("%.0f s" % (time.time() - t0))
