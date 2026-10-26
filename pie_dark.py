"""The sealed dark PIE test (pie_dark_protocol.txt, hashed before running)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys
import numpy as np
import scipy.io as sio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dark_law as DL
from lane_eval import E
from yale_gate import sectors
from yale_baselines import gradientface, weberface, tantriggs, angdist, l1

DL.SHAPE, DL.SCALE_OF = (32, 32), None
L = json.load(open("dark_lanes.json")); F = np.array(L["F"]); GATE = L["gate"]
X = sio.loadmat("data/pie/lights=27.mat")["fea"].astype(float).reshape(-1, 32, 32).transpose(0, 2, 1).reshape(68, 21, 32, 32)
TRAIN, NB = 10, (1, 2, 5, 10)
gal = [X[p, TRAIN] for p in range(68)]
GF = [DL.gallery_feats(g) for g in gal]; gs = np.array([g[0] for g in GF]); gf = np.array([g[1] for g in GF])
GL = np.array([sectors(g) for g in gal], np.int16)
GB = {k: np.array([fn(g) for g in gal]) for k, fn in (("Gradientfaces", gradientface), ("Weberfaces", weberface), ("Tan-Triggs", tantriggs))}
gmean = float(np.mean([g.mean() for g in gal]))
R = np.random.default_rng(31337)
names = ["OURS (engine dark law)", "Gradientfaces", "Weberfaces", "Tan-Triggs", "local law (clean, frozen)"]
acc = {n: {N: [] for N in NB} for n in names}
ans = {N: [0, 0] for N in NB}                                     # answered, wrong among answered
for p in range(68):
    for k in range(21):
        if k == TRAIN:
            continue
        for N in NB:
            for draw in range(3):
                Nh = DL.dark(X[p, k], N, R)
                ps, pc = DL.probe_feats(Nh)
                sc = F[DL.codes(ps, pc, gs, gf)].sum(1)
                o = np.argsort(-sc)
                acc[names[0]][N].append(o[0] == p)
                if E(GATE, DL.mbin(sc[o[0]] - sc[o[1]])) == 1:
                    ans[N][0] += 1; ans[N][1] += o[0] != p
                x = np.maximum(Nh, 0) * (gmean / N)
                for nm, fn, dist in (("Gradientfaces", gradientface, angdist), ("Weberfaces", weberface, l1), ("Tan-Triggs", tantriggs, l1)):
                    acc[nm][N].append(int(np.argmin(dist(fn(x), GB[nm]))) == p)
                s = sectors(Nh).astype(np.int16)
                d = np.abs(GL - s[None]); d = np.minimum(d, 8 - d).sum(1)
                acc[names[4]][N].append(int(np.argmin(d)) == p)
n = len(acc[names[0]][1])
print("SEALED DARK PIE TEST -- 68 people, 1 clean frontal enrolment, %d probes per photon level -- forced-choice recognition %%" % n)
print("  %-28s" % "photons per pixel" + "".join("%9d" % N for N in NB))
for nm in names:
    print("  %-28s" % nm + "".join("%9.2f" % (100 * np.mean(acc[nm][N])) for N in NB))
print("\n  OURS with the gate: answered " + "  ".join("N=%d %5.1f%%" % (N, 100 * ans[N][0] / n) for N in NB))
print("                     wrong among answered " + "  ".join("N=%d %s" % (N, ("%.2f%%" % (100 * ans[N][1] / ans[N][0])) if ans[N][0] else "-") for N in NB))
first = all(np.mean(acc[names[0]][N]) >= np.mean(acc[nm][N]) for nm in names[1:] for N in NB)
never = all((ans[N][1] <= 0.01 * ans[N][0]) for N in NB)
print("\nVERDICT (rules fixed in pie_dark_protocol.txt):")
print("   No. 1 in the dark:  %s" % ("YES" if first else "NO"))
print("   Never guesses:      %s" % ("YES" if never else "NO"))
json.dump(dict(acc={nm: {N: float(np.mean(v[N])) for N in NB} for nm, v in acc.items()},
               answered={N: ans[N][0] / n for N in NB}, wrong_among_answered={N: (ans[N][1] / ans[N][0] if ans[N][0] else None) for N in NB},
               first=bool(first), never=bool(never)), open("pie_dark_results.json", "w"), indent=1)
