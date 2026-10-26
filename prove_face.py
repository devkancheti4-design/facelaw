"""Exhaustive proof of every authored lane over its whole domain, twice: the engine's own _eval, and compiled C
(cc -O2) against the law written independently in C.  Then the census claims: any strictly increasing tone curve
cancels exactly; quantised curves leave a counted residue; and the decision glue equals the decision table."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, os, subprocess, sys, tempfile
from lane_eval import E
from author_face import LANES, decide, B, WHITE, TAU16, LAM16, RIGID

L = {l["name"]: l for l in json.load(open("lanes_face.json"))}
LAW = {"L1_photon": lambda x: x - B, "L1_sat": lambda x: int(x >= WHITE), "L2_census": lambda s: int((s & 255) > (s >> 8)),
       "L4_rigid": lambda r: int(r <= RIGID), "L5_accept": lambda z: int(z <= TAU16), "L6_dark": lambda l: int(l <= LAM16), "L5_open": lambda u: int(u <= 255), "L5_conf": lambda l: int(l >= 74), "OUT_abst": lambda s: int((s & 7) != 4), "OUT_abst_x": lambda s: (s & 7) ^ 4,
       "OUT_any3": lambda s: int((s & 7) != 0), "OUT_both": lambda s: (s >> 3) & (s >> 4) & 1}
LAWC = {"L1_photon": "x - 64", "L1_sat": "x >= 1023", "L2_census": "(x & 255) > (x >> 8)", "L4_rigid": "x <= 17",
        "L5_accept": "x <= 37", "L6_dark": "x <= 206", "L5_open": "x <= 255", "L5_conf": "x >= 74", "OUT_abst": "(x & 7) != 4", "OUT_abst_x": "(x & 7) ^ 4", "OUT_any3": "(x & 7) != 0", "OUT_both": "(x >> 3) & (x >> 4) & 1"}
rep = []
print("lane          domain              inputs   _eval wrong   C wrong   expression")
for name, l in L.items():
    lo, hi = l["domain"]
    bad = sum(E(l["expr"], x) != LAW[name](x) for x in range(lo, hi + 1))
    src = "#include <stdio.h>\nint main(void){long bad=0;for(int x=%d;x<=%d;x++){int a=(%s);int b=(%s);if(a!=b)bad++;}" \
          "printf(\"%%ld\\n\",bad);return 0;}\n" % (lo, hi, l["expr"], LAWC[name])
    d = tempfile.mkdtemp(); open(d + "/p.c", "w").write(src)
    subprocess.run(["cc", "-O2", "-fwrapv", "-o", d + "/p", d + "/p.c"], check=True)
    cbad = int(subprocess.run([d + "/p"], capture_output=True, text=True).stdout)
    print("%-12s  [%6d, %6d]  %8d   %11d   %7d   %s" % (name, lo, hi, hi - lo + 1, bad, cbad, l["expr"]))
    rep.append(dict(name=name, domain=[lo, hi], inputs=hi - lo + 1, eval_wrong=bad, c_wrong=cbad, expr=l["expr"]))

cen = lambda a, b: E(L["L2_census"]["expr"], a | (b << 8))
print("\nLAW 2 census -- tone and exposure cancellation: codes c = Q(h(k Phi)), 400,000 random pairs per curve")
# the order of exposures survives ANY strictly increasing tone curve h and any exposure k > 0; the only residue is
# quantisation: when two exposures land on the same 8-bit code (a tie, incl. clipping) the order is lost
import numpy as np
R = np.random.default_rng(3)
Pa, Pb = R.uniform(0, 1, 400_000), R.uniform(0, 1, 400_000)
knots = np.sort(R.uniform(0, 1, 9)); spline = lambda t: np.interp(t, np.linspace(0, 1, 11), np.concatenate([[0], knots, [1]]))
curves = {"gamma 1/2.2": lambda t: t ** (1 / 2.2), "S-curve": lambda t: 1 / (1 + np.exp(-10 * (t - 0.4))),
          "random monotone": spline, "phone-like (log + shoulder)": lambda t: np.log1p(40 * t) / np.log1p(40)}
for k_name, h in curves.items():
    for k in (0.25, 1.0, 3.0):
        ca = np.clip(np.rint(255 * h(np.minimum(1, k * Pa))), 0, 255).astype(int)
        cb = np.clip(np.rint(255 * h(np.minimum(1, k * Pb))), 0, 255).astype(int)
        c = np.array([cen(int(a), int(b)) for a, b in zip(ca[:40000], cb[:40000])])
        c = np.concatenate([c, (ca[40000:] > cb[40000:]).astype(int)])   # the lane == [a > b] is proven above
        t = (Pa > Pb).astype(int)
        flips = c != t
        nontie = int((flips & (ca != cb)).sum())
        print("   %-28s k = %4.2f   order lost on %6.3f%% of pairs, every one a tie: %s" % (
            k_name, k, 100 * flips.mean(), "yes" if nontie == 0 else "NO (%d)" % nontie))
        rep.append(dict(curve=k_name, k=k, lost=float(flips.mean()), nontie=nontie))

print("\nDECISION -- glue over the two OUT lanes vs the decision table, all 32 states")
A = lambda s: E(L["OUT_any3"]["expr"], s) | E(L["OUT_both"]["expr"], s)
D = lambda s: (A(s) << 1) | (~A(s) & ~(s >> 3) & 1)
wrong = [s for s in range(32) if D(s) != decide(s)]
print("   wrong states: %d of 32" % len(wrong))
rep.append(dict(decision_wrong=len(wrong)))
json.dump(rep, open("proof_face.json", "w"), indent=1)
