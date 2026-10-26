"""The engine authors the face recogniser's integer runtime laws.  Fresh engine state per lane;
rows are true BY CONSTRUCTION from the law's equation (sharp intent); lean material per lane.
Every authored lane is then proven exhaustively over its whole input domain (prove_face.py)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, sys, os
from lane_eval import E
try:
    from engine import author, header        # the authoring engine (not included in this repository)
except ImportError:
    author = header = None

# ---- the intents: the derived numbers each law gates on ------------------------------------------------------
B, WHITE = 64, 1023                      # LAW 1: black level (masked pixels), white level (clip)
TAU16 = 37                               # LAW 5: z <= tau = 2.326 (FRR 1%), z in 1/16 units -> z16 <= 37
LAM16 = 206                              # LAW 6: too dark iff lambda < 4 ln(1/(4 eps (1-eps))) = 12.92 (eps = 1%)
RIGID = 17                               # LAW 4: canonical rows r <= 17 have v >= -0.097 > -0.15 (expression-free)

LANES = []


def lane(name, rows, FAM, CON, law, domain, note, extra_from=()):
    LANES.append(dict(name=name, rows=rows, FAM=FAM, CON=CON, law=law, domain=domain, note=note, extra_from=extra_from))


def declare(name, expr):
    """a finished lane becomes ONE operation (lane law): its authored body, inline C, with its exact int32 meaning"""
    from engine import _OPS, s32           # authoring only (not included)
    _OPS[name] = (1, expr.replace("x", "({a})"), lambda a, _b, e=expr: s32(E(e, a)))


# LAW 1 -- the inverse: never round in the sensor law (the division by g happens once, in the final float step)
lane("L1_photon", [(x, x - B) for x in range(0, 1024, 7)] + [(1023, 1023 - B), (64, 0), (63, -1)],
     dict(LIN=("+", "-")), (B,), "L1", (0, 1023), "x - b: the photon estimate in DN units (N^ = this / g)")
lane("L1_sat", [(x, int(x >= WHITE)) for x in list(range(0, 1024, 9)) + [1021, 1022, 1023]],
     dict(LIN=("+", "+1"), SHF=(">>10",)), (), "L1", (0, 1023), "1 iff the pixel is clipped at white: no number")
# LAW 2 (JPEG path) -- the census bit: invariant to ANY strictly increasing tone curve and to exposure
cen = [(a | (b << 8), int(a > b)) for a in range(0, 256, 17) for b in range(0, 256, 17)]
cen += [(a | (b << 8), int(a > b)) for a in (0, 1, 127, 128, 254, 255) for b in (0, 1, 127, 128, 254, 255)]
cen += [(a | (b << 8), int(a > b)) for a in range(0, 256, 23) for b in (a - 1, a, a + 1) if 0 <= b < 256]
lane("L2_census", sorted(set(cen)), dict(LIN=("-",), SHF=(">>8",), SGN=(">>31",), BIT=("&",)), (255, 1),
     "L2", (0, 65535), "s = a | b<<8 (two 8-bit codes) -> [a > b]")
# LAW 4 -- the rigid (expression-free) rows of the canonical face
lane("L4_rigid", [(r, int(r <= RIGID)) for r in range(32)], dict(LIN=("-",), SGN=(">>31",), BIT=("&",)),
     (RIGID + 1, 1), "L4", (0, 31), "canonical row r -> 1 iff every point of it is outside the expression support")
# LAW 5 -- the accept gate
lane("L5_accept", [(z, int(z <= TAU16)) for z in list(range(-1024, 1024, 13)) + [TAU16 - 1, TAU16, TAU16 + 1]],
     dict(LIN=("-",), SGN=(">>31",), BIT=("&",)), (TAU16 + 1, 1), "L5", (-1024, 1023), "z16 -> 1 iff z <= tau")
# LAW 6 -- the limit gate
lane("L6_dark", [(l, int(l <= LAM16)) for l in list(range(0, 4096, 11)) + [LAM16 - 1, LAM16, LAM16 + 1]],
     dict(LIN=("-",), SGN=(">>31",), BIT=("&",)), (LAM16 + 1, 1), "L6", (0, 4095),
     "lambda16 -> 1 iff no method can reach error 1%: answer 'too dark to tell'")


# LAW 5 (open set) -- this METHOD's limit: answer a name only if even the nearest look-alike would be rejected at
# error eps.  u = 256 lambda / lambda_open, lambda_open = the smallest lambda with FAR(lambda) <= eps from Law 5's
# own moments (computed per image from its degrees of freedom).  Too dark for this method iff u < 256.
lane("L5_open", [(u, int(u <= 255)) for u in list(range(0, 4096, 11)) + [254, 255, 256, 257]],
     dict(LIN=("-",), SGN=(">>31",), BIT=("&",)), (256, 1), "L5", (0, 4095),
     "u = 256 lambda / lambda_open -> 1 iff this method cannot reach FAR <= eps: 'too dark to tell'")


# THE DECISION LAW -- state bits: 1 sat (>5% rigid pixels clipped), 2 out (residual outliers: the Lambertian law
# is broken here -- glasses, cast shadow, shine), 4 dark (Law 6 gate OR the L5_open gate: too dark for anyone, or for this method), 8 a1 (best accepted),
# 16 a2 (runner-up accepted too).  -> 0 known (the best candidate), 1 unknown, 2 too dark to tell.
def decide(s):
    sat, out, dark, a1, a2 = (s >> 0) & 1, (s >> 1) & 1, (s >> 2) & 1, (s >> 3) & 1, (s >> 4) & 1
    if sat or out or dark or (a1 and a2):
        return 2
    return 0 if a1 else 1


# lane law: the decision is cut into two lanes; the glue (written once, masks and ORs) is
#   abstain = OUT_any3 | OUT_both,   unknown = ~abstain & ~(s >> 3) & 1,   decision = (abstain << 1) | unknown
lane("OUT_any3", [(s, int((s & 7) != 0)) for s in range(32)], dict(LIN=("neg",), SGN=(">>31",), BIT=("&",)), (7, 1),
     "OUT", (0, 31), "1 iff sat | out | dark")
lane("OUT_both", [(s, (s >> 3) & (s >> 4) & 1) for s in range(32)], dict(SHF=(">>1", ">>3"), BIT=("&",)), (1,),
     "OUT", (0, 31), "1 iff the runner-up is accepted as well as the best (a1 & a2): not separable")


# THE OPTIMAL DECISION LAW (Chow) -- lo16 = 16 ln(P_w / (1 - P_w)) of the most probable hypothesis; confident iff
# P_w >= 1 - eps = 0.99, i.e. ln 99 = 4.595 -> lo16 >= 74.  State: 1 sat, 2 out, 4 conf, 8 winner is the stranger.
# glue: decision = (abst << 1) | (~abst & (s >> 3) & 1)   -> 0 known, 1 unknown, 2 too dark
LO16 = 74
lane("L5_conf", [(l, int(l >= LO16)) for l in list(range(-2048, 2048, 17)) + [LO16 - 1, LO16, LO16 + 1]],
     dict(LIN=("-", "+1"), SGN=(">>31",), BIT=("&",)), (LO16, 1), "L5", (-2048, 2047),
     "lo16 -> 1 iff the posterior of the winner is >= 1 - eps (Chow's optimal reject rule)")
# lane law (the whole abstain lane hit the size wall at ladder 4): first the state's distance from 'answer allowed'
# (sat = out = 0, conf = 1, i.e. s & 7 == 4), then its nonzero test with that finished lane declared as ONE op
lane("OUT_abst_x", [(s, (s & 7) ^ 4) for s in range(16)], dict(BIT=("&",), XOR=("^",)), (7, 4), "OUT", (0, 15),
     "s -> (s & 7) ^ 4: zero exactly when the law may answer")
lane("OUT_abst", [(s, int((s & 1) or (s >> 1) & 1 or not (s >> 2) & 1)) for s in range(16)],
     dict(LIN=("neg",), SGN=(">>31",), BIT=("&",)), (1,), "OUT", (0, 15),
     "1 iff the law must not answer: clipped, Lambertian law broken, or not confident", extra_from=("OUT_abst_x",))


if __name__ == "__main__":
    header("FACE LAWS -- the engine authors every integer runtime lane")
    out = []
    done = {}
    for L in LANES:
        for d in L["extra_from"]:
            declare(d, done[d])
        expr, info = author(L["name"], L["rows"], L["FAM"], L["CON"], extra=tuple(L["extra_from"]))
        done[L["name"]] = expr
        out.append(dict(name=L["name"], law=L["law"], domain=L["domain"], intent=L["note"], expr=expr,
                        **{k: v for k, v in info.items() if k != "expr"}))
    json.dump(out, open("lanes_face.json", "w"), indent=1)
    print("\n%d of %d lanes authored" % (sum(o["expr"] is not None for o in out), len(out)))
