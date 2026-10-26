"""LAW L (local cancellation, the other side): the direction sector of the brightness gradient.  Light scales a gradient,
it does not turn it where albedo edges dominate -- so the sector cancels light locally, without any 3D shape.
x = gx' | gy' << 12, gx' = clip(16 gx) + 2048 (12 bits each).  The engine authors, fresh engine state per lane:
    O_sy  [gy < 0]        O_sx  [gx < 0]        O_ax  |gx|        O_ay  |gy|
    [|gy| > |gx|] = the engine's proven L2_cast lane on  O_ax | O_ay << 12  ... reused, not re-authored
    O_perm  o = 4 sy + 2 sx + d  ->  the sector's order around the circle (0..7, counter-clockwise from +x)
Every lane proven over its whole domain by compiled C (2^24 packed gradients, 8 codes)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, os, subprocess, sys, tempfile, random
from lane_eval import E
try:
    from engine import author, header        # the authoring engine (not included in this repository)
except ImportError:
    author = header = None

R = random.Random(11)
pts = [(R.randrange(4096), R.randrange(4096)) for _ in range(400)]
pts += [(a, b) for a in (0, 1, 2046, 2047, 2048, 2049, 4094, 4095) for b in (0, 1, 2047, 2048, 2049, 4095)]
X = sorted({a | b << 12 for a, b in pts})
LAW = {"O_sy": lambda x: int((x >> 12) < 2048), "O_sx": lambda x: int((x & 4095) < 2048),
       "O_ax": lambda x: abs((x & 4095) - 2048), "O_ay": lambda x: abs((x >> 12) - 2048)}
CLAW = {"O_sy": "(x >> 12) < 2048", "O_sx": "(x & 4095) < 2048", "O_ax": "abs((x & 4095) - 2048)", "O_ay": "abs((x >> 12) - 2048)"}
PERM = [0, 1, 3, 2, 7, 6, 4, 5]     # o = 4 sy + 2 sx + d  ->  angular order (from the quadrant geometry)
FAMS = {"O_sy": (dict(SHF=(">>12", ">>23"), LIN=("-",), BIT=("&",)), (1, 2048)),
        "O_sx": (dict(SHF=(">>11",), LIN=("-",), BIT=("&",)), (1, 2048)),
        "O_ax": (dict(LIN=("-", "+"), SGN=(">>31",), BIT=("&", "^")), (4095, 2048)),
        "O_ay": (dict(LIN=("-", "+"), SGN=(">>31",), SHF=(">>12",), BIT=("^",)), (2048,))}


def prove(expr, claw, lo, hi):
    src = ("#include <stdio.h>\n#include <stdlib.h>\nint main(void){long bad=0;for(int x=%d;x<=%d;x++){if((%s)!=(%s))bad++;}"
           "printf(\"%%ld\\n\",bad);return 0;}" % (lo, hi, expr, claw))
    d = tempfile.mkdtemp(); open(d + "/p.c", "w").write(src)
    subprocess.run(["cc", "-O2", "-fwrapv", "-o", d + "/p", d + "/p.c"], check=True)
    return int(subprocess.run([d + "/p"], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    header("LAW L -- the engine authors the gradient-direction sector")
    out = {}
    for name in ("O_sy", "O_sx", "O_ax", "O_ay"):
        rows = [(x, LAW[name](x)) for x in X]
        FAM, CON = FAMS[name]
        expr, info = author(name, rows, FAM, CON)
        bad = prove(expr, CLAW[name], 0, (1 << 24) - 1) if expr else None
        print("        proof over all 16,777,216 packed gradients: %s wrong" % bad)
        out[name] = dict(expr=expr, wrong_of_2_24=bad, gen=info.get("gen"))
    rows = [(o, PERM[o]) for o in range(8)]
    expr, info = author("O_perm", rows, dict(BIT=("&", "^", "|"), SHF=(">>1", ">>2", "<<1"), LIN=("-", "+")), (1, 2, 3, 7))
    bad = sum(E(expr, o) != PERM[o] for o in range(8)) if expr else None
    print("        proof over all 8 codes: %s wrong" % bad)
    out["O_perm"] = dict(expr=expr, wrong_of_8=bad, gen=info.get("gen"))
    json.dump(out, open("lanes_sector.json", "w"), indent=1)
