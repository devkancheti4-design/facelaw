"""The engine authors the frozen dark law (dark_design.json): one lane per (scale k, gallery-flat bit, significance c)
over the sector gap d = 0..4 -> f + 128; and the gate over the margin bin -> answer.  Fresh engine state per lane;
every lane proven on all its rows (engine _eval + compiled C); the runtime table is then
rebuilt from the lanes ONLY and must equal the data table on every code the design data contains."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, os, subprocess, sys, tempfile
import numpy as np
from lane_eval import E
try:
    from engine import author, header        # the authoring engine (not included in this repository)
except ImportError:
    author = header = None

D = json.load(open("dark_design.json"))
f, seen = D["f"], D["seen"]
FAM = dict(LIN=("+", "-", "+1", "-1", "neg"), SHF=(">>1", "<<1", "<<2", "<<3"), BIT=("&", "|", "^"))


def cprove(expr, rows):
    src = "#include <stdio.h>\nint main(void){int bad=0;%s printf(\"%%d\\n\",bad);return 0;}" % "".join(
        "{int x=%d; if((%s)!=%d) bad++;}" % (x, expr, y) for x, y in rows)
    d = tempfile.mkdtemp(); open(d + "/p.c", "w").write(src)
    subprocess.run(["cc", "-O2", "-fwrapv", "-o", d + "/p", d + "/p.c"], check=True)
    return int(subprocess.run([d + "/p"], capture_output=True, text=True).stdout)


if __name__ == "__main__":
    header("DARK LAW -- the engine authors the frozen design table and its gate")
    lanes, nrows, bad = [], 0, 0
    for k in range(3):
        for flat in (0, 1):
            for c in range(8):
                rows = [(d, f[d | c << 3 | flat << 6 | k << 7] + 128) for d in range(5) if seen[d | c << 3 | flat << 6 | k << 7]]
                if not rows:
                    continue
                vals = sorted({y for _, y in rows})
                consts = tuple(vals[:3]) + (1, 2) if len(vals) > 1 else (vals[0],)
                expr, info = author("D%d%d%d" % (k, flat, c), rows, FAM, consts)
                if expr is None:
                    raise SystemExit("lane D%d%d%d: no expression -- cut it" % (k, flat, c))
                bad += sum(E(expr, x) != y for x, y in rows) + cprove(expr, rows); nrows += len(rows)
                lanes.append(dict(k=k, flat=flat, c=c, expr=expr, rows=rows, gen=info["gen"]))
    gate_rows = [(b, g) for b, g in enumerate(D["gate"])]
    gexpr, ginfo = author("D_gate", gate_rows, dict(LIN=("-",), SGN=(">>31",), SHF=(">>1", ">>2"), BIT=("&",)), (1, 6))
    bad += sum(E(gexpr, b) != g for b, g in gate_rows) + cprove(gexpr, gate_rows)
    # the runtime law from the lanes only
    F = np.zeros(512, int)
    for l in lanes:
        for d in range(5):
            F[d | l["c"] << 3 | l["flat"] << 6 | l["k"] << 7] = E(l["expr"], d) - 128
    diff = int(sum(F[i] != f[i] for i in range(512) if seen[i]))
    print("\n%d lanes + gate authored; %d rows; wrong (_eval + C): %d; codes where the lane law differs from the data table: %d"
          % (len(lanes), nrows + len(gate_rows), bad, diff))
    json.dump(dict(lanes=lanes, gate=gexpr, F=F.tolist(), wrong=bad, differ=diff), open("dark_lanes.json", "w"), indent=1)
