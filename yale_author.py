"""The engine authors the Yale B decision law from the data table (yale_law_tables.json), lane by lane:
for each fold, for each (level, sign) table, for each half of the 16 remainder bins: x = bin & 7 -> the law's value + 8.
A lane the engine cannot reach whole is cut into its 4 output bits (lane law).  Physically impossible codes (never
seen in the design data) are don't-cares and are not rows.  Fresh engine state per lane."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys
from lane_eval import E
try:
    from engine import author, header        # the authoring engine (not included in this repository)
except ImportError:
    author = header = None

FAM_VAL = dict(LIN=("+", "-", "+1", "-1", "neg"), SHF=(">>1", ">>2", "<<1"), BIT=("&", "|", "^", "~"))
FAM_BIT = dict(SHF=(">>1", ">>2"), BIT=("&", "|", "^", "~"))

if __name__ == "__main__":
    T = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "yale_law_tables.json"))
    header("YALE B DECISION LAW -- the engine authors the data's own log-likelihood table")
    out = []
    for k in (0, 1):
        f, seen = T["laws"][k], T["seen"][k]
        for t in range(8):                                       # t = sign | level << 1
            sign, lv = t & 1, t >> 1
            for half in (0, 1):
                rows = [(x, f[(x + 8 * half) | sign << 4 | lv << 5] + 8) for x in range(8)
                        if seen[(x + 8 * half) | sign << 4 | lv << 5]]
                name = "Y%d_L%d%s_h%d" % (k, lv, "-+"[sign == 0], half)
                if not rows:
                    out.append(dict(name=name, fold=k, t=t, half=half, kind="empty", rows=[]))
                    continue
                if len({y for _, y in rows}) == 1:
                    consts = (rows[0][1],)
                else:
                    consts = tuple(sorted({y for _, y in rows} | {1, 2, 4, 8}))[:6]
                expr, info = author(name, rows, FAM_VAL, consts, quiet=False)
                if expr:
                    out.append(dict(name=name, fold=k, t=t, half=half, kind="value", expr=expr, rows=rows,
                                    gen=info["gen"], secs=info["secs"]))
                    continue
                bits = []
                for b in range(4):                               # lane law: one lane per output bit
                    brows = [(x, (y >> b) & 1) for x, y in rows]
                    e, i2 = author("%s_b%d" % (name, b), brows, FAM_BIT, (1,), quiet=False)
                    bits.append(e)
                out.append(dict(name=name, fold=k, t=t, half=half, kind="bits", bits=bits, rows=rows))
    json.dump(out, open(sys.argv[2] if len(sys.argv) > 2 else "yale_lanes.json", "w"), indent=1)
    n_val = sum(o["kind"] == "value" for o in out); n_bit = sum(o["kind"] == "bits" for o in out)
    miss = sum(1 for o in out if o["kind"] == "bits" for b in o["bits"] if b is None)
    print("\n%d tables-halves authored whole, %d cut into bit lanes (%d bit lanes without an expression)" % (n_val, n_bit, miss))
