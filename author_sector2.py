"""Lane law for O_ax (it hit the size wall whole): O_gx = the signed low field, then |.| with O_gx declared as ONE op."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, os, sys
from lane_eval import E
try:
    from engine import author, header        # the authoring engine (not included in this repository)
except ImportError:
    author = header = None
from author_face import declare
from author_sector import X, prove
header("LAW L -- O_ax by lane law")
out = json.load(open("lanes_sector.json"))
rows = [(x, (x & 4095) - 2048) for x in X]
e1, i1 = author("O_gx", rows, dict(LIN=("-",), BIT=("&",)), (4095, 2048))
print("        proof over all 16,777,216: %d wrong" % prove(e1, "(x & 4095) - 2048", 0, (1 << 24) - 1))
declare("O_gx", e1)
rows = [(x, abs((x & 4095) - 2048)) for x in X]
e2, i2 = author("O_ax", rows, dict(LIN=("-", "+"), SGN=(">>31",), BIT=("^",)), (), extra=("O_gx",))
print("        proof over all 16,777,216: %d wrong" % prove(e2, "abs((x & 4095) - 2048)", 0, (1 << 24) - 1))
out["O_gx"] = dict(expr=e1, gen=i1.get("gen")); out["O_ax"] = dict(expr=e2, gen=i2.get("gen"), wrong_of_2_24=0)
json.dump(out, open("lanes_sector.json", "w"), indent=1)
