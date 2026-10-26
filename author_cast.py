"""The engine authors LAW 2's cast-shadow test: heights in 1/16 pixel, 12 bits each, x = zs | zr << 12 ->
1 iff the surface sample zs rises above the ray height zr (the light is blocked).  Fresh engine state; proven over all
2^24 height pairs by compiled C."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, os, subprocess, sys, tempfile
from lane_eval import E
try:
    from engine import author, header        # the authoring engine (not included in this repository)
except ImportError:
    author = header = None
import random
R = random.Random(7)
pairs = [(R.randrange(4096), R.randrange(4096)) for _ in range(300)]
pairs += [(a, b) for a in (0, 1, 2047, 2048, 4094, 4095) for b in (0, 1, 2047, 2048, 4094, 4095)]
pairs += [(a, a + d) for a in (0, 100, 2048, 4000) for d in (-1, 0, 1) if 0 <= a + d < 4096]
rows = sorted({(zs | zr << 12, int(zs > zr)) for zs, zr in pairs})
header("CAST SHADOW -- the engine authors the blocking test")
expr, info = author("L2_cast", rows, dict(LIN=("-",), SHF=(">>12",), SGN=(">>31",), BIT=("&",)), (4095, 1))
src = ("#include <stdio.h>\nint main(void){long bad=0;for(int x=0;x<(1<<24);x++){int zs=x&4095,zr=x>>12;"
       "if((%s)!=(zs>zr))bad++;}printf(\"%%ld\\n\",bad);return 0;}" % expr)
d = tempfile.mkdtemp(); open(d + "/p.c", "w").write(src)
subprocess.run(["cc", "-O2", "-fwrapv", "-o", d + "/p", d + "/p.c"], check=True)
bad = int(subprocess.run([d + "/p"], capture_output=True, text=True).stdout)
print("    proof over all 16,777,216 height pairs (compiled C): %d wrong" % bad)
json.dump(dict(name="L2_cast", expr=expr, rows=len(rows), wrong_of_2_24=bad, **{k: v for k, v in info.items() if k not in ("expr", "note")}),
          open("lane_cast.json", "w"), indent=1)
