"""Prove every authored Yale lane on its rows (engine _eval and compiled C), rebuild the law from the LANES ONLY,
and score protocol A with it (cross-fitted: each fold judged by the law authored from the other fold's people)."""
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import json, os, pickle, subprocess, sys, tempfile
import numpy as np
from lane_eval import E
L = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "yale_lanes.json"))
bad_e = bad_c = n = 0
for l in L:
    if l["kind"] != "value":
        continue
    for x, y in l["rows"]:
        n += 1; bad_e += E(l["expr"], x) != y
    src = "#include <stdio.h>\nint main(void){int bad=0;%s printf(\"%%d\\n\",bad);return 0;}" % "".join(
        "{int x=%d; if((%s)!=%d) bad++;}" % (x, l["expr"], y) for x, y in l["rows"])
    d = tempfile.mkdtemp(); open(d + "/p.c", "w").write(src)
    subprocess.run(["cc", "-O2", "-fwrapv", "-o", d + "/p", d + "/p.c"], check=True)
    bad_c += int(subprocess.run([d + "/p"], capture_output=True, text=True).stdout)
print("lanes: %d authored, %d rows: _eval wrong %d, compiled C wrong %d" % (sum(l["kind"] == "value" for l in L), n, bad_e, bad_c))


def law_from_lanes(k):
    """the 128-code law of fold k, evaluated ONLY through the engine's expressions (empty halves: impossible codes -> 0)"""
    f = np.zeros(128, int)
    for l in L:
        if l["fold"] != k or l["kind"] != "value":
            continue
        sign, lv = l["t"] & 1, l["t"] >> 1
        for x in range(8):
            f[(x + 8 * l["half"]) | sign << 4 | lv << 5] = E(l["expr"], x) - 8
    return f


D = pickle.load(open(sys.argv[2] if len(sys.argv) > 2 else "yale_law_rem.pkl", "rb"))
T = json.load(open(sys.argv[3] if len(sys.argv) > 3 else "yale_law_tables.json"))
acc, acc_t, diff_codes = {}, {}, 0
for k in (0, 1):
    f = law_from_lanes(k)
    ft = np.array(T["laws"][k])
    seen = np.array(T["seen"][k])
    diff_codes += int((f[seen] != ft[seen]).sum())
    for p, sb, h in D["rem"][k]:
        acc.setdefault(sb, []).append(max(h, key=lambda q: int(h[q] @ f)) == p)
        acc_t.setdefault(sb, []).append(max(h, key=lambda q: int(h[q] @ ft)) == p)
print("engine law vs data table on every code the design data contains: %d codes differ" % diff_codes)
print("\nPROTOCOL A, 38 people (each judged by the law authored from the other half of the people) -- recognition %")
print("  subset   machine-authored data law   (table)   plain residual   VGG-Face FC7   VGG FC7 + preprocessing")
plain = {2: 100.0, 3: 100.0, 4: 91.83, 5: 65.13}; vgg = {2: 100, 3: 92.32, 4: 52.44, 5: 18.28}; vggp = {4: 75.56, 5: 76.32}
for sb in sorted(acc):
    print("     %d          %6.2f              (%6.2f)        %6.2f          %6.2f          %s" % (
        sb, 100 * np.mean(acc[sb]), 100 * np.mean(acc_t[sb]), plain[sb], vgg[sb], ("%.2f" % vggp[sb]) if sb in vggp else "-"))
