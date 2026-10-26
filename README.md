# facelaw: exact laws for recognising a face in one dark image

Six laws for face recognition from one dim, noisy phone image. Nuisances cancel **algebraically**, not learned away:
- Law 1, **sensor**;
- Law 2, **light**;
- Law 3, **geometry**;
- Law 4, **identity**;
- Law 5, **decision**;
- Law 6, **limit**: "too dark to tell", never a guess.

**The core idea:** for the right person, every nuisance cancels to exactly zero. For anyone else, a remainder survives.

**How the laws are built:**
- The physics and statistics are written as equations.
- Every integer runtime law is **machine-authored by an integer program-synthesis engine** (not included here), from rows that are true by construction or counted from design data.
- Each authored lane is **proven exhaustively** over its whole input domain (engine evaluator and compiled C).
- At runtime the lanes are plain branchless integer expressions. `lane_eval.py` evaluates them exactly as compiled C, so **no engine is needed to run the laws**.

Full derivations, every check and every result: **[LAWS.md](LAWS.md)**.

## Verdict (honest)

| test | result |
|---|---|
| synthetic world: predicted vs measured for Laws 1–6 | match; false-accept predicted to about 1 point; the Law 6 bound held in 96 of 96 pairs |
| Extended Yale B, standard split (train on 7 near-frontal photos, 38 people) | 100 / 100 / **99.05** / **98.46**: ahead of every method run here and of published VGG-Face. **Not sealed**: variants were explored on it |
| **sealed** CMU PIE illumination (1 training photo, 68 people) | 100%, tied with Gradientfaces, Weberfaces and Tan–Triggs: the test is saturated |
| **sealed** dark CMU PIE (1 / 2 / 5 / 10 photons per pixel) | 64.73 / 94.39 / 99.58 / 99.98. **Not No. 1**: Weberfaces 64.83 / 97.57 / 100 / 100. **Never guesses**: 0 wrong among its answers, but the gate was over-cautious |
| optimal (Bayes, reject option) law with strangers, synthetic | **not calibrated yet**: 4.5–14% wrong among answers, against a promise of 1% or less |

About the sealed hashes:
- The protocols were hashed before their runs, under the engine's private name.
- The copies here replace that one name with "engine", so their hashes differ.
- Both hashes are in `pie_protocol.sha256` and `pie_dark_protocol.sha256`.

## Real time?

**Recognition: yes**, once a face is found and cropped. Measured per probe, one CPU core, Python/numpy:

| task | ours | Weberfaces (the winner) |
|---|---|---|
| dark PIE, 32×32, 68 people | 0.73 ms (dark law) | 0.08 ms |
| Yale B, 192×168, 266 gallery photos | 3.2 ms (local law) | 4.6 ms |
| global light law with cast shadows | ~210 ms (light-direction search) | — |

- The local law costs **34 integer operations per pixel**, with no floating point, no division and no branches. That suits C, SIMD or an FPGA.
- **Detection: not built.** The laws assume a cropped, aligned face (as Yale B and PIE provide).
- **The synthetic world's pose search is far from real time:** it is seconds per probe in Python.
- **A law-based detector does not exist yet.** Learned detectors are excluded by design.

## Ours vs the winner (Weberfaces)

| | Weberfaces | these laws |
|---|---|---|
| what it is | a hand-designed filter: smoothed local brightness ratio, then nearest neighbour | equations for the sensor, light, geometry and noise, plus machine-authored integer lanes |
| cost | fastest at 32×32: 0.08 ms | dark law 0.73 ms (9× slower); local law 3.2 ms (faster than Weberfaces at 192×168); global law ~210 ms |
| arithmetic | floating point | proven integer lanes |
| enrolment | 1 photo | the local and dark laws need 1 photo. The global light law needs at least 3 photos with known light directions (photometric stereo) |
| noise | no model | Law 1's exact noise per pixel. Each vote is weighted by its significance, so the method adapts to the photon count |
| when unsure | always answers | can answer "too dark to tell". Law 6 states the limit; the sealed dark test had 0 wrong answers (but the gate answered rarely) |
| error rates | measured only | derived before measuring. False-accept and false-reject predicted to about 1 point |
| accuracy | wins the sealed dark test | wins Yale B's hard lights when the global and local laws are combined, but that result is not sealed |

## Running

Python 3 with numpy and scipy, and a C compiler for the proofs.
- `python3 -I prove_face.py`: re-prove every face lane, with no data needed.
- `python3 -I check_static.py` and `python3 -I check_noise.py 96`: the synthetic checks.
- `python3 -I pie_test.py` and `python3 -I pie_dark.py`: the sealed PIE tests (data below).
- `yale_*.py`: the Yale B experiments.
- `author_*.py`, `yale_author.py` and `dark_author.py` record how each lane was authored. They need the authoring engine, which is not included.

### Data (not included: research licences)

Put the data under `data/` and run with `python3 -I`.

- **Extended Yale B, cropped**, into `data/yaleb/CroppedYale`
  - The official UCSD links are dead. Use the Internet Archive's copy of `vision.ucsd.edu/extyaleb/CroppedYaleBZip/CroppedYale.zip`.
  - Size: 58,493,820 bytes.
  - SHA-1 (base32): `5Y2B5Y44MGKBHLPLOKF5366WSQYKRQM3`.
  - Please cite Georghiades, Belhumeur, Kriegman 2001, and Lee, Ho, Kriegman 2005.
- **CMU PIE, as processed by Deng Cai**, into `data/pie/lights=27.mat`
  - Internet Archive copy of `www.cad.zju.edu.cn/home/dengcai/Data/PIE/lights=27.mat`.
  - Size: 1,303,519 bytes.
  - SHA-1 (base32): `OULOPDBCGH27BQNIILMJ7LQ5F43Q7M2L`.
  - Please cite Sim, Baker, Bsat 2002.

**Baselines:** Gradientfaces, Weberfaces and Tan–Triggs are reimplementations from the papers' formulas (`yale_baselines.py`), not the authors' code.
