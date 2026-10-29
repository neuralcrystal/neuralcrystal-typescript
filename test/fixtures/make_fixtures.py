"""Parity fixtures for the TypeScript engine: tiny crystals (64² grid) run through the training code's Crystal — with the inputs
and every intermediate result saved, so the engine's tests check it against the trainer term for term.

    PYTHONPATH=<the training code> python3 test/fixtures/make_fixtures.py

This script needs the training code (its `nc` package), which isn't public. The public `pip install neuralcrystal` runs the published
crystals forward but isn't enough here: it refuses the fixed surface kinds (lens, lens array, grating), per-section roles, the board
ghost and the residual band that these fixtures exercise, it doesn't expose the per-pass internals saved below (the camera pictures
between passes, each pass on its own), and the MNIST kit fixture (E) uses the trainer's own detector reading (det_masks / det_terms).
The real-crystal fixtures (D, E) are made only when NC_CRYSTALS points at a folder holding mnist-padrot.safetensors; E also needs
NC_DATA pointing at the original data set (a folder holding qmnist/test-000.json).

Each fixture is <name>.safetensors next to this file: __metadata__.neuralcrystal is the recipe (maps stripped, as a v2 crystal),
plane.<i>.phi[.<k>] the maps, and expect.* the results. Seconds of CPU; nothing large.
"""
import json, os, sys
import numpy as np
import torch
from nc.crystal import Crystal, st_write

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

def maps(n, count):
    return [(rng.standard_normal((n, n)) * 3).astype(np.float32) for _ in range(count)]

def base(**kw):
    m = dict(format="neuralcrystal.model.v2", lam_um=0.617, n_medium=1.4607, N=64, dx_um=1, inGap_mm=0.05, ext=75, outScale=1,
             featUm=1, wholeGlass=True, winUm=48, solveMirror=False, sideBlack=True, dmdMirrors=48,
             imager=dict(g=3.95, b=0.01, gm=1.4, flip=False, inv=False, sen="cell8", mode="fixed"), overlay=dict(on=False))
    m.update(kw); return m

def build(name, model, secmaps, F, Fr):
    """secmaps[i] = list of section maps for plane i (or None). F: the input frame, Fr: the readout size."""
    tensors = {}
    for i, sm in enumerate(secmaps):
        if not sm: continue
        model["planes"][i]["phi"] = sm[0]; model["planes"][i]["n"] = int(sm[0].shape[0]); tensors[f"plane.{i}.phi"] = sm[0]
        if len(sm) > 1:
            model["planes"][i]["phis"] = sm
            for k in range(1, len(sm)): tensors[f"plane.{i}.phi.{k}"] = sm[k]
    c = Crystal(json.loads(json.dumps(model, default=lambda a: None)) | {"planes": model["planes"]}, device="cpu")
    frame = (rng.random((1, F, F)) < 0.3).astype(np.float32)
    fr = torch.tensor(frame)
    with torch.no_grad():
        E, E0 = c.run(fr)
        tensors["expect.frame"] = frame[0]
        for k, h in enumerate(c._hids): tensors[f"expect.hidden.{k}"] = h[0].numpy().astype(np.float32)
        tensors["expect.exit.re"] = E[0].real.numpy().astype(np.float32); tensors["expect.exit.im"] = E[0].imag.numpy().astype(np.float32)
        tensors["expect.entry.re"] = E0[0].real.numpy().astype(np.float32); tensors["expect.entry.im"] = E0[0].imag.numpy().astype(np.float32)
        # each pass on its own, from Python's own input picture: section-level parity without dither-level knock-on
        ins = [fr] + [h for h in c._hids]
        for k in range(c.hops):
            Ek = c.forward(c.dmd(ins[k], hidden=k > 0), c.q_pass[k] if c.q_pass else k)
            tensors[f"expect.pass.{k}.re"] = Ek[0].real.numpy().astype(np.float32); tensors[f"expect.pass.{k}.im"] = Ek[0].imag.numpy().astype(np.float32)
        v = c.readout(E, Fr); tensors["expect.readout"] = v[0].numpy().astype(np.float32)
        tensors["expect.picture"] = c.imager(v)[0].numpy().astype(np.float32)
    for p in model["planes"]: p.pop("phi", None); p.pop("phis", None)
    meta = {"format": "neuralcrystal.model.v2", "neuralcrystal": json.dumps(model), "fixture": json.dumps({"F": F, "Fr": Fr})}
    st_write(os.path.join(HERE, f"{name}.safetensors"), meta, tensors)
    print(f"{name}: hops {c.hops}, side {c.side}, zmax {c.z_max():.1f} um, tensors {len(tensors)}")

# A: a quad crystal like the launch ones — learned planes, absorbing rim, auto-exposure, residual, unrotation
qa = base(absorbRim=True, planes=[dict(kind="learned", dist_mm=d) for d in (0.02, 0.02, 0.04, 0.03)],
          quad=dict(n=3, mir=4, levels=17, leak=0.1, gains=[0.7, 2.6], norm=dict(t=1.0, s=4.0), res=0.25, unrot=True, ghost=0.0, band=0.0))
build("quad", qa, [[m for m in maps(32, 3)] for _ in range(4)], F=16, Fr=8)

# B: one pass — fixed kinds, the flip and polarity, a relay-k sensor (turned readout), hops split at z_max
sb = base(zMaxSplit=True, inGap_mm=0.3, imager=dict(g=2.0, b=0.05, gm=1.0, flip=True, inv=True, sen=4, mode="fixed"),
          planes=[dict(kind="learned", dist_mm=0.3), dict(kind="lens", param=0.4, dist_mm=0.2),
                  dict(kind="lensarr", param=0.3, dist_mm=0.1, arr=dict(nx=2, ny=2, pitch_mm=0.02, cells=[0.3, None, 0.25, 0.3])),
                  dict(kind="learned", dist_mm=0.05), dict(kind="grating", param=8, dist_mm=0.05, off=True)])
build("single", sb, [maps(64, 1), None, None, maps(16, 1), None], F=12, Fr=6)

# C: per-section roles on shared wafers, merged glass, the fitted gains, the board ghost and the text band
rc = base(planes=[dict(kind="learned", kinds=["learned", "glass", "lens"], params=[0, 0, 0.5], dist_mm=0.03),
                  dict(kind="learned", kinds=["glass", "learned", "learned"], dist_mm=0.02),
                  dict(kind="lens", kinds=["lens", "lens", "glass"], params=[0.3, 0.6, 0], dist_mm=0.04)],
          quad=dict(n=3, mir=4, gains=[1.5, 3.0], ghost=1.0, res=0.0, unrot=False, band=2.0, bandY0=12))
build("roles", rc, [maps(32, 3), maps(32, 3), None], F=16, Fr=8)

# D: a real trained crystal, when NC_CRYSTALS points at it: mnist-padrot, 5 planes at 512². Only the expected results are stored;
# the test reads the crystal itself.
REAL = os.path.join(os.environ.get("NC_CRYSTALS", ""), "mnist-padrot.safetensors")
if os.environ.get("NC_CRYSTALS") and os.path.exists(REAL):
    from nc.crystal import load
    c = load(REAL, device="cpu"); F = 28
    frame = (rng.random((1, F, F)) < 0.25).astype(np.float32)
    with torch.no_grad():
        E, _ = c.run(torch.tensor(frame)); v = c.readout(E, F)
    st_write(os.path.join(HERE, "mnist-real.safetensors"), {"fixture": json.dumps({"F": F, "Fr": F, "crystal": "crystals/mnist-padrot.safetensors"})},
             {"expect.frame": frame[0], "expect.exit.re": E[0].real.numpy().astype(np.float32), "expect.exit.im": E[0].imag.numpy().astype(np.float32),
              "expect.readout": v[0].numpy().astype(np.float32), "expect.picture": c.imager(v)[0].numpy().astype(np.float32)})
    print(f"mnist-real: N {c.N}, side {c.side}, planes {len(c.planes)}")

# E: the MNIST kit against the trainer's own reading (det_masks / det_terms) on real test hands, with the real crystal
SHARD = os.path.join(os.environ.get("NC_DATA", ""), "qmnist/test-000.json")
if os.environ.get("NC_CRYSTALS") and os.environ.get("NC_DATA") and os.path.exists(REAL) and os.path.exists(SHARD):
    import base64
    from nc.train_mnist import det_masks, det_terms, FRAME, PAD
    c = load(REAL, device="cpu"); D = det_masks(c)
    sh = json.load(open(SHARD))
    px = np.frombuffer(base64.b64decode(sh["data"]), dtype=np.uint8).reshape(sh["n"], sh["h"], sh["w"])
    idx = list(range(0, 240, 20)); fr = np.zeros((len(idx), FRAME, FRAME), np.uint8); fr[:, PAD:PAD + 28, PAD:PAD + 28] = px[idx]
    lab = torch.tensor([sh["labels"][i] for i in idx])
    with torch.no_grad():
        f = torch.tensor(fr).float() / 255.0; E0 = c.dmd(f); Pin = (E0.real ** 2 + E0.imag ** 2).sum((1, 2)).reshape(-1, 1); E = c.forward(E0)
        out, pred, hits, eff, pp = det_terms(c, E, Pin, lab, D, "sce")
    st_write(os.path.join(HERE, "mnist-kit.safetensors"), {"fixture": json.dumps({"F": FRAME, "Fr": FRAME, "idx": idx, "labels": lab.tolist(), "pred": pred.tolist()})},
             {"expect.p": pp.numpy().astype(np.float32), "expect.eff": eff.numpy().astype(np.float32)})
    print(f"mnist-kit: {len(idx)} test hands, trainer reads {int(hits.sum())}/{len(idx)} right")

# F: the DMD relay's pupil (dmdNA, the Craftax e2e crystal's 0.15): a quad with lens planes, every launched picture low-passed
pf = base(absorbRim=True, dmdNA=0.15, planes=[dict(kind="lens", param=0.3, dist_mm=0.02), dict(kind="learned", dist_mm=0.02), dict(kind="learned", dist_mm=0.03)],
          quad=dict(n=3, mir=4, levels=17, leak=0.1, gains=[0.7, 2.6], norm=dict(t=1.0, s=4.0), res=0.25, unrot=True, ghost=0.0, band=0.0))
build("pupil", pf, [None, maps(32, 3), maps(32, 3)], F=16, Fr=8)
