# G1 largebox sim2sim audit — 2026-11-08

Checkpoint: `g1-29dof-wbt-largebox-omniretarget-fastsac-b/.../26-11-08_06-30-38-120441/checkpoints/model_0028000.pt`.

## Findings

- Both initial actor observations match source play within 3e-7. ONNX parity error is 5.96e-6. Joint order, action scale, PD gains and history use the same artifact in both backends.
- Fixed an actual MuJoCo timing mismatch: `mj_step` integrates qpos/qvel after computing derived pose/sensors. Refresh with `mj_forward` at the control boundary so joint values and IMU refer to the same instant. This is independently regression-tested; it does NOT solve transfer failure.
- The two worlds contain the same robot, floor and dynamic box. All corresponding compiled body masses and COM positions agree within numerical tolerance. Box mass is 0.1 kg, diagonal inertia 0.002 kg m².
- The box collision and visual declarations both reference `largebox.obj`, scale 1 and zero local origin. Source, compiled MuJoCo vertices transformed back into body coordinates, and MotrixSim source-geometry AABB share bounds: min (-0.23408277, -0.23008044, -0.19819215), max (0.23707144, 0.22864969, 0.20970333) m. This rules out an observed scale or placement discrepancy; matching bounds alone does not prove identical contact surfaces/points.
- MuJoCo uses the mesh convex hull for collision, not arbitrary visible surface triangles. Installed MotrixSim reports Shape.Mesh and Default cooking, documented by its SDK as single-convex with AABB fallback; no SDF or explicit decomposition is selected. The internal native cooked hull was not independently extracted, so exact hull/contact-point equality is not claimed.
- URDF `<contact>` values (lateral friction 0.9, rolling friction 0.5, stiffness 30000, damping 1000) are not the effective compiled parameters here. Both compiled box geoms have friction [1, 0.005, 0.0001], condim 3, collision mask 1/1. The root-level `<dynamics>` tag does not supply effective joint dynamics. Native material reports stiffness 2500/damping 100 and the default impedance; MuJoCo uses solref [0.02,1] and solimp [0.9,0.95,0.001,0.5,2]. Changing MuJoCo to direct solref [-2500,-100] did not restore transfer.
- Native uses PGS, friction_model=2 and tolerance 0.001. MuJoCo defaults to Newton/pyramidal friction and different tolerance; equal iteration counts do not imply equivalent contact solves. Alternative solvers/integrators/contacts were diagnostic overrides only, not committed tuning.

## Controlled rollouts after timestamp correction

All use the same ONNX policy, frame-zero state, 0.02 s control period, and 0.005 s physics step unless explicitly noted. Every MuJoCo case below terminates with bad_ref_ori.

| Case | Policy steps |
|---|---:|
| MotrixSim baseline | 325, completed |
| MuJoCo baseline | 91 |
| Box collision disabled | 48 |
| Box replaced by its axis-aligned bounding cuboid | 87 |
| Joint friction disabled | 89 |
| Robot self-collision disabled | 87 |
| Elliptic friction, 100 iterations | 88 |
| Contact time constant 0.005, 100 iterations | 97 |
| Implicitfast, 100 iterations | 101 |
| RK4, 100 iterations | 94 |
| PGS, 3 iterations | 86 |
| PGS, 100 iterations | 83 |
| PGS + elliptic, 100 iterations | 81 |

The contact-disabled fixed-command probe has only ~0.0016 rad maximum joint-position disagreement at the first 20 ms interval, versus progressively large stance failure with closed-loop contact. This supports contact/solver/feedback sensitivity as a major contributor, not a proof of a single root cause. Removing the box changes the trained task and is not a sufficient test to exonerate box contacts. Replacing the mesh also changes its contact surface and is not a faithful fix.

**Conclusion:** No gross box asset scale/pose mismatch was found. A real state-timing defect was corrected. Correct transfer has NOT been established in MuJoCo; the remaining gap involves coupled contact dynamics and policy feedback. Do not claim all physics is aligned or that changing a single solver option fixes it. MotrixSim completes the playback budget; this alone is not a measured object-carry success score.

See `trajectory.svg`, `models.txt`, `geometry.txt` and the JSON traces. The scripts are one-off acceptance tools, not permanent unit tests. Run them from the worktree using `.venv/bin/python`.

## Replay and export

```bash
source .venv/bin/activate
artifact=artifacts/g1-29dof-wbt-largebox-omniretarget-fastsac-b-model_0028000.deploy
mxlab deploy task=g1-largebox/sim artifact="$artifact" runtime.backend=motrixsim loop=true
mxlab deploy task=g1-largebox/sim artifact="$artifact" runtime.backend=mujoco loop=true
```

Replay restores the world, reference clock and action history after completion or task termination, retaining the viewer. Close the viewer or Ctrl+C to exit. Hardware replay is rejected. Two consecutive episodes have identical trace hashes in both backend regression tests; CLI loop and Ctrl+C also passed on both backends.

For another a–h checkpoint:

```bash
ckpt=/absolute/path/to/run/checkpoints/model_XXXXXXX.pt
run_dir="$(dirname "$(dirname "$ckpt")")"
out=artifacts/g1-largebox-h-model_XXXXXXX.deploy
mxlab export-deploy run_dir="$run_dir" checkpoint="$ckpt" output="$out"
mxlab deploy task=g1-largebox/sim artifact="$out" runtime.backend=motrixsim loop=true
mxlab deploy task=g1-largebox/sim artifact="$out" runtime.backend=mujoco loop=true
```

The run metadata selects the environment/profile. The environment name is not inferred from the output filename. Other task families require their own registered deployment profiles and matching sim recipes.

Reference: https://mujoco.readthedocs.io/en/stable/computation/#collision-detection
