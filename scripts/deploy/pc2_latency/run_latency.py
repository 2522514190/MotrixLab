"""PC2-specific timing entry point; uses the isolated, instrumented native build."""

import importlib.util
import json
import os
import runpy
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/home/unitree/g1-sbto005-latency")
sys.path.insert(0, str(ROOT / "native/python"))
# PC2's editable finder otherwise redirects this package to the old build.
spec = importlib.util.spec_from_file_location(
    "motrix_unitree_native",
    ROOT / "native/python/motrix_unitree_native/__init__.py",
    submodule_search_locations=[str(ROOT / "native/python/motrix_unitree_native")],
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
extension_spec = importlib.util.spec_from_file_location(
    "motrix_unitree_native._native", next((ROOT / "native/python/motrix_unitree_native").glob("_native*.so"))
)
extension = importlib.util.module_from_spec(extension_spec)
sys.modules[extension_spec.name] = extension
extension_spec.loader.exec_module(extension)
spec.loader.exec_module(module)

# Load framework adapters only after selecting the instrumented extension.
from motrix_deploy.policy import OnnxPolicy  # noqa: E402
from motrix_deploy_tasks.tasks.g1_wbt import G1WbtDeployTask  # noqa: E402
from motrix_deploy_unitree.g1.native_robot import UnitreeG1NativeRobot  # noqa: E402

backends = []
steps = []
inferences = []
capacity = 100000
dropped = {"steps": 0, "inferences": 0}
original_factory = UnitreeG1NativeRobot._make_native_backend
original_step = G1WbtDeployTask.step
original_infer = OnnxPolicy.infer
original_write = UnitreeG1NativeRobot.write_command
commands = []


def factory(self):
    backend = original_factory(self)
    backend.enable_timing()
    backends.append(backend)
    return backend


def step(self, state, context):
    start = time.monotonic_ns()
    for backend in backends:
        backend.timing_source(state.receive_time_ns, start)
    result = original_step(self, state, context)
    end = time.monotonic_ns()
    if len(steps) < capacity:
        phase = self._startup.phase if self._startup.phase != "complete" else self.phase
        steps.append((state.receive_time_ns, start, end, phase))
    else:
        dropped["steps"] += 1
    return result


def infer(self, observation):
    start = time.monotonic_ns()
    result = original_infer(self, observation)
    end = time.monotonic_ns()
    if len(inferences) < capacity:
        inferences.append((start, end))
    else:
        dropped["inferences"] += 1
    return result


def write_command(self, command):
    start = time.monotonic_ns()
    result = original_write(self, command)
    if len(commands) < capacity and hasattr(command, "joint_position"):
        commands.append((start, np.asarray(command.joint_position).copy()))
    return result


def stats(values):
    values = np.asarray(values, dtype=np.float64) / 1e6
    if not len(values):
        return {"count": 0}
    return dict(
        count=len(values),
        **dict(zip(("p50_ms", "p95_ms", "p99_ms", "max_ms"), np.percentile(values, [50, 95, 99, 100]).tolist())),
    )


def save():
    prefix = Path(os.environ["PC2_TIMING_PREFIX"])
    rows = [row for backend in backends for row in backend.timing_rows()]
    native = np.asarray(rows, dtype=np.int64).reshape(-1, 6)
    ticks = np.asarray([row[:3] for row in steps], dtype=np.int64).reshape(-1, 3)
    inference = np.asarray(inferences, dtype=np.int64).reshape(-1, 2)
    feedback = np.asarray([row for b in backends for row in b.feedback_rows()], dtype=np.float64).reshape(-1, 60)
    writer = np.asarray([row for b in backends for row in b.writer_rows()], dtype=np.int64).reshape(-1, 4)
    feedback = feedback[np.argsort(feedback[:, 0])]
    writer = writer[np.argsort(writer[:, 0])]
    np.savez(
        prefix.with_suffix(".timing.npz"),
        native=native,
        steps=ticks,
        phases=np.asarray([row[3] for row in steps]),
        inference=inference,
        feedback=feedback,
        writer=writer,
        command_time_ns=np.asarray([c[0] for c in commands], dtype=np.int64),
        command_position=np.asarray([c[1] for c in commands], dtype=np.float32).reshape(-1, 29),
    )
    metrics = {
        "receive_to_publish_return": stats(native[:, 5] - native[:, 1]),
        "receive_to_step": stats(ticks[:, 1] - ticks[:, 0]),
        "task_step": stats(ticks[:, 2] - ticks[:, 1]),
        "command_update_to_publish_start": stats(native[:, 4] - native[:, 3]),
        "publish_call": stats(native[:, 5] - native[:, 4]),
        "inference": stats(inference[:, 1] - inference[:, 0]),
        "step_interval": stats(np.diff(ticks[:, 1])),
        "lowlevel_writer_interval": stats(np.diff(writer[:, 0])),
        "lowstate_arrival_interval": stats(np.diff(feedback[:, 0])),
    }
    policy_starts = np.array([row[1] for row in steps if row[3] in ("policy_hold", "playback")])
    ids = set(policy_starts.tolist())
    policy = native[np.array([row[2] in ids for row in native], dtype=bool)]
    metrics["policy_receive_to_publish_return"] = stats(policy[:, 5] - policy[:, 1])
    report = {
        "scope": (
            "PC2 native LowState callback arrival to first successful DDS publish return per command update; "
            "not motor execution latency"
        ),
        "clock": "Linux CLOCK_MONOTONIC / C++ steady_clock, nanoseconds",
        "native_columns": [
            "command_id",
            "receive_ns",
            "step_start_ns",
            "command_update_ns",
            "publish_start_ns",
            "publish_return_ns",
        ],
        "steps_columns": ["receive_ns", "step_start_ns", "step_end_ns"],
        "inference_columns": ["start_ns", "end_ns"],
        "feedback_columns": ["receive_ns", "robot_tick", "joint_position[29]", "joint_velocity[29]"],
        "writer_columns": [
            "publish_start_ns",
            "publish_return_ns",
            "command_id",
            "status: 0=failed,1=active,2=damping",
        ],
        "writer_intervals_over_2ms": int(np.count_nonzero(np.diff(writer[:, 0]) > 2000000)),
        "policy_intervals_over_22ms": int(np.count_nonzero(np.diff(policy_starts) > 22000000)),
        "execution_delay": (
            "Command and feedback trajectories recorded for offline response-lag estimation; "
            "no motor execution timestamp available"
        ),
        "metrics": metrics,
        "dropped": {**dropped, "native": sum(b.timing_dropped() for b in backends)},
        "recording": (
            "bounded in-memory buffers; feedback/writer retain last 100000 rows, other streams first 100000; "
            "native dropped includes overwritten rows; files written after CLI cleanup; "
            "abrupt power loss/SIGKILL loses buffered data"
        ),
    }
    prefix.with_suffix(".timing.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    UnitreeG1NativeRobot._make_native_backend = factory
    G1WbtDeployTask.step = step
    OnnxPolicy.infer = infer
    UnitreeG1NativeRobot.write_command = write_command
    try:
        runpy.run_path("/home/unitree/g1-pc2-wbt-v2/trace_pc2.py", run_name="__main__")
    finally:
        save()
