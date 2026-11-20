"""Record Unitree G1 LowState without publishing robot commands."""

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import quoteattr

import numpy as np


def record(interface, duration_s, output):
    # Set the transport before importing the DDS runtime.
    os.environ["CYCLONEDDS_URI"] = (
        '<CycloneDDS><Domain Id="0"><General><Interfaces>'
        f"<NetworkInterface name={quoteattr(interface)}/>"
        "</Interfaces></General></Domain></CycloneDDS>"
    )
    from cyclonedds.core import Policy, Qos
    from cyclonedds.domain import DomainParticipant
    from cyclonedds.sub import DataReader
    from cyclonedds.topic import Topic
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

    participant = DomainParticipant(0)
    topic = Topic(participant, "rt/lowstate", LowState_)
    reader = DataReader(
        participant, topic, qos=Qos(Policy.History.KeepLast(128), Policy.Reliability.BestEffort)
    )
    timestamps = []
    publication_handles = []
    motor_feedback = []
    deadline = time.monotonic() + duration_s
    print(f"Listening to rt/lowstate on {interface}. Output: {output}", flush=True)
    try:
        while time.monotonic() < deadline:
            samples = reader.take(128)
            read_ns = time.monotonic_ns()
            for sample in samples:
                if not isinstance(sample, LowState_):
                    continue
                # Samples taken together share one PC2 read time.
                info = getattr(sample, "sample_info", None)
                source_ns = getattr(info, "source_timestamp", -1)
                feedback = [(motor.q, motor.dq, motor.tau_est) for motor in sample.motor_state[:29]]
                timestamps.append((read_ns, source_ns, sample.tick))
                publication_handles.append(getattr(info, "publication_handle", 0))
                motor_feedback.append(feedback)
            if len(samples) < 128:
                time.sleep(0.001)
    except KeyboardInterrupt:
        pass

    if not timestamps:
        raise SystemExit("No rt/lowstate samples received. No file written.")
    stamp = np.asarray(timestamps, dtype=np.int64)
    values = np.asarray(motor_feedback, dtype=np.float32)
    with output.open("xb") as file:
        np.savez(
            file,
            pc2_read_monotonic_ns=stamp[:, 0],
            dds_source_timestamp_ns=stamp[:, 1],
            dds_publication_handle=np.asarray(publication_handles, dtype=np.uint64),
            robot_tick=stamp[:, 2].astype(np.uint32),
            motor_index=np.arange(29, dtype=np.uint8),
            joint_position_rad=values[:, :, 0],
            joint_velocity_rad_s=values[:, :, 1],
            estimated_torque_nm=values[:, :, 2],
        )
    print(f"Saved {len(stamp)} LowState samples to {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", default="eth0")
    parser.add_argument("--duration", type=float, default=180, help="Recording duration in seconds")
    parser.add_argument("--output", type=Path, help="Output .npz path")
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be a finite positive number")
    output = args.output or (
        Path.home() / "MotrixLab" / "logs" / "wbt"
        / datetime.now().strftime("motor-feedback-%Y%m%d-%H%M%S.npz")
    )
    if output.suffix != ".npz" or output.exists():
        parser.error("--output must be a new .npz file")
    output.parent.mkdir(parents=True, exist_ok=True)
    record(args.interface, args.duration, output)


if __name__ == "__main__":
    main()
