"""Run under the archived PC2 runtime; never construct hardware or connect sockets."""

import argparse
import importlib.abc
import json
import socket
import sys


class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("motrix_deploy_unitree", "unitree_sdk2py")):
            raise RuntimeError("hardware imports forbidden during offline validation")


class NoSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise RuntimeError("network forbidden during offline validation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact")
    args = parser.parse_args()
    sys.meta_path.insert(0, NoHardware())
    socket.socket = NoSocket
    import numpy as np
    from motrix_deploy_pc2.wbt_box import G1WbtDeployTask

    import motrix_deploy.task as factory
    from motrix_deploy.artifact.io import read_artifact
    from motrix_deploy.policy import OnnxPolicy

    original = factory.load_task_type
    factory.load_task_type = lambda name: G1WbtDeployTask if name == "g1_wbt_box_pc2/v1" else original(name)
    artifact = read_artifact(args.artifact)
    spec = artifact.manifest
    policy = OnnxPolicy(artifact.policy_path, spec.policy.input, spec.policy.output)
    task = G1WbtDeployTask(spec.task, spec.robot, policy, steps=745, artifact=artifact)
    result = policy.infer(np.zeros(154, np.float32))
    assert result.shape == (29,) and np.isfinite(result).all()
    target = task.policy_io.process_action(np.full(29, 100, np.float32)).joint_position
    midpoint = (spec.robot.position_lower + spec.robot.position_upper) * 0.5
    radius = (spec.robot.position_upper - spec.robot.position_lower) * 0.475
    assert np.all(target <= midpoint + radius + 1e-6)
    assert np.all(target >= midpoint - radius - 1e-6)
    print(
        json.dumps(
            dict(
                valid=True,
                frames=spec.task.motion_frames,
                clip=0.95,
                task_type=spec.task.name,
                hardware_imports=False,
                network=False,
            )
        )
    )


if __name__ == "__main__":
    main()
