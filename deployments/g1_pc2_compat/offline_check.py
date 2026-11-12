"""Offline acceptance for PC2: no SDK, runtime construction or network."""
import argparse
import hashlib
import importlib.abc
import inspect
import json
from pathlib import Path
import socket
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
args = parser.parse_args()
class NoHardware(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('motrix_deploy_unitree', 'unitree_sdk2py')):
            raise RuntimeError('Offline check forbids hardware imports: ' + fullname)
sys.meta_path.insert(0, NoHardware())
class NoSocket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise RuntimeError('Offline check forbids network sockets')
socket.socket = NoSocket

import numpy as np
from motrix_deploy.artifact.io import read_artifact
from motrix_deploy.contracts import RobotState
from motrix_deploy.policy import NoOpPolicy, OnnxPolicy
from motrix_deploy.runtime.context import ControlContext
from motrix_deploy.task import create_task, load_task_type

results=[]
for mode, name in [('box','g1-box-78000-pc2.deploy'),('robot_only','g1-robot-only-pc2.deploy')]:
    artifact=read_artifact(args.bundle/'artifacts'/name)
    m=artifact.manifest
    cls=load_task_type(m.task.name)
    module=sys.modules[cls.__module__]
    assert cls.__module__ == 'motrix_deploy_pc2.wbt_'+mode
    actual=create_task(m.task,m.robot,OnnxPolicy(artifact.policy_path,m.policy.input,m.policy.output),
                       artifact=artifact,settings={'preparation':{'wait_for_remote_buttons':True}})
    state=RobotState(0,0,m.robot.default_joint_position.copy(),np.zeros(29,np.float32),
                     np.array([0,0,0,1],np.float32),np.zeros(3,np.float32),np.zeros(3,np.float32))
    actual.policy_io.align_reference_to(state)
    actual.policy_io.set_reference_frame(0)
    obs=actual.policy_io.build_observation(state,ControlContext(0,0,None,.02))
    action=actual.policy_controller.policy.infer(obs)
    assert obs.shape==(154,) and action.shape==(29,) and np.isfinite(action).all()
    cmd=actual.policy_io.process_action(np.full(29,100,np.float32))
    if mode=='box':
        midpoint=(m.robot.position_lower+m.robot.position_upper)*.5
        half=(m.robot.position_upper-m.robot.position_lower)*(.5*m.task.action_safety_clip_fraction)
        np.testing.assert_allclose(cmd.joint_position,np.clip(m.robot.default_joint_position+np.array(m.task.action_scale)*100,midpoint-half,midpoint+half),atol=1e-5)
    else:
        np.testing.assert_allclose(cmd.joint_position,m.robot.default_joint_position+np.array(m.task.action_scale)*100,atol=1e-5)
    # Exercise lifecycle and heading with synthetic stationary measurements, not a physics claim.
    for heading in [-1.7,0,1.2]:
        spec=m.task.model_copy(update={'motion_frames':2,'preparation_ramp_duration_s':0.,'preparation_settle_duration_s':0.})
        yaw=-1.6
        motion=module.G1WbtMotion(np.tile(m.robot.default_joint_position,(2,1)),np.zeros((2,29),np.float32),
                                 np.tile(np.array([0,0,np.sin(yaw/2),np.cos(yaw/2)],np.float32),(2,1)))
        t=cls(spec,m.robot,NoOpPolicy(29),motion=motion,settings={'preparation':{'default_transition_duration_s':.02},'final_frame_hold_s':.06})
        state.base_orientation_xyzw[:]=[0,0,np.sin(heading/2),np.cos(heading/2)]
        t.prepare(simulation=True);t.reset(state)
        phases=set()
        held=0
        for tick in range(150):
            decision=t.step(state,ControlContext(tick,tick*.02,None,.02));phases.add(t.phase)
            if decision.policy_tick:
                orientation=t.policy_io.build_observation(state,ControlContext(tick,tick*.02,None,.02))[58:64]
                np.testing.assert_allclose(orientation,[1,0,0,0,1,0],atol=1e-6)
            if t.phase=='final_hold':
                held+=1
                np.testing.assert_array_equal(t.policy_io._latest_frame,1)
                if mode=='robot_only' and held==20:t.request_stop()
            if decision.complete:break
        assert 'final_hold' in phases and t.phase=='complete',(mode,phases,t.phase)
        assert t.playback_steps==2
        if mode=='robot_only':assert held>=20
        else:assert held==3,held
    results.append({'mode':mode,'task_type':m.task.name,'frames':m.task.motion_frames,'onnx':'OK','action_transform':'OK','heading_and_lifecycle':'OK'})
print(json.dumps({'valid':True,'hardware_imports':False,'network':False,'checks':results},indent=2))
