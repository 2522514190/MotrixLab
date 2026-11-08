import json,sys
import numpy as np
from motrix_deploy.artifact.io import read_artifact
from motrix_deploy.policy import OnnxPolicy
from motrix_deploy.task import create_task
from motrix_deploy.config import SimRuntimeCfg
from motrix_deploy.runtime.config import SensorBindings
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_deploy.runtime.control import ControlSession
from motrix_env_core.config.sim import SimCfg
art=read_artifact('artifacts/g1-29dof-wbt-largebox-omniretarget-fastsac-b-model_0028000.deploy');m=art.manifest
for variant in sys.argv[1:]:
 backend='motrixsim' if variant=='native' else 'mujoco'
 policy=OnnxPolicy(art.policy_path,m.policy.input,m.policy.output)
 task=create_task(m.task,m.robot,policy,artifact=art)
 cfg=SimRuntimeCfg(backend=backend,viewer=False,deploy_env_id='g1-largebox',robot_id='g1-29dof',physics=SimCfg(dt=.005,solver_iterations=3),sensor_bindings=SensorBindings(base_angular_velocity='base_gyro',base_linear_acceleration='base_accel'))
 runtime=create_simulation_runtime(backend,task.configure_simulation(cfg.build_runtime_config()))
 session=ControlSession(controller=task,robot=runtime.robot,period_s=.02,state_timeout_s=.1)
 runtime.bind_session(session)
 rows=[]
 with runtime:
  if backend=='mujoco':
   model=runtime.model; gid=model.geom('largebox_collision').id
   if variant=='no_box': model.geom_contype[gid]=0;model.geom_conaffinity[gid]=0
   if variant=='no_joint_friction': model.dof_frictionloss[:]=0
   if variant=='no_joint_friction_100': model.dof_frictionloss[:]=0;model.opt.iterations=100
   if variant=='no_self_collision':
    model.geom_contype[:]=2;model.geom_conaffinity[:]=1
    model.geom_contype[gid]=1;model.geom_conaffinity[gid]=3
    for i in range(model.ngeom):
     if model.geom_bodyid[i]==0:model.geom_contype[i]=1;model.geom_conaffinity[i]=3
   if variant=='elliptic':model.opt.cone=1;model.opt.iterations=100
   if variant=='stiff_contacts':model.geom_solref[:]=[.005,1];model.opt.iterations=100
   if variant=='multiccd':model.opt.enableflags |= 1<<4;model.opt.iterations=100
   if variant=='pgs3':model.opt.solver=0
   if variant=='pgs100':model.opt.solver=0;model.opt.iterations=100
   if variant=='pgs_elliptic':model.opt.solver=0;model.opt.iterations=100;model.opt.cone=1
   if variant=='implicitfast':model.opt.integrator=3;model.opt.iterations=100
   if variant=='implicitfast_elliptic':model.opt.integrator=3;model.opt.iterations=100;model.opt.cone=1
   if variant=='rk4':model.opt.integrator=1;model.opt.iterations=100
   if variant=='direct_contact':model.geom_solref[:]=[-2500,-100];model.opt.iterations=100
   if variant=='direct_elliptic':model.geom_solref[:]=[-2500,-100];model.opt.iterations=100;model.opt.cone=1
   if variant=='box_primitive':
    model.geom_type[gid]=6;model.geom_size[gid]=[.235577,.229365,.203948];model.geom_pos[gid]=[.001494,-.000715,.005756];model.geom_quat[gid]=[1,0,0,0]
   if variant=='fresh_state':
    old_read=runtime.robot.read_state
    def fresh_read(timeout):
     runtime.mj.mj_forward(runtime.model,runtime.data)
     return old_read(timeout)
    runtime.robot.read_state=fresh_read
  original=task.step
  def step(state,ctx):
   if backend=='mujoco':
    root=runtime.data.xpos[runtime.model.body('pelvis').id].copy();box=runtime.data.xpos[runtime.model.body('baseLink').id].copy()
    contacts=[(runtime.model.geom(int(c.geom1)).name,runtime.model.geom(int(c.geom2)).name,float(c.dist)) for c in runtime.data.contact]
   else:
    root=runtime.model.get_link('pelvis').get_position(runtime.data).reshape(-1).copy();box=runtime.model.get_link('baseLink').get_position(runtime.data).reshape(-1).copy();contacts=[]
   rows.append({'step':ctx.step,'root':root.tolist(),'box':box.tolist(),'q':state.joint_position.tolist(),'w':state.base_angular_velocity.tolist(),'contacts':contacts})
   return original(state,ctx)
  task.step=step
  result=runtime.run()
  print(variant,result.completed_steps,result.error, 'root',rows[-1]['root'],'box',rows[-1]['box'],flush=True)
  with open('/home/yudong-chen/.codex/worktrees/box-carry-training/MotrixLab/artifacts/g1-largebox-sim2sim-audit/'+variant+'.json','w') as f:json.dump({'result':result.to_dict(),'trace':rows},f)
