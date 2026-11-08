import numpy as np
from motrix_deploy.config import SimRuntimeCfg
from motrix_deploy.runtime.config import SensorBindings
from motrix_deploy.runtime.factory import create_simulation_runtime
from motrix_env_core.config.sim import SimCfg
cfg=SimRuntimeCfg(backend='mujoco',viewer=False,deploy_env_id='g1-largebox',robot_id='g1-29dof',physics=SimCfg(dt=.005,solver_iterations=3),sensor_bindings=SensorBindings(base_angular_velocity='base_gyro',base_linear_acceleration='base_accel')).build_runtime_config()
a=create_simulation_runtime('mujoco',cfg);b=create_simulation_runtime('motrixsim',cfg)
with a,b:
 m=a.model;n=b.model
 for name in ['baseLink','pelvis']:
  i=m.body(name).id;print('MU BODY',name,m.body_mass[i],m.body_inertia[i],m.body_ipos[i],m.body_iquat[i])
  x=n.get_link(name);print('MT LINK',name,[(k,getattr(x,k)) for k in ['mass','inertia','com'] if hasattr(x,k)])
 for i in range(m.ngeom):
  name=m.geom(i).name
  x=n.get_geom(name)
  if name=='largebox_collision' or 'ankle' in name or 'wrist' in name:
   print('MU GEOM',name,'type',m.geom_type[i],'size',m.geom_size[i],'friction',m.geom_friction[i],'condim',m.geom_condim[i],'mask',m.geom_contype[i],m.geom_conaffinity[i], 'solref',m.geom_solref[i], 'pos',m.geom_pos[i])
   if x is not None:print('MT GEOM',name,[(k,getattr(x,k)) for k in ['friction','condim','size','typ','position','rotation'] if hasattr(x,k)])
 print('native geom dir',dir(n.get_geom('largebox_collision')))
 print('native link dir',dir(n.get_link('baseLink')))
 for name in a.robot.spec.joint_names[:6]:
  j=m.joint(name).id;d=m.jnt_dofadr[j];x=n.get_joint(name)
  print('JOINT',name,'MU',m.dof_damping[d],m.dof_armature[d],m.dof_frictionloss[d], 'MT',[(k,getattr(x,k)) for k in ['damping','armature','friction_loss'] if hasattr(x,k)])
