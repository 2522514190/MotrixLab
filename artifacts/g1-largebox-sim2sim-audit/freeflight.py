import json
from dataclasses import replace
import numpy as np
exec(open('/home/yudong-chen/.codex/worktrees/box-carry-training/MotrixLab/artifacts/g1-largebox-sim2sim-audit/sweep.py').read().split('for variant in sys.argv[1:]:')[0])
traces={}
for backend in ('motrixsim','mujoco'):
 policy=OnnxPolicy(art.policy_path,m.policy.input,m.policy.output)
 task=create_task(m.task,m.robot,policy,artifact=art)
 cfg=SimRuntimeCfg(backend=backend,viewer=False,deploy_env_id='g1-largebox',robot_id='g1-29dof',physics=SimCfg(dt=.005,solver_iterations=3),sensor_bindings=SensorBindings(base_angular_velocity='base_gyro',base_linear_acceleration='base_accel'))
 config=task.configure_simulation(cfg.build_runtime_config())
 initial=config.initial_state
 config=replace(config, initial_state=replace(initial,bodies={name:replace(body,position=(body.position[0],body.position[1],body.position[2]+5)) for name,body in initial.bodies.items()}))
 runtime=create_simulation_runtime(backend,config)
 if backend=='motrixsim':
  runtime.world.simulate_option.disable_contacts=True
  runtime.model=runtime.world.build()
 runtime.bind_session(ControlSession(controller=task,robot=runtime.robot,period_s=.02,state_timeout_s=.1))
 states=[]
 with runtime:
  if backend=='mujoco':runtime.model.opt.disableflags |= int(runtime.mj.mjtDisableBit.mjDSBL_CONTACT)
  state=runtime.robot.read_state(.1)
  action=policy.infer(task.policy_io.actor_observation(0,task.policy_io.torso_orientation(state),state.base_angular_velocity,state.joint_position,state.joint_velocity))
  runtime.robot.write_command(task.policy_io.process_action(action))
  for i in range(11):
   st=runtime.robot.read_state(.1);states.append(np.r_[st.joint_position,st.joint_velocity,st.base_orientation_xyzw,st.base_angular_velocity].tolist())
   runtime.advance_control_period()
 traces[backend]=states
for i in (0,1,5,10):
 a=np.asarray(traces['motrixsim'][i]);b=np.asarray(traces['mujoco'][i]);print(i,'q',np.max(abs(a[:29]-b[:29])),'v',np.max(abs(a[29:58]-b[29:58])),'angular',np.max(abs(a[-3:]-b[-3:])))
json.dump(traces,open('/home/yudong-chen/.codex/worktrees/box-carry-training/MotrixLab/artifacts/g1-largebox-sim2sim-audit/freeflight.json','w'))
