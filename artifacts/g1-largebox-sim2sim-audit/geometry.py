exec(open('/home/yudong-chen/.codex/worktrees/box-carry-training/MotrixLab/artifacts/g1-largebox-sim2sim-audit/inspect_models.py').read().split('with a,b:')[0])
import mujoco
with a,b:
 m=a.model;n=b.model
 g=n.get_geom('largebox_collision')
 for key in ['shape','mesh_name','mesh_scale','mesh_transform','mesh_variant_set','local_pose','visual_only','margin','gap','collision_affinity','collision_group']:
  print('BOX',key,getattr(g,key))
 print('aabb doc',g.get_local_aabb.__doc__)
 print('native world geom')
 for body in b.world.hierarchy.bodies:
  if body.link.name=='baseLink':
   for x in body.link.geoms: print(x.name,x.shape,x.physics_material)
 print('MU JOINTS',m.nq,m.nv,'MT',n.num_dof_pos,n.num_dof_vel)
 for i in range(m.ngeom):
  name=m.geom(i).name;x=n.get_geom(name)
  print('GEOM',i,name,'MU',m.geom_type[i],m.geom_friction[i],m.geom_condim[i],m.geom_solref[i],m.geom_solimp[i],m.geom_contype[i],m.geom_conaffinity[i], 'MT',None if x is None else (x.shape,x.friction.tolist(),x.collision_group,x.collision_affinity))
 gid=m.geom('largebox_collision').id;mid=m.geom_dataid[gid];start=m.mesh_vertadr[mid];count=m.mesh_vertnum[mid];v=m.mesh_vert[start:start+count];R=np.empty(9);mujoco.mju_quat2Mat(R,m.geom_quat[gid]);v=v@R.reshape(3,3).T+m.geom_pos[gid];print('MU BOX BOUNDS',v.min(0),v.max(0))
 from pathlib import Path
 path=Path('motrix_envs/src/motrix_envs/locomotion/wbt/assets/objects/largebox/largebox.obj');v=np.array([[float(z) for z in l.split()[1:4]] for l in path.read_text().splitlines() if l.startswith('v ')]);print('OBJ BOUNDS',v.min(0),v.max(0), 'verts',len(v))
