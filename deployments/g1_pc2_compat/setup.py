from setuptools import setup, find_namespace_packages
setup(name="motrix-deploy-pc2-wbt", version="0.1.0", python_requires=">=3.10,<3.14",
      package_dir={"": "src"}, packages=find_namespace_packages("src"),
      package_data={"motrix_deploy_pc2.config": ["task/*/*.yaml"]},
      entry_points={"motrix_deploy.tasks": [
          "g1_wbt_box_pc2/v1 = motrix_deploy_pc2.wbt_box:G1WbtDeployTask",
          "g1_wbt_robot_only_pc2/v1 = motrix_deploy_pc2.wbt_robot_only:G1WbtDeployTask"]})
