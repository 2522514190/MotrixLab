from setuptools import setup, find_namespace_packages

setup(
    name="motrix-deploy-robot-only", version="0.1.0",
    python_requires=">=3.10,<3.14",
    package_dir={"": "src"}, packages=find_namespace_packages("src"),
    package_data={"motrix_deploy_robot_only.config": ["task/g1-wbt-robot-only/*.yaml"]},
    install_requires=["motrix-deploy==0.4.0", "motrix-env-core==0.4.0", "motrix-deploy-tasks==0.4.0"],
    entry_points={"motrix_deploy.tasks": ["g1_wbt_robot_only/v1 = motrix_deploy_robot_only.g1_wbt:G1WbtDeployTask"]},
)
