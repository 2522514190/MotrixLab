from hydra.plugins.search_path_plugin import SearchPathPlugin


class MotrixRobotOnlySearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path):
        if any(entry.provider == "main" and entry.path == "pkg://motrix_deploy.config" for entry in search_path.get_path()):
            search_path.append(provider="motrix-robot-only", path="pkg://motrix_deploy_robot_only.config")
