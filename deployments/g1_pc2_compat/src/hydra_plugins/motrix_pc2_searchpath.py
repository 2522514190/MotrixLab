from hydra.plugins.search_path_plugin import SearchPathPlugin

class MotrixPc2SearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path):
        if any(e.provider == "main" and e.path == "pkg://motrix_deploy.config" for e in search_path.get_path()):
            search_path.append(provider="motrix-pc2", path="pkg://motrix_deploy_pc2.config")
