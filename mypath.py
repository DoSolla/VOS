from util.path_abstract import PathAbstract


class Path(PathAbstract):
    @staticmethod
    def db_root_dir():
        return '/Users/Zzi/Dropbox/UMass/2019Fall/670/project/DAVIS'

    @staticmethod
    def save_root_dir():
        return '/Users/Zzi/Dropbox/UMass/2019Fall/670/project/Result'

    @staticmethod
    def models_dir():
        return "/Users/Zzi/Dropbox/UMass/2019Fall/670/project/models"