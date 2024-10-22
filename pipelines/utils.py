import os


def get_project_root() -> str:
    return os.path.abspath(os.getcwd())
