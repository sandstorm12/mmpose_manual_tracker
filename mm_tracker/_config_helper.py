import os

from shutil import copyfile


def copy_sample_config(destination_path):
    sample_config_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'template_config.yml'
    )
    destination_path = os.path.join(
        destination_path,
        'config.yml'
    )

    copyfile(sample_config_path, destination_path)

    print("Config file created at: {}".format(os.path.abspath(destination_path)))
