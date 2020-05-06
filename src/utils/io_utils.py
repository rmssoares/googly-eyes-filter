import os
import yaml

GOOGLY_DIR = "googly_images"
CONFIG_DIR = "config"


def load_config(config_dir=CONFIG_DIR):
    config_filename = "config.yml"
    yaml_path = os.path.join(config_dir, config_filename)
    with open(yaml_path, 'r') as stream:
        return yaml.safe_load(stream)


def get_googly_filepath(filename, googly_dir=GOOGLY_DIR):
    if not os.path.exists(googly_dir):
        os.makedirs(GOOGLY_DIR)
    filename = filename.split('.')
    filename[-2] += "_googlified"
    googly_filename = '.'.join(filename)
    return os.path.join(googly_dir, googly_filename)


