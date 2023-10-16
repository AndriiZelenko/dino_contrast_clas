import yaml
import os




def parse_training_conf(path):
    config_file = open(path, 'r')
    config_data = yaml.safe_load(config_file)
    return config_data


if __name__ == '__main__':
    parse_training_conf(path = '/home/andrii/adient/configs/training_conf.yaml')


