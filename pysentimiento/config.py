import os
import configparser

config_path = os.path.join(os.path.dirname(__file__), "../config/config.ini")


config = configparser.ConfigParser()

# Read config file if it exists
if os.path.exists(config_path):
    config.read(config_path)
