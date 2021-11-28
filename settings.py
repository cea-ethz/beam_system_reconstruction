import configparser

_settings_dict = {}


def read(name):
    return _settings_dict[name]


def write(name, value):
    _settings_dict[name] = value


def load_user_settings():
    config = configparser.ConfigParser()
    config.read("user_settings.ini")

    def load_setting(section, item, function_name):
        write("{}.{}".format(section, item), getattr(config[section], function_name)(item))

    load_setting("verbosity", "floor_test", "getboolean")
