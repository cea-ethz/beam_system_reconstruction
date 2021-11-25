_settings_dict = {}


def read(name):
    return _settings_dict[name]


def write(name, value):
    _settings_dict[name] = value
