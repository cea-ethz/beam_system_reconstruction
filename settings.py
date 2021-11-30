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

    load_setting("display", "dag", "getboolean")
    load_setting("display", "histogram", "getboolean")
    load_setting("display", "hough", "getboolean")

    load_setting("export", "alpha_shape", "getboolean")

    load_setting("verbosity", "floor_test", "getboolean")

    load_setting("visibility", "beams_final", "getboolean")
    load_setting("visibility", "beams_rejected", "getboolean")
    load_setting("visibility", "column_clusters", "getboolean")
    load_setting("visibility", "columns_final", "getboolean")
    load_setting("visibility", "split_points", "getboolean")
    load_setting("visibility", "walls_extracted", "getboolean")
    load_setting("visibility", "world_axis", "getboolean")
    load_setting("visibility", "world_aabb", "getboolean")

