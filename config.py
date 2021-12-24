import configparser
import argparse


# convert string to the right type, nested lists are not allowed
def _value_convert(value: str):
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "None":
        return None

    if value.startswith("["):
        result = []
        values = value[1: -1].split(",")
        for v in values:
            v = _value_convert(v.strip())
            result.append(v)
        return result
    try:
        value = int(value)
        return value
    except ValueError:
        try:
            value = float(value)
            return value
        except ValueError:
            return value


def _read_and_parser():
    par = configparser.ConfigParser()
    arg_parser = argparse.ArgumentParser(description='PyTorch Weak-shot Learning')
    par.read("config.ini")
    for title in par.keys():
        if title == "DEFAULT":
            continue
        for key in par[title].keys():
            val = _value_convert(par[title][key])
            arg_parser.add_argument(f"--{key}", default=val, type=type(val))
    return arg_parser.parse_args()


args = _read_and_parser()
