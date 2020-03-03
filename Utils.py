import re


def locate_string_in_arr(arr: [], string: str):
    index = 0
    while arr[index] != string:
        index+=1
    return index


def extract_parameter_value_as_int(json_string: str, parameter: str):
    extracted_value = re.search(parameter + r': (.*|\d*)(\s|,|})', json_string).group(1)
    try:
        return int(extracted_value)
    except ValueError:
        return extracted_value
