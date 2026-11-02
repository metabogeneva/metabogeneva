from collections.abc import Callable, Collection, Mapping
from re import compile as regex_compile
from typing import Any

from numpy import nan

Filter = Callable[[Any, Any], tuple[bool, Any]]


def never(_i: Any, _value: Any) -> tuple[bool, Any]:
    return False, None


def as_is(_i: Any, value: Any) -> tuple[bool, Any]:
    return True, value


def valid_float(_i: Any, value: Any) -> tuple[bool, float]:
    try:
        float_value = float(value)
        return True, float_value
    except ValueError:
        return False, nan


def start_at(index: int) -> Filter:
    def result(i: Any, value: Any) -> tuple[bool, Any]:
        return i >= index, value

    return result


def valid_id(
    valid_regex: str = ".*",
    exact: str | None = None,
    invalid_identifiers: Collection[str] | None = None,
) -> Filter:
    regex = regex_compile(valid_regex)

    def result(_i: Any, value: Any) -> tuple[bool, Any]:
        if len(value) == 0:
            return False, value
        if invalid_identifiers is not None and value in invalid_identifiers:
            return False, value
        if regex.fullmatch(value) is None:
            return False, value
        if exact is not None and value != exact:
            return False, value
        return True, value

    return result


def map_values(value_dict: Mapping[Any, Any], missing: Any = None) -> Filter:
    def result(_i: Any, o: Any) -> tuple[bool, Any]:
        if o not in value_dict:
            return False, missing
        else:
            return True, value_dict[o]

    return result
