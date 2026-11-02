from collections.abc import Callable
from typing import Any

from numpy import array, concatenate, copy, ndarray, ones, prod
from numpy.typing import ArrayLike
from pandas import DataFrame

from metabogeneva.io.filters import Filter


class DataTable:
    def __init__(
        self,
        observation_metadata: dict[str, "MetaData"],
        variable_metadata: dict[str, "MetaData"],
        data: ndarray,
    ) -> None:
        self.observation_metadata = observation_metadata
        self.variable_metadata = variable_metadata
        self.data = data
        self.validate()

    def flatten(self, list_dict: dict[str, "MetaData"]) -> list[dict[str, Any]]:
        keys = list_dict.keys()

        n = len(list_dict[max(keys)].values)
        result = []
        for i in range(n):
            new_dict = {}
            for key in keys:
                if list_dict[key].mask[i]:
                    new_dict[key] = list_dict[key].values[i]
            result.append(new_dict)
        return result

    def observation_metadata_dicts(self) -> list[dict[str, Any]]:
        return self.flatten(self.observation_metadata)

    def variable_metadata_dicts(self) -> list[dict[str, Any]]:
        return self.flatten(self.variable_metadata)

    def observation_mask(
        self, key: str, filter_function: Filter | None = None
    ) -> ndarray:
        return self.filter_mask(self.observation_metadata, key, filter_function)

    def variable_mask(self, key: str, filter_function: Filter | None = None) -> ndarray:
        return self.filter_mask(self.variable_metadata, key, filter_function)

    def filter_mask(
        self, metadata: dict[str, "MetaData"], key: str, filter_function: Filter | None
    ) -> ndarray:
        mask = metadata[key].mask
        values = metadata[key].values
        if filter_function is not None:
            mask = array(
                [
                    filter_function(i, value)[0] if value_mask else False
                    for i, (value_mask, value) in enumerate(zip(mask, values))
                ]
            )
        return mask

    def select(
        self,
        variable_mask: ndarray | None = None,
        observation_mask: ndarray | None = None,
    ) -> "DataTable":
        if variable_mask is None:
            variable_mask = ones(self.data.shape[1], dtype=bool)
        if observation_mask is None:
            observation_mask = ones(self.data.shape[0], dtype=bool)

        new_observation_metadata = {
            key: metadata.select(observation_mask)
            for key, metadata in self.observation_metadata.items()
        }
        new_variable_metadata = {
            key: metadata.select(variable_mask)
            for key, metadata in self.variable_metadata.items()
        }
        new_data = copy(self.data[observation_mask, :][:, variable_mask])

        return DataTable(new_observation_metadata, new_variable_metadata, new_data)

    def map_metadata(
        self,
        over_variables: bool,
        new_key: str,
        mapping_function: Callable[..., Any],
        *input_keys: str,
    ) -> None:
        metadata = (
            self.variable_metadata if over_variables else self.observation_metadata
        )
        n_metadata = self.data.shape[1] if over_variables else self.data.shape[0]
        result_values = []
        result_mask = []

        if len(input_keys) > 0:
            value_generator = zip(*[metadata[key].values for key in input_keys])
            mask_generator = zip(*[metadata[key].mask for key in input_keys])
            for i, input_masks, input_values in zip(
                range(n_metadata), mask_generator, value_generator
            ):
                result_values.append(mapping_function(i, *input_values))
                result_mask.append(prod(input_masks).astype(bool))
        else:
            for i in range(n_metadata):
                result_values.append(mapping_function(i))
                result_mask.append(True)

        metadata[new_key] = MetaData(array(result_mask), array(result_values))
        self.validate()

    def map_observation_metadata(self, *args: Any, **kwargs: Any) -> None:
        self.map_metadata(False, *args, **kwargs)

    def map_variable_metadata(self, *args: Any, **kwargs: Any) -> None:
        self.map_metadata(True, *args, **kwargs)

    def add_data(
        self,
        as_variables: bool,
        new_data: ndarray,
        new_metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        if as_variables:
            metadata = self.variable_metadata
        else:
            metadata = self.observation_metadata
        if new_metadata is None:
            new_metadata = [{}] * new_data.shape[1 if as_variables else 0]

        for key in metadata:
            new_values = []
            new_mask = []
            for new_metadata_entry in new_metadata:
                if key in new_metadata_entry:
                    new_values.append(new_metadata_entry[key])
                    new_mask.append(True)
                else:
                    new_values.append(None)
                    new_mask.append(False)
            value_type = (
                str
                if metadata[key].values.dtype.kind == "U"
                else metadata[key].values.dtype
            )
            metadata[key].values = concatenate(
                (metadata[key].values, array(new_values, dtype=value_type))
            )
            metadata[key].mask = concatenate((metadata[key].mask, array(new_mask)))

        if as_variables:
            self.data = concatenate((self.data, new_data), axis=1)
        else:
            self.data = concatenate((self.data, new_data), axis=0)
        self.validate()

    def add_observations(self, *args: Any, **kwargs: Any) -> None:
        self.add_data(False, *args, **kwargs)

    def add_variables(self, *args: Any, **kwargs: Any) -> None:
        self.add_data(True, *args, **kwargs)

    def to_dataframe(
        self,
        force_type: type | None = str,
        empty_value: str = "",
        missing_value: str = "",
        **pandas_args: Any,
    ) -> DataFrame:
        observation_keys = sorted(self.observation_metadata.keys())
        variable_keys = sorted(self.variable_metadata.keys())
        table: list[list[Any]] = []
        for key in observation_keys:
            table.append(
                [empty_value] * len(self.variable_metadata)
                + [key]
                + [
                    (value if mask else missing_value)
                    for mask, value in zip(
                        self.observation_metadata[key].mask,
                        self.observation_metadata[key].values,
                    )
                ]
            )
        table.append(variable_keys + [empty_value] * (self.data.shape[0] + 1))
        for j in range(self.data.shape[1]):
            table.append(
                [
                    self.variable_metadata[key].values[j]
                    if self.variable_metadata[key].mask[j]
                    else missing_value
                    for key in variable_keys
                ]
                + [empty_value]
                + [self.data[i, j] for i in range(self.data.shape[0])]
            )

        args: dict[str, Any] = {}
        if force_type is not None:
            args["dtype"] = force_type
        args.update(**pandas_args)
        return DataFrame(table, **args)

    def validate(self) -> None:
        for key, metadata in self.variable_metadata.items():
            if (
                len(metadata.values) != self.data.shape[1]
                or len(metadata.mask) != self.data.shape[1]
            ):
                raise MetadataConsistencyException(
                    f"Variable metadata for key {key} does not have the same size as the data"
                )
        for key, metadata in self.observation_metadata.items():
            if (
                len(metadata.values) != self.data.shape[0]
                or len(metadata.mask) != self.data.shape[0]
            ):
                raise MetadataConsistencyException(
                    f"Observation metadata for key {key} does not have the same size as the data"
                )


class MetaData:
    def __init__(self, mask: ArrayLike, values: ArrayLike) -> None:
        self.mask = array(mask)
        self.values = array(values)

    def select(self, submask: ndarray | None = None) -> "MetaData":
        if submask is None:
            submask = self.mask
        return MetaData(self.mask[submask], self.values[submask])


class MetadataConsistencyException(Exception):
    pass
