from collections.abc import Callable, Hashable, Iterable
from pickle import dump
from pickle import load as pickle_load
from typing import Any

from numpy import argwhere, array, transpose
from pandas import DataFrame, Series, read_csv, read_excel

from metabogeneva.io.datatable import DataTable, MetaData
from metabogeneva.io.display import NullMonitor
from metabogeneva.io.filters import Filter, as_is


def load(file_name: str) -> Any:
    with open(file_name, "rb") as in_file:
        return pickle_load(in_file)


def save(obj: object, file_name: str) -> None:
    with open(file_name, "wb") as out_file:
        dump(obj, out_file)


def read(
    file_name: str, file_type: str | None = None, **pandas_kwargs: Any
) -> DataFrame:
    if file_type is None:
        file_type = file_name.split(".")[-1]
    file_type = file_type.lower()

    pandas_merged_kwargs: dict[str, Any] = dict(
        header=None, index_col=False, keep_default_na=False, dtype=str, encoding="utf-8"
    )
    pandas_merged_kwargs.update(**pandas_kwargs)
    if file_type in {"csv", "tsv"}:
        pandas_frame = read_csv(file_name, **pandas_merged_kwargs)
    elif file_type in {"xls", "xlsx", "xlsm", "xlsb", "odf"}:
        pandas_frame = read_excel(file_name, **pandas_merged_kwargs)
    else:
        raise Exception(f"File type {file_type} is not a CSV or Excel file type")
    return pandas_frame


def parse(
    dataframe: DataFrame,
    observation_data_filter: Filter,
    variable_data_filter: Filter,
    observation_metadata_filters: list[tuple[str, int, Filter | None]],
    variable_metadata_filters: list[tuple[str, int, Filter | None]],
    cell_data_filter: Filter | None = None,
    observations_in_rows: bool = False,
    monitor: NullMonitor | None = None,
) -> DataTable:
    if monitor is None:
        monitor = NullMonitor()

    frame_observations: Iterable[tuple[Hashable, Series]]
    frame_variables: Iterable[tuple[Hashable, Series]]
    if observations_in_rows:
        frame_observations = dataframe.iterrows()
        frame_variables = dataframe.items()
    else:
        frame_observations = dataframe.items()
        frame_variables = dataframe.iterrows()

    monitor.log("Filtering observations")
    selected_observations = argwhere(
        [observation_data_filter(i, obs)[0] for i, obs in frame_observations]
    )[:, 0]
    monitor.log("Filtering variables")
    selected_variables = argwhere(
        [variable_data_filter(j, var)[0] for j, var in frame_variables]
    )[:, 0]

    def extract_metadata(
        filter_list: list[tuple[str, int, Filter | None]],
        extractor: Callable[[int], Series],
    ) -> dict[str, MetaData]:
        result = {}
        for metadata_key, index, filter_function in filter_list:
            monitor.state(f"Extracting {metadata_key}")
            the_filter_function = filter_function
            if the_filter_function is None:
                the_filter_function = as_is
            data_slice = extractor(index)
            filtered = [
                the_filter_function(i, cell) for i, cell in enumerate(data_slice)
            ]
            valids = []
            values = []
            for is_valid, value in filtered:
                valids.append(is_valid)
                values.append(value)
            result[metadata_key] = MetaData(array(valids), array(values))
        return result

    if observations_in_rows:
        observation_extractor = lambda index: dataframe.iloc[
            selected_observations, index
        ]
        variable_extractor = lambda index: dataframe.iloc[index, selected_variables]
    else:
        observation_extractor = lambda index: dataframe.iloc[
            index, selected_observations
        ]
        variable_extractor = lambda index: dataframe.iloc[selected_variables, index]

    monitor.log("Extracting observation metadata")
    observation_metadata = extract_metadata(
        observation_metadata_filters, observation_extractor
    )
    monitor.log("Extracting variable metadata")
    variable_metadata = extract_metadata(variable_metadata_filters, variable_extractor)

    monitor.log("Extracting data")
    if cell_data_filter is None:
        cell_data_filter = as_is
    if observations_in_rows:
        data_selection = dataframe.iloc[selected_observations, selected_variables]
    else:
        data_selection = dataframe.iloc[selected_variables, selected_observations]

    rows = []
    row_count = data_selection.shape[0]
    for i, (_, row) in enumerate(data_selection.iterrows()):
        monitor.state(f"Row {i + 1}/{row_count}")
        rows.append([cell_data_filter(None, cell)[1] for cell in row])

    monitor.log("Building data array")
    data = array(rows)
    if not observations_in_rows:
        data = transpose(data)

    monitor.state("")
    monitor.log("Done")
    return DataTable(observation_metadata, variable_metadata, data)
