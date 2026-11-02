from re import compile as regex_compile
from typing import Any

from numpy import (
    argwhere,
    array,
    concatenate,
    isin,
    logical_not,
    mean,
    ndarray,
    sqrt,
    var,
    zeros_like,
)
from numpy.random import shuffle as random_shuffle
from numpy.typing import ArrayLike

from metabogeneva.io.datatable import DataTable


def label(
    data_table: DataTable, observation_group_property: str, variable_id_property: str
) -> "LabeledDataset":
    group_metadata = data_table.observation_metadata[observation_group_property]
    id_metadata = data_table.variable_metadata[variable_id_property]

    groups, observation_mask = group_metadata.values, group_metadata.mask
    variable_ids, variable_mask = id_metadata.values, id_metadata.mask

    data = data_table.data[observation_mask, :][:, variable_mask]
    groups = groups[observation_mask]
    variable_ids = variable_ids[variable_mask]

    return LabeledDataset(groups, variable_ids, data)


class LabeledDataset:
    """
    Represents a data matrix with group annotations
    """

    def __init__(
        self,
        groups: ArrayLike,
        variables: ArrayLike,
        data: ArrayLike,
        group_metadata: dict[str, "GroupMetadata"] | None = None,
    ) -> None:
        self.groups = array(groups)
        self.variables = array(variables)
        self.data = array(data)
        self.group_metadata: dict[str, GroupMetadata] = (
            {} if group_metadata is None else {**group_metadata}
        )

    def add_group_information(self, metadata: dict[str, dict[str, str]]) -> None:
        for group, md in metadata.items():
            self.group_metadata[group] = GroupMetadata(**md)

    def group_set(self) -> set[str]:
        return set(self.groups)

    def group_mask(self, selected_groups: list[str]) -> ndarray:
        return isin(self.groups, selected_groups)

    def split_by_index(
        self, indices: int | ndarray
    ) -> tuple["LabeledDataset", "LabeledDataset"]:
        selected_mask = zeros_like(self.groups, dtype=bool)
        selected_mask[indices] = True
        unselected_mask = logical_not(selected_mask)
        return (
            LabeledDataset(
                self.groups[selected_mask],
                self.variables,
                self.data[selected_mask],
                self.group_metadata,
            ),
            LabeledDataset(
                self.groups[unselected_mask],
                self.variables,
                self.data[unselected_mask],
                self.group_metadata,
            ),
        )

    def split_by_fraction(
        self, fraction: float = 0.75, shuffle: bool = False
    ) -> tuple["LabeledDataset", "LabeledDataset"]:
        train_indices = array([], dtype=int)
        for group in self.group_set():
            group_indices = argwhere(self.groups == group)[:, 0]
            if shuffle:
                random_shuffle(group_indices)
            to_train_len = int(fraction * len(group_indices))
            train_indices = concatenate((train_indices, group_indices[:to_train_len]))
        return self.split_by_index(train_indices)

    def split_by_groups(
        self, selected_groups: list[str]
    ) -> tuple["LabeledDataset", "LabeledDataset"]:
        return self.split_by_index(self.group_mask(selected_groups))

    def select_variables_by_index(self, index_array: ndarray) -> "LabeledDataset":
        return LabeledDataset(
            self.groups,
            self.variables[index_array],
            self.data[:, index_array],
            self.group_metadata,
        )

    def select_variables_by_name(self, regex_string: str) -> "LabeledDataset":
        regex = regex_compile(regex_string)
        indices = argwhere([bool(regex.match(name)) for name in self.variables])[:, 0]
        return self.select_variables_by_index(indices)

    def renormalize(
        self, other_dataset: "LabeledDataset", renormalization_groups: list[str]
    ) -> None:
        self_means = {}
        other_means = {}
        self_vars = {}
        other_vars = {}
        weights = {g: 1 for g in renormalization_groups}
        for group in renormalization_groups:
            self_data = self.data[self.groups == group]
            other_data = other_dataset.data[other_dataset.groups == group]
            self_means[group] = mean(self_data, axis=0, keepdims=True)
            other_means[group] = mean(other_data, axis=0, keepdims=True)
            self_vars[group] = var(self_data, axis=0, keepdims=True)
            other_vars[group] = var(other_data, axis=0, keepdims=True)
        scale = sqrt(
            sum(
                [
                    weights[g] * self_vars[g] * other_vars[g]
                    for g in renormalization_groups
                ]
            )
            / sum([weights[g] * (self_vars[g] ** 2) for g in renormalization_groups])
        )
        shift = sum(
            [
                weights[g] * (other_means[g] - scale * self_means[g])
                for g in renormalization_groups
            ]
        ) / sum([weights[g] for g in renormalization_groups])
        self.data = scale * self.data + shift


class GroupMetadata:
    def __init__(
        self, description: str | None = None, color: str | None = None, **kwargs: Any
    ) -> None:
        self.description = description
        self.color = color
        self.other = {**kwargs}
