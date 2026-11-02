from collections.abc import Callable
from re import compile as regex_compile
from typing import Any

from numpy import array, exp, mean, ndarray, sqrt

from metabogeneva.barca.classification import Classification, ClassificationBuilder
from metabogeneva.barca.data import GroupMetadata, LabeledDataset
from metabogeneva.barca.model import DecisionModelBase
from metabogeneva.barca.plot import coefficient_plot
from metabogeneva.io.display import NullMonitor

ForcedGroups = dict[bool, Any]


def make_tree(
    dataset: LabeledDataset,
    mirror_tree: "DecisionNode | None" = None,
    **param_kwargs: Any,
) -> "DecisionNode":
    node = DecisionNode(DecisionNodeParameters(**param_kwargs))
    if mirror_tree is None:
        node.build(dataset)
    else:
        node.mirror(mirror_tree)
        node.fit(dataset)
    return node


class DecisionNodeParameters:
    def __init__(
        self,
        model_maker: tuple[Callable[..., DecisionModelBase], dict[str, Any]],
        max_accept_groups: int,
        min_relative_spread: float = 0.01,
        force_single_prediction: bool = False,
        force_prediction_targets: bool = False,
        forced_groups: ForcedGroups | None = None,
        monitor: NullMonitor | None = None,
    ) -> None:
        self.model_maker = model_maker
        self.max_accept_groups = max_accept_groups
        self.min_relative_spread = min_relative_spread
        self.monitor = monitor if monitor is not None else NullMonitor()
        self.force_single_prediction = force_single_prediction
        self.force_prediction_targets = force_prediction_targets
        self.forced_groups = forced_groups


class DecisionNode:
    def __init__(
        self, parameters: DecisionNodeParameters, parent: "DecisionNode | None" = None
    ) -> None:
        self.parameters = parameters
        self.parent = parent
        self.accept_child: DecisionNode | None = None
        self.reject_child: DecisionNode | None = None

        self.variables: list[str] = []
        self.group_metadata: dict[str, GroupMetadata] = {}
        self.all_groups: list[str] = []
        self.accept_groups: list[str] = []
        self.reject_groups: list[str] = []

        model_maker, model_maker_args = self.parameters.model_maker
        self.model = model_maker(**model_maker_args)
        self.accept_prediction_spread: float = 0
        self.reject_prediction_spread: float = 0
        self.accept_prediction_target: float = 0
        self.reject_prediction_target: float = 0

    # =============================================================================================
    # API methods

    def classify(
        self, dataset: LabeledDataset, monitor: NullMonitor | None = None
    ) -> Classification:
        if monitor is None:
            monitor = NullMonitor()

        builder = ClassificationBuilder()
        count = len(dataset.groups)
        for i, group in enumerate(dataset.groups):
            probabilities = self.classify_single(dataset.data[i])
            builder.add(group, i, probabilities)
            monitor.state(f"Classifying observation {i + 1} of {count} [{group}]")
        results = builder.build()
        monitor.log("Done")
        return results

    def validate(
        self,
        dataset: LabeledDataset,
        leave_one_out: bool = True,
        monitor: NullMonitor | None = None,
    ) -> Classification:
        if monitor is None:
            monitor = NullMonitor()

        validation_node = DecisionNode(self.parameters)
        validation_node.mirror(self)
        validation_node.fit(dataset)

        builder = ClassificationBuilder()
        count = len(dataset.groups)
        for i, group in enumerate(dataset.groups):
            if leave_one_out:
                _, fit_dataset = dataset.split_by_index(i)
                validation_node.fit(fit_dataset)
            probabilities = validation_node.classify_single(dataset.data[i])
            builder.add(group, i, probabilities)
            monitor.state(
                f"Validating observation {i + 1} of {count} [{group}] (leave-one-out: {leave_one_out})"
            )
        results = builder.build()
        monitor.log(f"Average accuracy {100 * results.average_accuracy():.2f}%")
        monitor.log(f"Prediction accuracy {100 * results.prediction_accuracy():.2f}%")
        return results

    def reduce_dataset(
        self,
        dataset: LabeledDataset,
        relevant_variables_per_node: int = 3,
        regex: str = r".*",
    ) -> LabeledDataset:
        index_list = list(self.relevant_variables(relevant_variables_per_node, regex))
        return dataset.select_variables_by_index(array(index_list, dtype=int))

    def show(self, **args: Any) -> None:
        coefficient_plot(self, **args)

    def flatten(
        self,
        map_function: Callable[[int, "DecisionNode"], Any] = lambda _, m: m,
        base_index: int = 0,
    ) -> list[Any]:
        next_index = base_index

        result = [map_function(next_index, self)]
        next_index += 1

        if self.accept_child is not None:
            child_result = self.accept_child.flatten(map_function, next_index)
            result.extend(child_result)
            next_index += len(child_result)

        if self.reject_child is not None:
            child_result = self.reject_child.flatten(map_function, next_index)
            result.extend(child_result)
            next_index += len(child_result)

        return result

    def name(self) -> str:
        return ", ".join(self.accept_groups) + " vs " + ", ".join(self.reject_groups)

    # =============================================================================================
    # External utility methods
    def path(self) -> list["DecisionNode"]:
        return [self] if self.parent is None else self.parent.path() + [self]

    def build(self, dataset: LabeledDataset) -> None:
        self.build_recursive(dataset, None, self.parameters.forced_groups, True, True)

    def mirror(self, node: "DecisionNode") -> None:
        self.build_recursive(None, node, None, True, False)

    def fit(self, dataset: LabeledDataset) -> None:
        self.build_recursive(dataset, None, None, False, True)

    def classify_single(
        self,
        variable_data: ndarray,
        priors: dict[str, float] | None = None,
        default_prior: float = 1.0,
    ) -> dict[str, float]:
        given_priors = self.prepare_priors(priors, default_prior)
        posteriors = self.compute_posteriors(variable_data, given_priors)
        if self.parameters.force_single_prediction:
            max_probability_group = max(
                posteriors.keys(), key=lambda group: posteriors[group]
            )
            return {max_probability_group: 1.0}
        return posteriors

    # Internal utility methods

    def choose_prediction_groups(self, dataset: LabeledDataset) -> None:
        self.variables = [*dataset.variables]
        self.group_metadata = {**dataset.group_metadata}
        self.all_groups = sorted(list(dataset.group_set()))

        best: dict[str, Any] = {"r2": -float("inf")}
        path_tabs = "  " * (len(self.path()) - 1)

        def test_groups(accept_groups: list[str]) -> None:
            self.model.fit(dataset.data, dataset.group_mask(accept_groups))
            model_r2 = self.model.r2
            self.parameters.monitor.state(
                f"For groups {accept_groups}: score {model_r2}"
            )
            if model_r2 > best["r2"]:
                best["r2"] = model_r2
                best["groups"] = accept_groups

        n_groups = len(self.all_groups)

        def test_recursive(base_indices: list[int]) -> None:
            max_base_index = max(base_indices, default=-1)
            for i in range(max_base_index + 1, n_groups):
                chosen_indices = base_indices + [i]
                test_groups([self.all_groups[j] for j in chosen_indices])
                if int(n_groups / 2) > len(
                    chosen_indices
                ) and self.parameters.max_accept_groups > len(chosen_indices):
                    test_recursive(chosen_indices)

        test_recursive([])

        self.accept_groups = best["groups"]
        self.reject_groups = sorted(
            list(set(self.all_groups) - set(self.accept_groups))
        )
        self.parameters.monitor.log(
            f"{path_tabs}Selected {self.accept_groups} vs {self.reject_groups}"
        )
        self.parameters.monitor.log(f"{path_tabs}R2={best['r2']:.3f}")

    def copy_prediction_groups(self, node: "DecisionNode") -> None:
        self.variables = [*node.variables]
        self.group_metadata = {**node.group_metadata}
        self.all_groups = [*node.all_groups]
        self.accept_groups = [*node.accept_groups]
        self.reject_groups = [*node.reject_groups]

    def force_prediction_groups(
        self, dataset: LabeledDataset, forced_groups: ForcedGroups
    ) -> tuple[ForcedGroups | None, ForcedGroups | None]:
        self.variables = [*dataset.variables]
        self.group_metadata = {**dataset.group_metadata}
        self.all_groups = sorted(list(dataset.group_set()))

        accept_groups, forced_groups_accept = forced_groups[True]
        forced_groups_reject = forced_groups[False]

        self.accept_groups = sorted(accept_groups)
        self.reject_groups = sorted(
            list(set(self.all_groups) - set(self.accept_groups))
        )

        path_tabs = "  " * (len(self.path()) - 1)
        self.parameters.monitor.log(
            f"{path_tabs}Selected (forced) {self.accept_groups} vs {self.reject_groups}"
        )
        return forced_groups_accept, forced_groups_reject

    def fit_dataset(self, dataset: LabeledDataset) -> None:
        self.variables = [*dataset.variables]
        self.group_metadata = {**dataset.group_metadata}

        accept_mask = dataset.group_mask(self.accept_groups)
        reject_mask = dataset.group_mask(self.reject_groups)
        self.model.fit(dataset.data, accept_mask)
        if self.parameters.force_prediction_targets:
            self.accept_prediction_target = self.model.accept_target
            self.reject_prediction_target = self.model.reject_target
        else:
            self.accept_prediction_target = self.prediction_targets(
                dataset.data, accept_mask
            )
            self.reject_prediction_target = self.prediction_targets(
                dataset.data, reject_mask
            )
        self.accept_prediction_spread = self.prediction_spreads(
            dataset.data, accept_mask, self.accept_prediction_target
        )
        self.reject_prediction_spread = self.prediction_spreads(
            dataset.data, reject_mask, self.reject_prediction_target
        )

    def prediction_targets(self, data: ndarray, group_mask: ndarray) -> float:
        predictions = self.model.predict(data[group_mask])
        return mean(predictions)

    def prediction_spreads(
        self, data: ndarray, group_mask: ndarray, target: float
    ) -> float:
        predictions = self.model.predict(data[group_mask])
        spread = max(
            sqrt(mean((predictions - target) ** 2)),
            abs(target * self.parameters.min_relative_spread),
        )
        return spread

    def build_recursive(
        self,
        dataset: LabeledDataset | None,
        mirror_node: "DecisionNode | None",
        forced_groups: ForcedGroups | None,
        do_build: bool,
        do_fit: bool,
    ) -> None:
        forced_groups_accept = None
        forced_groups_reject = None
        accept_dataset: LabeledDataset | None
        reject_dataset: LabeledDataset | None
        if do_build:
            if forced_groups is not None:
                assert dataset is not None
                forced_groups_accept, forced_groups_reject = (
                    self.force_prediction_groups(dataset, forced_groups)
                )
            elif dataset is not None:
                self.choose_prediction_groups(dataset)
            elif mirror_node is not None:
                self.copy_prediction_groups(mirror_node)

        if dataset is not None:
            if do_fit:
                self.fit_dataset(dataset)
            accept_dataset, reject_dataset = dataset.split_by_groups(self.accept_groups)
        else:
            accept_dataset, reject_dataset = None, None

        if len(self.accept_groups) > 1:
            child_mirror = None
            if mirror_node is not None:
                child_mirror = mirror_node.accept_child
            if do_build:
                self.accept_child = DecisionNode(self.parameters, self)
            assert self.accept_child is not None
            self.accept_child.build_recursive(
                accept_dataset, child_mirror, forced_groups_accept, do_build, do_fit
            )
        else:
            if do_build:
                self.accept_child = None
        if len(self.reject_groups) > 1:
            child_mirror = None
            if mirror_node is not None:
                child_mirror = mirror_node.reject_child
            if do_build:
                self.reject_child = DecisionNode(self.parameters, self)
            assert self.reject_child is not None
            self.reject_child.build_recursive(
                reject_dataset, child_mirror, forced_groups_reject, do_build, do_fit
            )
        else:
            if do_build:
                self.reject_child = None

    def prepare_priors(
        self, partial_priors: dict[str, float] | None, default_prior: float
    ) -> dict[str, float]:
        given_priors = partial_priors
        if given_priors is None:
            given_priors = {}
        for group in self.all_groups:
            if group not in given_priors:
                given_priors[group] = default_prior
        normalization = sum(given_priors.values())
        return {group: given_priors[group] / normalization for group in self.all_groups}

    def compute_posteriors(
        self, variable_data: ndarray, priors: dict[str, float]
    ) -> dict[str, float]:
        score = self.model.predict(variable_data).item()
        score_on_accept = (
            exp(
                -0.5
                * (
                    (score - self.accept_prediction_target)
                    / self.accept_prediction_spread
                )
                ** 2
            )
            / self.accept_prediction_spread
        )
        score_on_reject = (
            exp(
                -0.5
                * (
                    (score - self.reject_prediction_target)
                    / self.reject_prediction_spread
                )
                ** 2
            )
            / self.reject_prediction_spread
        )
        posteriors = {}

        if len(self.accept_groups) > 1:
            assert self.accept_child is not None
            child_posteriors = self.accept_child.compute_posteriors(
                variable_data, priors
            )
            for group, probability in child_posteriors.items():
                posteriors[group] = probability * score_on_accept
        else:
            (accept_group,) = self.accept_groups
            posteriors[accept_group] = priors[accept_group] * score_on_accept

        if len(self.reject_groups) > 1:
            assert self.reject_child is not None
            child_posteriors = self.reject_child.compute_posteriors(
                variable_data, priors
            )
            for group, probability in child_posteriors.items():
                posteriors[group] = probability * score_on_reject
        else:
            (reject_group,) = self.reject_groups
            posteriors[reject_group] = priors[reject_group] * score_on_reject

        normalization = sum(posteriors.values())
        if normalization == 0:
            posteriors = {group: 1.0 for group in posteriors}
            normalization = sum(posteriors.values())
        posteriors = {group: posteriors[group] / normalization for group in posteriors}
        return posteriors

    def relevant_variables(self, number_per_node: int, regex: str) -> set[int]:
        regex_compiled = regex_compile(regex)
        filtered_indices = [
            (index, value)
            for index, (value, name) in enumerate(
                zip(self.model.weights, self.variables)
            )
            if regex_compiled.match(name)
        ]
        sorted_indices = [
            index for index, value in sorted(filtered_indices, key=lambda p: -abs(p[1]))
        ]
        selected_indices = set(sorted_indices[:number_per_node])
        if self.accept_child is not None:
            selected_indices = selected_indices | self.accept_child.relevant_variables(
                number_per_node, regex
            )
        if self.reject_child is not None:
            selected_indices = selected_indices | self.reject_child.relevant_variables(
                number_per_node, regex
            )
        return selected_indices
