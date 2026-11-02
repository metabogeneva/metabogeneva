from typing import Any

from numpy import array, mean, ndarray, ones_like

from metabogeneva.barca.plot import classification_plot

BuildData = dict[str, list[tuple[int, dict[str, float]]]]


class Classification:
    def __init__(self, build_data: BuildData) -> None:
        target_groups: set[str] = set([])
        for classifications in build_data.values():
            for _, classification in classifications:
                for target_group in classification:
                    target_groups.add(target_group)

        self.source_groups = sorted(list(build_data.keys()))
        self.target_groups = sorted(list(target_groups))

        probabilities: dict[str, dict[str, list[float]]] = {
            target_group: {source_group: [] for source_group in self.source_groups}
            for target_group in self.target_groups
        }
        observations: dict[str, list[int]] = {
            source_group: [] for source_group in self.source_groups
        }
        for source_group in self.source_groups:
            for observation_index, classification in build_data.get(source_group, []):
                observations[source_group].append(observation_index)
                for target_group in self.target_groups:
                    probabilities[target_group][source_group].append(
                        classification.get(target_group, 0)
                    )

        self.probabilities = {
            target_group: {
                source_group: array(probabilities[target_group][source_group])
                for source_group in self.source_groups
            }
            for target_group in self.target_groups
        }
        self.observations = {
            source_group: array(observations[source_group])
            for source_group in self.source_groups
        }

    def show(self, **args: Any) -> None:
        classification_plot(self, **args)

    def matrix(self, average_sources: bool = False) -> tuple[ndarray, ndarray, ndarray]:
        target_group_array = array(self.target_groups)
        if average_sources:
            averages = self.group_averages()
            source_group_array = array(self.source_groups)
            probability_array = array(
                [
                    [
                        averages[target_group][source_group]
                        for source_group in source_group_array
                    ]
                    for target_group in target_group_array
                ]
            )
        else:
            source_group_array = array(
                [
                    source_group
                    for source_group in self.source_groups
                    for _ in self.observations[source_group]
                ]
            )
            probability_array = array(
                [
                    [
                        probability
                        for source_group in self.source_groups
                        for probability in self.probabilities[target_group][
                            source_group
                        ]
                    ]
                    for target_group in self.target_groups
                ]
            )
        return probability_array.transpose(), target_group_array, source_group_array

    def group_averages(self) -> dict[str, dict[str, float]]:
        return {
            target_group: {
                source_group: mean(self.probabilities[target_group][source_group])
                for source_group in self.source_groups
            }
            for target_group in self.target_groups
        }

    def group_accuracies(self) -> dict[str, float]:
        averages = self.group_averages()
        return {
            target_group: averages[target_group].get(target_group, 0)
            for target_group in self.target_groups
        }

    def average_accuracy(self) -> float:
        accuracies = [
            probability
            for source_group in self.source_groups
            for probability in self.probabilities[source_group][source_group]
        ]
        if len(accuracies) > 0:
            return mean(accuracies)
        return 0

    def predict(self) -> dict[str, ndarray]:
        best_probabilities = {
            source_group: float("-inf") * ones_like(self.observations[source_group])
            for source_group in self.source_groups
        }
        best_predictions = {
            source_group: array(
                [source_group] * len(self.observations[source_group]), dtype=object
            )
            for source_group in self.source_groups
        }
        for source_group in self.source_groups:
            for target_group in self.target_groups:
                best_selection = (
                    best_probabilities[source_group]
                    < self.probabilities[target_group][source_group]
                )
                best_probabilities[source_group][best_selection] = self.probabilities[
                    target_group
                ][source_group][best_selection]
                best_predictions[source_group][best_selection] = target_group
        return best_predictions

    def prediction_accuracy(self, merge_groups: list[list[str]] | None = None) -> float:
        group_dict: dict[str, list[str]] = {}
        if merge_groups is not None:
            for groups in merge_groups:
                for group in groups:
                    group_dict[group] = groups
        predictions = self.predict()
        matches = array(
            [
                source_group in group_dict.get(predicted_group, [predicted_group])
                for source_group in self.source_groups
                for predicted_group in predictions[source_group]
            ],
            dtype=float,
        )
        return mean(matches)


class ClassificationBuilder:
    def __init__(self) -> None:
        self.data: BuildData = {}

    def add(
        self,
        source_group: str,
        observation_index: int,
        classification: dict[str, float],
    ) -> None:
        if source_group not in self.data:
            self.data[source_group] = []
        self.data[source_group].append((observation_index, {**classification}))

    def build(self) -> Classification:
        return Classification(self.data)
