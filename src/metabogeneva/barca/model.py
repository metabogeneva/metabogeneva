from typing import Any

from numpy import abs as np_abs
from numpy import (
    array,
    copy,
    einsum,
    empty,
    maximum,
    mean,
    ndarray,
    ones_like,
    power,
    sqrt,
    std,
    where,
    zeros_like,
)
from numpy import max as np_max
from numpy import sum as np_sum
from numpy.linalg import inv, norm


class DecisionModelBase:
    loadings: ndarray | dict[str, ndarray]

    def __init__(self) -> None:
        self.r2: float = 0.0
        self.weights: ndarray = empty(0)
        self.accept_target: float = +1
        self.reject_target: float = -1

    def fit(self, data: ndarray, accept_mask: ndarray) -> None:
        """
        Fits this model to predict the observations where <accept_mask> is True as score=+1, and the rest as score=-1
        """

    def predict(self, data: ndarray) -> ndarray:
        """
        Returns the predicted score for the given variables
        """
        return mean(data, axis=0)


def pls(
    n_components: int = 1, min_uncertainty: float = 0
) -> tuple[type["DecisionModelPLS"], dict[str, Any]]:
    return DecisionModelPLS, dict(
        n_components=n_components, min_uncertainty=min_uncertainty
    )


class DecisionModelPLS(DecisionModelBase):
    def __init__(self, n_components: int, min_uncertainty: float) -> None:
        super().__init__()
        self.n_components = n_components
        self.min_uncertainty = min_uncertainty
        self.accept_target = +1
        self.reject_target = -1

    def fit(self, data: ndarray, accept_mask: ndarray) -> None:
        from sklearn.cross_decomposition import PLSRegression

        self.center = mean(data, axis=0, keepdims=True)
        self.scales = sqrt(
            power(std(data, axis=0, keepdims=True), 2) + power(self.min_uncertainty, 2)
        )

        self.accept_target = 1 / sqrt(np_sum(accept_mask))
        self.reject_target = -1 / sqrt(np_sum(~accept_mask))

        uv = self.apply_transform(data)
        dummy = where(accept_mask, self.accept_target, self.reject_target)
        self.pls = PLSRegression(n_components=self.n_components, scale=False)
        self.pls.fit(uv, dummy)
        self.r2 = self.pls.score(uv, dummy)
        self.weights = sqrt(np_sum(power(self.pls.x_weights_, 2), axis=1))
        self.loadings = self.pls.x_weights_[:, 0]

    def predict(self, data: ndarray) -> ndarray:
        return self.pls.predict(self.apply_transform(data))

    def apply_transform(self, data: ndarray) -> ndarray:
        return (data - self.center) / self.scales


def mpls(
    block_masks: dict[str, ndarray], min_uncertainty: float = 0
) -> tuple[type["DecisionModelMPLS"], dict[str, Any]]:
    return DecisionModelMPLS, dict(
        block_masks=block_masks, min_uncertainty=min_uncertainty
    )


class DecisionModelMPLS(DecisionModelBase):
    def __init__(self, block_masks: dict[str, ndarray], min_uncertainty: float) -> None:
        super().__init__()
        self.block_masks = {block: mask for block, mask in block_masks.items()}
        self.sorted_blocks = sorted(self.block_masks.keys())
        self.min_uncertainty = min_uncertainty
        self.accept_target = +1
        self.reject_target = -1

    def fit(self, data: ndarray, accept_mask: ndarray) -> None:
        self.center = mean(data, axis=0, keepdims=True)
        self.scales = maximum(
            sqrt(power(std(data, axis=0, keepdims=True), 2)), self.min_uncertainty
        )

        self.accept_target = 1 / sqrt(np_sum(accept_mask))
        self.reject_target = -1 / sqrt(np_sum(~accept_mask))

        uv = self.apply_transform(data)
        dummy = where(accept_mask, self.accept_target, self.reject_target)

        projection = einsum("ij,i->j", uv, dummy)
        self.loadings: dict[str, ndarray] = {}
        for block, mask in self.block_masks.items():
            block_projection = copy(projection)
            block_projection[~mask] = 0
            block_projection /= norm(block_projection)
            self.loadings[block] = block_projection

        scores = self.scores(uv)
        fit_data = array(
            [ones_like(dummy)] + [scores[block] for block in self.sorted_blocks]
        )
        discriminant = inv(einsum("ai,bi->ab", fit_data, fit_data))
        ordered_coefficients = einsum("ab,bi,i->a", discriminant, fit_data, dummy)

        self.constant_coefficient = ordered_coefficients[0]
        self.block_coefficient: dict[str, float] = {}
        for block, coefficient in zip(self.sorted_blocks, ordered_coefficients[1:]):
            self.block_coefficient[block] = coefficient

        self.weights = zeros_like(projection)
        for _, loading in self.loadings.items():
            self.weights += loading / np_max(np_abs(loading))

        predictions = self.predict(data)
        self.r2 = 1.0 - np_sum(power(predictions - dummy, 2)) / np_sum(power(dummy, 2))

    def predict(self, data: ndarray) -> ndarray:
        scores = self.scores(self.apply_transform(data))
        return self.constant_coefficient + np_sum(
            [
                scores[block] * self.block_coefficient[block]
                for block in self.sorted_blocks
            ],
            axis=0,
        )

    def apply_transform(self, data: ndarray) -> ndarray:
        return (data - self.center) / self.scales

    def scores(self, uv_data: ndarray) -> dict[str, ndarray]:
        result = {}
        for block, loading in self.loadings.items():
            result[block] = einsum("ij,j->i", uv_data, loading)
        return result
