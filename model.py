import math
from typing import Callable, Optional

import numpy as np
from scipy import optimize


def logistic_function(x, a, k, C):
    return C / (1 + a * np.exp(-k * x))


def exponential_function(x, a, k, c):
    return a * np.exp(k * x) + c


def linear_function(x, a, b):
    return a * x + b


def polynomial_2_function(x, a, b, c):
    return a * np.power(x, 2) + b * x + c


class Model:
    def __init__(
        self,
        x_values: list[int] | list[float],
        y_values: list[int] | list[float],
        regression_target_function: Callable,
        initial_regression_parameters: Optional[list[float]] = None,
    ) -> None:
        self.x_values = x_values
        self.y_values = y_values
        self.regression_target_function = regression_target_function

        self.optimal_parameters = optimize.curve_fit(
            regression_target_function,
            x_values,
            y_values,
            p0=initial_regression_parameters,
            maxfev=999999,
        )[0]

    def get_y_values(self, x_values, max_value=math.inf, min_value=0):
        return [
            max(
                min(
                    self.regression_target_function(x, *self.optimal_parameters),
                    max_value,
                ),
                min_value,
            )
            for x in x_values
        ]


def main():
    x_values = list(range(10))
    y_values = list(map(lambda x: 100 / (1 + math.exp(-2 * x)), x_values))

    model = Model(
        x_values, y_values, logistic_function, initial_regression_parameters=[1, 1, 50]
    )
    print(model.optimal_parameters)
    print(model.get_y_values(range(-10, 10)))


if __name__ == "__main__":
    main()
