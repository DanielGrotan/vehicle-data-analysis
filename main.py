import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from model import Model, exponential_function, linear_function, polynomial_2_function


def load_data_from_filename(base_path: str, filename: str) -> DataFrame:
    return pd.read_csv(
        os.path.join(base_path, filename),
        sep=";",
        encoding="utf-8",
    )


def get_fuel_type_counts(
    fuel_type_counts_data_frame: DataFrame,
) -> dict[str, list[int]]:
    fuel_types = set(fuel_type_counts_data_frame["drivstofftype"])
    indexed_data_frame = fuel_type_counts_data_frame.set_index("drivstofftype")

    fuel_type_counts = {}

    for fuel_type in fuel_types:
        fuel_type_counts[fuel_type] = []

        fuel_type_rows = indexed_data_frame.loc[fuel_type]

        for column_name, column_data in fuel_type_rows.items():
            try:
                int(column_name)
            except ValueError:
                continue

            fuel_type_counts[fuel_type].append(column_data.sum())

    return fuel_type_counts


def smoothen_graph(y_values, k):
    smoothened_values = []

    for i in range(k, len(y_values) - k):
        smoothened_values.append(np.mean(y_values[(i - k) : (i + k)]))

    return smoothened_values


def plot_all_data(fuel_type_counts, fuel_type_years, co2_emission_data_frame):
    fuel_type_axes = plt.subplot(2, 1, 1)
    fuel_type_axes.set_title("Kjøretøy etter drivstofftype")
    fuel_type_axes.set_ylabel("Antall kjøretøy")

    for fuel_type, counts in fuel_type_counts.items():
        fuel_type_axes.plot(fuel_type_years, counts, label=fuel_type)

    fuel_type_axes.legend()
    fuel_type_axes.grid()

    co2_emission_axes = plt.subplot(2, 1, 2, sharex=fuel_type_axes)
    co2_emission_axes.set_title("CO2-utslipp")
    co2_emission_axes.set_xlabel("År")
    co2_emission_axes.set_ylabel("Utslipp til luft (1 000 tonn CO2-ekvivalenter)")

    co2_emission_axes.plot(
        co2_emission_data_frame["År"],
        co2_emission_data_frame["Utslipp til luft (1 000 tonn CO2-ekvivalenter)"],
        label="CO2-utslipp",
    )

    k = 3
    co2_emission_smoothened = smoothen_graph(
        co2_emission_data_frame["Utslipp til luft (1 000 tonn CO2-ekvivalenter)"], k
    )
    co2_emission_axes.plot(
        co2_emission_data_frame["År"][k : len(co2_emission_data_frame["År"]) - k],
        co2_emission_smoothened,
        label="Glidende gjennomsnitt",
    )

    co2_emission_axes.grid()
    co2_emission_axes.legend()

    plt.show()


def plot_fuel_type_model(
    fuel_type,
    fuel_type_years,
    fuel_type_counts,
    regression_target,
    num_years,
    model_logger_callback,
):
    model = Model(
        list(range(len(fuel_type_years))),
        fuel_type_counts[fuel_type],
        regression_target,
    )

    model_logger_callback(model.optimal_parameters)

    plt.plot(
        np.linspace(2008, 2008 + num_years, 1000),
        model.get_y_values(np.linspace(0, num_years, 1000), max_value=2_906_012),
        label=f"{fuel_type} modell",
    )

    plt.legend()
    plt.grid()
    plt.xlabel("År")
    plt.ylabel("Antall registrerte kjøretøy")
    plt.show()


def main() -> None:
    data_folder_path = os.path.join(os.path.dirname(__file__), "data")

    co2_emission_data_frame = load_data_from_filename(
        data_folder_path, "co2-emission.csv"
    )

    fuel_type_counts_data_frame = load_data_from_filename(
        data_folder_path, "fuel-type-counts.csv"
    )

    fuel_type_counts = get_fuel_type_counts(fuel_type_counts_data_frame)
    fuel_type_years = list(range(2008, 2022 + 1))

    plot_all_data(fuel_type_counts, fuel_type_years, co2_emission_data_frame)

    # Plot regresjonsmodeller

    num_years = 52

    plot_fuel_type_model(
        "Elektrisk",
        fuel_type_years,
        fuel_type_counts,
        exponential_function,
        num_years,
        lambda optimal_parameters: print(
            f"\nElektrisk modell\nf(x) = {optimal_parameters[0]} * e^({optimal_parameters[1]} * x)"
        ),
    )

    plot_fuel_type_model(
        "Bensin",
        fuel_type_years,
        fuel_type_counts,
        linear_function,
        num_years,
        lambda optimal_parameters: print(
            f"\nBensin modell\nf(x) = {optimal_parameters[0]}x + {optimal_parameters[1]}"
        ),
    )

    plot_fuel_type_model(
        "Diesel",
        fuel_type_years,
        fuel_type_counts,
        polynomial_2_function,
        num_years,
        lambda optimal_parameters: print(
            f"\nDiesel modell\nf(x) = {optimal_parameters[0]}x^2 + {optimal_parameters[1]}x + {optimal_parameters[2]}"
        ),
    )


if __name__ == "__main__":
    main()
