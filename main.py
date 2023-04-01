import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame

from model import (
    Model,
    exponential_function,
    linear_function,
    logistic_function,
    polynomial_2_function,
)


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


def derive_graph(x_values, y_values):
    derived_values = []

    for i in range(len(x_values) - 1):
        derived_values.append(
            (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
        )

    return derived_values


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

    co2_emission_axes.grid()
    co2_emission_axes.legend()

    plt.show()


def plot_fuel_types_derived(fuel_types: list[str], fuel_type_counts, fuel_type_years):
    for fuel_type in fuel_types:
        counts_derived = derive_graph(fuel_type_years, fuel_type_counts[fuel_type])
        plt.plot(fuel_type_years[:-1], counts_derived, label=fuel_type + " derivert")

        # counts_derived_smoothened = smoothen_graph(counts_derived, 2)
        # plt.plot(
        #     fuel_type_years[2:-3],
        #     counts_derived_smoothened,
        #     label=fuel_type + " derivert (glidende gjennomsnitt)",
        # )

    plt.title("Drivstofftyper derivert")
    plt.xlabel("År")
    plt.ylabel("Vekstfart")
    plt.grid()
    plt.legend()
    plt.axhline(y=0, color="black")
    plt.show()


def plot_co2_emission_smoothened(co2_emission_data_frame, k, plot_normal_graph=True):
    emission = co2_emission_data_frame["Utslipp til luft (1 000 tonn CO2-ekvivalenter)"]
    years = co2_emission_data_frame["År"]

    if plot_normal_graph:
        plt.plot(years, emission, label="CO2-utslipp")

    smoothened_emission = smoothen_graph(emission, k)
    plt.plot(
        years[k : len(years) - k], smoothened_emission, label="Glidende gjennomsnitt"
    )

    plt.legend()
    plt.grid()
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

    # plot_fuel_types_derived(
    #     ["Bensin", "Elektrisk", "Diesel"], fuel_type_counts, fuel_type_years
    # )

    # plot_co2_emission_smoothened(co2_emission_data_frame, 4)

    # with open("temp.csv", "w") as f:
    #     for year, count in zip(fuel_type_years, fuel_type_counts["Elektrisk"]):
    #         f.write(f"{year-2008},{count}\n")

    num_years = 52

    electric_model = Model(
        range(len(fuel_type_years)),
        fuel_type_counts["Elektrisk"],
        exponential_function,
    )
    plt.plot(
        np.linspace(2008, 2008 + num_years, 1000),
        electric_model.get_y_values(
            np.linspace(0, num_years, 1000), max_value=2_906_012
        ),
        label="Elektrisk modell",
    )

    petrol_model = Model(
        range(len(fuel_type_years)), fuel_type_counts["Bensin"], linear_function
    )
    plt.plot(
        np.linspace(2008, 2008 + num_years, 1000),
        petrol_model.get_y_values(np.linspace(0, num_years, 1000)),
        label="Bensin modell",
    )

    diesel_model = Model(
        range(len(fuel_type_years)), fuel_type_counts["Diesel"], polynomial_2_function
    )
    plt.plot(
        np.linspace(2008, 2008 + num_years, 1000),
        diesel_model.get_y_values(np.linspace(0, num_years, 1000)),
        label="Diesel modell",
    )

    plt.legend()
    plt.grid()
    plt.xlabel("År")
    plt.ylabel("Antall registrerte kjøretøy")
    plt.title("Utvikling av registrerte kjøretøy (modeller)")

    plt.show()


if __name__ == "__main__":
    main()
