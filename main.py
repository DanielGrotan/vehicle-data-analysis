import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas import DataFrame


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

        for column_name, column_data in fuel_type_rows.iteritems():
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

    fuel_type_axes = plt.subplot(2, 1, 1)
    fuel_type_axes.set_title("Kjøretøy etter drivstofftype")
    fuel_type_axes.set_xlabel("År")
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

    smoothened_co2_emission = smoothen_graph(
        co2_emission_data_frame["Utslipp til luft (1 000 tonn CO2-ekvivalenter)"], k
    )
    co2_emission_axes.plot(
        co2_emission_data_frame["År"][k : len(co2_emission_data_frame["År"]) - k],
        smoothened_co2_emission,
        label="Glidende gjennomsnitt",
    )

    co2_emission_axes.grid()
    co2_emission_axes.legend()

    plt.show()


if __name__ == "__main__":
    main()
