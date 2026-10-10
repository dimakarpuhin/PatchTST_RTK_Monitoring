from collections import Counter
from datetime import datetime
from pathlib import Path
import csv

import numpy as np


EXPERIMENT_IDS = {
    "304",
    "305",
    "306",
    "308",
    "309",
    "310",
    "311",
}

LABEL_NAMES = {
    0: "Норма",
    1: "Отказ датчика",
    2: "Отказ компонента",
}

WINDOW_LENGTH = 16
WINDOW_STRIDE = 4


def parse_time(value):
    without_timezone = value.split(" GMT")[0]

    return datetime.strptime(
        without_timezone,
        "%Y-%m-%d %H:%M:%S.%f",
    )


def get_experiment_id(file_path):
    return file_path.stem.split("_")[1]


def create_windows(
    values,
    label,
    experiment_id,
    segment_start,
):
    windows = []
    labels = []
    groups = []
    metadata = []

    for local_start in range(
        0,
        len(values) - WINDOW_LENGTH + 1,
        WINDOW_STRIDE,
    ):
        local_end = local_start + WINDOW_LENGTH

        global_start = segment_start + local_start
        global_end = segment_start + local_end

        window = values[local_start:local_end]

        windows.append(window)
        labels.append(label)
        groups.append(experiment_id)

        metadata.append({
            "experiment_id": experiment_id,
            "label": label,
            "start_index": global_start,
            "end_index": global_end,
        })

    return windows, labels, groups, metadata


def load_adapt_data():
    project_dir = Path(__file__).resolve().parent

    data_dir = (
        project_dir
        / "data"
        / "NASA ADAPT Dataset"
    )

    files = sorted(
        data_dir.rglob("Exp_*_comp3_pb.txt")
    )

    all_windows = []
    all_labels = []
    all_groups = []
    all_metadata = []

    reference_channels = None

    for file_path in files:
        experiment_id = get_experiment_id(file_path)

        if experiment_id not in EXPERIMENT_IDS:
            continue

        with file_path.open(
            encoding="utf-8-sig",
            newline="",
        ) as file:
            rows = list(
                csv.reader(file, delimiter="\t")
            )

        header = next(
            row
            for row in rows
            if (
                len(row) > 1
                and row[0] == "SensorData"
                and row[1] == "Time"
            )
        )

        channels = header[2:]

        if reference_channels is None:
            reference_channels = channels
        elif channels != reference_channels:
            raise ValueError(
                f"Каналы эксперимента "
                f"{experiment_id} отличаются"
            )

        measurements = [
            row
            for row in rows
            if row and row[0] == "AntagonistData"
        ]

        faults = [
            row
            for row in rows
            if row and row[0] == "FaultInject"
        ]

        if len(faults) != 1:
            raise ValueError(
                f"В эксперименте {experiment_id} "
                f"ожидался один отказ, найдено: "
                f"{len(faults)}"
            )

        timestamps = [
            parse_time(row[1])
            for row in measurements
        ]

        values = np.array(
            [
                [
                    float(value)
                    for value in row[
                        2:2 + len(channels)
                    ]
                ]
                for row in measurements
            ],
            dtype=np.float32,
        )

        fault_time = parse_time(faults[0][1])

        time_difference = np.array([
            abs(
                (timestamp - fault_time)
                .total_seconds()
            )
            for timestamp in timestamps
        ])

        fault_index = int(
            np.argmin(time_difference)
        )

        component_type = faults[0][3]

        if component_type == "Sensor":
            fault_label = 1
        else:
            fault_label = 2

        # Нормальный участок:
        # [начало эксперимента, момент отказа)
        normal_values = values[:fault_index]

        # Аварийный участок:
        # [момент отказа, конец эксперимента]
        fault_values = values[fault_index:]

        normal_result = create_windows(
            values=normal_values,
            label=0,
            experiment_id=experiment_id,
            segment_start=0,
        )

        fault_result = create_windows(
            values=fault_values,
            label=fault_label,
            experiment_id=experiment_id,
            segment_start=fault_index,
        )

        for result in (normal_result, fault_result):
            windows, labels, groups, metadata = result

            all_windows.extend(windows)
            all_labels.extend(labels)
            all_groups.extend(groups)
            all_metadata.extend(metadata)

        print(
            f"Эксперимент {experiment_id}: "
            f"отсчётов={len(values)}, "
            f"индекс отказа={fault_index}, "
            f"класс={LABEL_NAMES[fault_label]}"
        )

    X = np.asarray(
        all_windows,
        dtype=np.float32,
    )

    y = np.asarray(
        all_labels,
        dtype=np.int64,
    )

    groups = np.asarray(all_groups)

    if not np.isfinite(X).all():
        raise ValueError(
            "В подготовленных данных есть NaN или Inf"
        )

    return (
        X,
        y,
        groups,
        reference_channels,
        all_metadata,
    )


if __name__ == "__main__":
    X, y, groups, channels, metadata = (
        load_adapt_data()
    )

    print("\nИтоговая форма данных:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"groups: {groups.shape}")
    print(f"каналов: {len(channels)}")

    print("\nРаспределение классов:")

    class_counts = Counter(y)

    for label, label_name in LABEL_NAMES.items():
        print(
            f"{label} — {label_name}: "
            f"{class_counts[label]}"
        )

    print("\nРаспределение по экспериментам:")

    for experiment_id in sorted(set(groups)):
        experiment_mask = groups == experiment_id

        experiment_counts = Counter(
            y[experiment_mask]
        )

        print(
            f"{experiment_id}: "
            f"норма={experiment_counts[0]}, "
            f"датчик={experiment_counts[1]}, "
            f"компонент={experiment_counts[2]}"
        )