from pathlib import Path
import csv


data_dir = Path("data") / "NASA ADAPT Dataset"
file_path = next(data_dir.rglob("Exp_304_comp3_pb.txt"))

with file_path.open(encoding="utf-8-sig", newline="") as file:
    rows = list(csv.reader(file, delimiter="\t"))

header = next(
    row
    for row in rows
    if len(row) > 1 and row[0] == "SensorData" and row[1] == "Time"
)

channels = header[2:]
measurements = [
    row for row in rows
    if row and row[0] == "AntagonistData"
]
faults = [
    row for row in rows
    if row and row[0] == "FaultInject"
]

commands = [
    row for row in rows
    if row and row[0] == "UserCommand"
]


print(f"Файл: {file_path}")
print(f"Каналов: {len(channels)}")
print(f"Отсчётов: {len(measurements)}")
print(f"Начало: {measurements[0][1]}")
print(f"Конец: {measurements[-1][1]}")

print("\nСобытия внесения неисправностей:")

for fault in faults:
    print(f"Время: {fault[1]}")
    print(f"Компонент: {fault[2]}")
    print(f"Тип компонента: {fault[3]}")
    print(f"Неисправность: {fault[4]}")
    print(f"Характер: {fault[7]}")
    print(f"Параметры: {fault[8]}")


    from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


def parse_time(value):
    without_timezone = value.split(" GMT")[0]
    return datetime.strptime(
        without_timezone,
        "%Y-%m-%d %H:%M:%S.%f",
    )


measurement_times = [
    parse_time(row[1])
    for row in measurements
]

start_time = measurement_times[0]

relative_time = np.array([
    (timestamp - start_time).total_seconds()
    for timestamp in measurement_times
])

values = np.array([
    [float(value) for value in row[2:2 + len(channels)]]
    for row in measurements
])

fault_time = parse_time(faults[0][1])
fault_second = (fault_time - start_time).total_seconds()

command_seconds = []

print("\nКоманды управления:")

for command in commands:
    command_time = parse_time(command[1])
    command_second = (
        command_time - start_time
    ).total_seconds()

    command_seconds.append(command_second)

    print(
        f"{command_second:7.3f} с: "
        f"{command[2]} = {command[3]}"
    )

fault_index = int(np.argmin(
    np.abs(relative_time - fault_second)
))

print(f"\nОтказ внесён через: {fault_second:.3f} с")
print(f"Ближайший отсчёт: {fault_index}")
print(f"Время ближайшего отсчёта: {relative_time[fault_index]:.3f} с")

window_size = 10

before_fault = values[
    fault_index - window_size:fault_index
]

after_fault = values[
    fault_index:fault_index + window_size
]

mean_before = before_fault.mean(axis=0)
mean_after = after_fault.mean(axis=0)

signal_range = values.max(axis=0) - values.min(axis=0)

normalized_change = (
    np.abs(mean_after - mean_before)
    / (signal_range + 1e-8)
)

sorted_indices = np.argsort(normalized_change)[::-1]

print("\nКаналы с наибольшим изменением после отказа:")

for channel_index in sorted_indices[:15]:
    print(
        f"{channels[channel_index]:10s} "
        f"до={mean_before[channel_index]:9.3f} "
        f"после={mean_after[channel_index]:9.3f} "
        f"оценка={normalized_change[channel_index]:.3f}"
    )

selected_channels = [
    "E261",
    "E265",
    "E267",
    "ESH260A",
    "ISH262",
    "IT261",
]

figure, axes = plt.subplots(
    len(selected_channels),
    1,
    figsize=(12, 10),
    sharex=True,
)

for axis, channel_name in zip(axes, selected_channels):
    channel_index = channels.index(channel_name)

    axis.plot(
        relative_time,
        values[:, channel_index],
        linewidth=1.2,
    )

    for command_number, command_second in enumerate(command_seconds):
        axis.axvline(
            command_second,
            color="gray",
            linestyle=":",
            alpha=0.6,
            label=(
                "Команда управления"
                if axis is axes[0] and command_number == 0
                else None
            ),
    )



    axis.axvline(
        fault_second,
        color="red",
        linestyle="--",
        label="Внесение отказа",
    )

    axis.set_ylabel(channel_name)
    axis.grid(alpha=0.3)

axes[0].legend()
axes[-1].set_xlabel("Время от начала эксперимента, с")

figure.suptitle(
    "Эксперимент 304: отказ реле EY260",
    fontsize=14,
)

figure.tight_layout()

output_path = Path("images") / "adapt_exp304_fault.png"
output_path.parent.mkdir(exist_ok=True)

figure.savefig(output_path, dpi=150)
print(f"График сохранён: {output_path}")

plt.show()