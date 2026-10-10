from collections import Counter
from datetime import datetime
from pathlib import Path
import csv


def parse_time(value):
    """Преобразует время ADAPT в datetime."""
    without_timezone = value.split(" GMT")[0]

    return datetime.strptime(
        without_timezone,
        "%Y-%m-%d %H:%M:%S.%f",
    )


def get_value(row, index):
    """Безопасно получает значение столбца."""
    if index < len(row):
        return row[index].strip()

    return ""


def extract_metadata_value(metadata, parameter_name):
    """Извлекает значение вида 'OperationCode = ...'."""
    prefix = f"{parameter_name} ="

    for value in metadata:
        value = value.strip()

        if value.startswith(prefix):
            return value.split("=", 1)[1].strip()

    return ""


project_dir = Path(__file__).resolve().parent

data_dir = (
    project_dir
    / "data"
    / "NASA ADAPT Dataset"
)

files = sorted(data_dir.rglob("Exp_*_comp3_pb.txt"))

if not files:
    raise FileNotFoundError(
        f"Файлы экспериментов не найдены в {data_dir}"
    )

output_dir = project_dir / "reports"
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "adapt_fault_events.csv"

event_records = []
component_counter = Counter()
mode_counter = Counter()

print(f"Найдено файлов: {len(files)}")
print("=" * 75)

for file_path in files:
    with file_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.reader(file, delimiter="\t"))

    experiment_id = file_path.stem.split("_")[1]

    metadata = rows[0]

    operation_code = extract_metadata_value(
        metadata,
        "OperationCode",
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

    if not measurements:
        print(
            f"Эксперимент {experiment_id}: "
            f"измерения отсутствуют"
        )
        continue

    start_time = parse_time(measurements[0][1])
    end_time = parse_time(measurements[-1][1])

    duration_seconds = (
        end_time - start_time
    ).total_seconds()

    print(
        f"Эксперимент {experiment_id}: "
        f"{len(measurements)} отсчётов, "
        f"{duration_seconds:.1f} с, "
        f"отказов: {len(faults)}, "
        f"режим: {operation_code}"
    )

    for fault_number, fault in enumerate(faults, start=1):
        fault_time = parse_time(get_value(fault, 1))

        fault_second = (
            fault_time - start_time
        ).total_seconds()

        location = get_value(fault, 2)
        component_type = get_value(fault, 3)
        fault_mode = get_value(fault, 4)
        development = get_value(fault, 7)
        parameters = get_value(fault, 8)

        component_counter[component_type] += 1
        mode_counter[fault_mode] += 1

        print(
            f"  {fault_number}. "
            f"{fault_second:7.3f} с | "
            f"{location:8s} | "
            f"{component_type:15s} | "
            f"{fault_mode:15s} | "
            f"{development:10s} | "
            f"{parameters}"
        )

        event_records.append({
            "experiment_id": experiment_id,
            "file_name": file_path.name,
            "operation_code": operation_code,
            "sample_count": len(measurements),
            "duration_seconds": round(
                duration_seconds,
                3,
            ),
            "fault_number": fault_number,
            "fault_time_seconds": round(
                fault_second,
                3,
            ),
            "fault_location": location,
            "component_type": component_type,
            "fault_mode": fault_mode,
            "fault_development": development,
            "fault_parameters": parameters,
        })

    print("-" * 75)


fieldnames = [
    "experiment_id",
    "file_name",
    "operation_code",
    "sample_count",
    "duration_seconds",
    "fault_number",
    "fault_time_seconds",
    "fault_location",
    "component_type",
    "fault_mode",
    "fault_development",
    "fault_parameters",
]

with output_path.open(
    "w",
    encoding="utf-8-sig",
    newline="",
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(event_records)


print("\nКоличество событий по типам компонентов:")

for component_type, count in component_counter.most_common():
    print(f"  {component_type:15s}: {count}")


print("\nКоличество событий по видам отказов:")

for fault_mode, count in mode_counter.most_common():
    print(f"  {fault_mode:15s}: {count}")


print(f"\nВсего событий отказа: {len(event_records)}")
print(f"Таблица сохранена: {output_path}")