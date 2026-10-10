from collections import Counter

import numpy as np

from load_adapt_data import (
    LABEL_NAMES,
    load_adapt_data,
)


# Это пока проверочное разбиение для настройки конвейера.
# В итоговом эксперименте перейдём к групповой
# перекрёстной проверке.
TRAIN_IDS = {"304", "305", "306"}
VAL_IDS = {"308", "310"}
TEST_IDS = {"309", "311"}
NORMALIZED_CLIP_VALUE = 10.0

def select_experiments(X, y, groups, experiment_ids):
    mask = np.isin(
        groups,
        list(experiment_ids),
    )

    return (
        X[mask].copy(),
        y[mask].copy(),
        groups[mask].copy(),
    )


def print_distribution(name, y, groups):
    counts = Counter(y)

    print(f"\n{name}:")
    print(f"  окон: {len(y)}")
    print(
        "  эксперименты: "
        + ", ".join(sorted(set(groups)))
    )

    for label, label_name in LABEL_NAMES.items():
        print(
            f"  {label} — {label_name}: "
            f"{counts[label]}"
        )


def prepare_adapt_data():
    X, y, groups, channels, metadata = (
        load_adapt_data()
    )

    all_split_ids = (
        TRAIN_IDS
        | VAL_IDS
        | TEST_IDS
    )

    if (
        TRAIN_IDS & VAL_IDS
        or TRAIN_IDS & TEST_IDS
        or VAL_IDS & TEST_IDS
    ):
        raise ValueError(
            "Эксперименты пересекаются между выборками"
        )

    if set(groups) != all_split_ids:
        raise ValueError(
            "Не все эксперименты распределены "
            "между train, val и test"
        )

    X_train, y_train, groups_train = (
        select_experiments(
            X,
            y,
            groups,
            TRAIN_IDS,
        )
    )

    X_val, y_val, groups_val = (
        select_experiments(
            X,
            y,
            groups,
            VAL_IDS,
        )
    )

    X_test, y_test, groups_test = (
        select_experiments(
            X,
            y,
            groups,
            TEST_IDS,
        )
    )

    # Статистики вычисляются только по train.
    raw_channel_mean = X_train.mean(
        axis=(0, 1),
        keepdims=True,
        dtype=np.float64,
    ).astype(np.float32)

    raw_channel_std = X_train.std(
        axis=(0, 1),
        keepdims=True,
        dtype=np.float64,
    ).astype(np.float32)

    # Каналы отбираются исключительно по train,
    # без использования val и test.
    active_channel_mask = (
        raw_channel_std.reshape(-1) >= 1e-6
    )

    selected_channels = [
        channel
        for channel, is_active in zip(
            channels,
            active_channel_mask,
        )
        if is_active
    ]

    removed_channels = [
        channel
        for channel, is_active in zip(
            channels,
            active_channel_mask,
        )
        if not is_active
    ]

    # Удаляем постоянные обучающие признаки
    # из всех трёх выборок.
    X_train = X_train[
        :, :, active_channel_mask
    ]

    X_val = X_val[
        :, :, active_channel_mask
    ]

    X_test = X_test[
        :, :, active_channel_mask
    ]

    channel_mean = raw_channel_mean[
        :, :, active_channel_mask
    ]

    channel_std = raw_channel_std[
        :, :, active_channel_mask
    ]

    X_train = (
        (X_train - channel_mean) / channel_std
    ).astype(np.float32)

    X_val = (
        (X_val - channel_mean) / channel_std
    ).astype(np.float32)

    X_test = (
        (X_test - channel_mean) / channel_std
    ).astype(np.float32)

    print("\nДоля значений за пределами нормализации:")

    for name, values in (
        ("Train", X_train),
        ("Val", X_val),
        ("Test", X_test),
    ):
        clipped_fraction = np.mean(
            np.abs(values)
            > NORMALIZED_CLIP_VALUE
        )

        print(
            f"  {name}: "
            f"{clipped_fraction * 100:.3f}%"
        )

    # Ограничение применяется одинаково ко всем выборкам.
    # Порог не вычисляется по val или test.
    X_train = np.clip(
        X_train,
        -NORMALIZED_CLIP_VALUE,
        NORMALIZED_CLIP_VALUE,
    )

    X_val = np.clip(
        X_val,
        -NORMALIZED_CLIP_VALUE,
        NORMALIZED_CLIP_VALUE,
    )

    X_test = np.clip(
        X_test,
        -NORMALIZED_CLIP_VALUE,
        NORMALIZED_CLIP_VALUE,
    )

    for name, values in (
        ("train", X_train),
        ("val", X_val),
        ("test", X_test),
    ):
        if not np.isfinite(values).all():
            raise ValueError(
                f"После нормализации в {name} "
                "обнаружены NaN или Inf"
            )

    # Веса классов вычисляются только по train.
    class_counts = np.bincount(
        y_train,
        minlength=len(LABEL_NAMES),
    )

    class_weights = (
        len(y_train)
        / (
            len(LABEL_NAMES)
            * class_counts
        )
    ).astype(np.float32)

    print_distribution(
        "Обучающая выборка",
        y_train,
        groups_train,
    )
    print_distribution(
        "Валидационная выборка",
        y_val,
        groups_val,
    )
    print_distribution(
        "Тестовая выборка",
        y_test,
        groups_test,
    )

    print(
        "\nУдалено постоянных каналов: "
        f"{len(removed_channels)} "
        f"из {len(channels)}"
    )

    print(
        "Оставлено каналов: "
        f"{len(selected_channels)}"
    )

    print(
        "Удалённые каналы: "
        + ", ".join(removed_channels)
    )

    print("\nФормы после отбора каналов:")
    print(f"  Train: {X_train.shape}")
    print(f"  Val:   {X_val.shape}")
    print(f"  Test:  {X_test.shape}")

    print("\nДиапазон нормализованных значений:")

    for name, values in (
        ("Train", X_train),
        ("Val", X_val),
        ("Test", X_test),
    ):
        absolute_values = np.abs(values)

        print(
            f"  {name}: "
            f"99-й процентиль="
            f"{np.quantile(absolute_values, 0.99):.3f}, "
            f"максимум={absolute_values.max():.3f}"
        )

    print("\nВеса классов:")
    for label, weight in enumerate(class_weights):
        print(
            f"  {label} — {LABEL_NAMES[label]}: "
            f"{weight:.4f}"
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "groups_train": groups_train,
        "X_val": X_val,
        "y_val": y_val,
        "groups_val": groups_val,
        "X_test": X_test,
        "y_test": y_test,
        "groups_test": groups_test,
        "class_weights": class_weights,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "channels": selected_channels,
        "removed_channels": removed_channels,
        "metadata": metadata,
        "normalization_clip_value": NORMALIZED_CLIP_VALUE,
    }


if __name__ == "__main__":
    prepare_adapt_data()