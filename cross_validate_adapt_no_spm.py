import copy
import csv
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
)

from config import Config
from load_adapt_data import load_adapt_data
from model import create_model
from prepare_adapt_data import prepare_adapt_data
from train_adapt import create_loader, evaluate


NUMBER_OF_FOLDS = 5


def configure_model(prepared):
    X_train = prepared["X_train"]

    Config.WINDOW_LENGTH = X_train.shape[1]
    Config.NUM_CHANNELS = X_train.shape[2]
    Config.NUM_CLASSES = 3

    Config.PATCH_LEN = 4
    Config.PATCH_STRIDE = 2

    Config.D_MODEL = 2 * Config.NUM_CHANNELS
    Config.NUM_HEADS = 2
    Config.NUM_LAYERS = 2

    Config.DROPOUT = 0.3
    Config.MASK_PROB = 0.0
    Config.USE_ADAPTIVE_ENCODING = True
    Config.USE_CHANNEL_ATTENTION = True

    Config.BATCH_SIZE = 16
    Config.LEARNING_RATE = 3e-4
    Config.NUM_EPOCHS = 60
    Config.EARLY_STOPPING_PATIENCE = 12


def calculate_metrics(targets, predictions):
    binary_targets = (
        targets != 0
    ).astype(np.int64)

    binary_predictions = (
        predictions != 0
    ).astype(np.int64)

    return {
        "multiclass_accuracy": float(
            accuracy_score(
                targets,
                predictions,
            )
        ),
        "multiclass_balanced_accuracy": float(
            balanced_accuracy_score(
                targets,
                predictions,
            )
        ),
        "multiclass_macro_f1": float(
            f1_score(
                targets,
                predictions,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "multiclass_confusion_matrix": (
            confusion_matrix(
                targets,
                predictions,
                labels=[0, 1, 2],
            ).tolist()
        ),
        "binary_accuracy": float(
            accuracy_score(
                binary_targets,
                binary_predictions,
            )
        ),
        "binary_balanced_accuracy": float(
            balanced_accuracy_score(
                binary_targets,
                binary_predictions,
            )
        ),
        "binary_precision": float(
            precision_score(
                binary_targets,
                binary_predictions,
                zero_division=0,
            )
        ),
        "binary_recall": float(
            recall_score(
                binary_targets,
                binary_predictions,
                zero_division=0,
            )
        ),
        "binary_f1": float(
            f1_score(
                binary_targets,
                binary_predictions,
                zero_division=0,
            )
        ),
        "binary_confusion_matrix": (
            confusion_matrix(
                binary_targets,
                binary_predictions,
                labels=[0, 1],
            ).tolist()
        ),
    }


def train_one_fold(prepared, fold_number):
    Config.set_seed()
    configure_model(prepared)

    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]

    train_loader = create_loader(
        X_train,
        y_train,
        Config.BATCH_SIZE,
        shuffle=True,
    )

    val_loader = create_loader(
        X_val,
        y_val,
        Config.BATCH_SIZE,
        shuffle=False,
    )

    test_loader = create_loader(
        X_test,
        y_test,
        Config.BATCH_SIZE,
        shuffle=False,
    )

    model = create_model(Config)
    device = Config.DEVICE

    class_weights = torch.tensor(
        prepared["class_weights"],
        dtype=torch.float32,
        device=device,
    )

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=1e-4,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=5,
        )
    )

    best_state = None
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = []

    number_of_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"Каналов: {Config.NUM_CHANNELS}, "
        f"параметров: {number_of_parameters:,}"
    )

    for epoch in range(
        1,
        Config.NUM_EPOCHS + 1,
    ):
        model.train()

        train_losses = []
        train_targets = []
        train_predictions = []

        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits, _ = model(
                data,
                use_masking=True,
            )

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            optimizer.step()

            predictions = logits.argmax(dim=1)

            train_losses.append(loss.item())
            train_targets.extend(
                targets.cpu().numpy()
            )
            train_predictions.extend(
                predictions.cpu().numpy()
            )

        train_macro_f1 = f1_score(
            train_targets,
            train_predictions,
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )

        validation = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(
            validation["macro_f1"]
        )

        epoch_row = {
            "fold": fold_number,
            "epoch": epoch,
            "train_loss": float(
                np.mean(train_losses)
            ),
            "train_macro_f1": float(
                train_macro_f1
            ),
            "val_loss": validation["loss"],
            "val_accuracy": (
                validation["accuracy"]
            ),
            "val_macro_f1": (
                validation["macro_f1"]
            ),
            "learning_rate": (
                optimizer.param_groups[0]["lr"]
            ),
        }

        history.append(epoch_row)

        improved = (
            validation["macro_f1"]
            > best_val_f1
            or (
                validation["macro_f1"]
                == best_val_f1
                and validation["loss"]
                < best_val_loss
            )
        )

        if improved:
            best_val_f1 = validation[
                "macro_f1"
            ]
            best_val_loss = validation["loss"]
            best_epoch = epoch

            # Лучшее состояние переносится на CPU,
            # чтобы не занимать память видеокарты.
            best_state = {
                name: value.detach().cpu().clone()
                for name, value
                in model.state_dict().items()
            }

            patience_counter = 0

            print(
                f"  эпоха {epoch:02d}: "
                f"новый лучший val Macro-F1="
                f"{best_val_f1:.4f}"
            )
        else:
            patience_counter += 1

        if (
            patience_counter
            >= Config.EARLY_STOPPING_PATIENCE
        ):
            print(
                f"  ранняя остановка: "
                f"эпоха {epoch}, "
                f"лучшая эпоха {best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            f"Fold {fold_number}: "
            "лучшая модель не сохранена"
        )

    model.load_state_dict(best_state)

    test_result = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    metrics = calculate_metrics(
        test_result["targets"],
        test_result["predictions"],
    )

    metrics.update({
        "fold": fold_number,
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(
            best_val_f1
        ),
        "number_of_channels": (
            Config.NUM_CHANNELS
        ),
        "number_of_parameters": (
            number_of_parameters
        ),
    })

    targets = test_result["targets"]
    predictions = test_result["predictions"]
    groups = prepared["groups_test"].copy()

    del model
    del optimizer
    del criterion

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (
        metrics,
        history,
        targets,
        predictions,
        groups,
    )


def write_csv(path, rows):
    if not rows:
        return

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    Config.set_seed()

    # Загружаем данные один раз только для
    # построения групповых разбиений.
    with redirect_stdout(io.StringIO()):
        X_all, y_all, groups_all, _, _ = (
            load_adapt_data()
        )

    splitter = StratifiedGroupKFold(
        n_splits=NUMBER_OF_FOLDS,
        shuffle=True,
        random_state=Config.RANDOM_SEED,
    )

    folds = []

    for _, test_indices in splitter.split(
        X_all,
        y_all,
        groups=groups_all,
    ):
        fold_ids = {
            str(experiment_id)
            for experiment_id in np.unique(
                groups_all[test_indices]
            )
        }

        folds.append(fold_ids)

    all_experiment_ids = {
        str(experiment_id)
        for experiment_id
        in np.unique(groups_all)
    }

    fold_union = set().union(*folds)

    if fold_union != all_experiment_ids:
        raise ValueError(
            "Не все эксперименты вошли "
            "в перекрёстную проверку"
        )

    for left_index in range(len(folds)):
        for right_index in range(
            left_index + 1,
            len(folds),
        ):
            if folds[left_index] & folds[right_index]:
                raise ValueError(
                    "Эксперименты пересекаются "
                    "между тестовыми fold"
                )

    print("Групповые разбиения:")

    for fold_index, fold_ids in enumerate(
        folds,
        start=1,
    ):
        mask = np.isin(
            groups_all,
            list(fold_ids),
        )

        counts = np.bincount(
            y_all[mask],
            minlength=3,
        )

        print(
            f"  Fold {fold_index}: "
            f"{sorted(fold_ids)}; "
            f"окна={int(mask.sum())}; "
            f"классы={counts.tolist()}"
        )

    fold_rows = []
    history_rows = []

    oof_targets = []
    oof_predictions = []
    oof_groups = []
    oof_folds = []

    for fold_index in range(NUMBER_OF_FOLDS):
        test_ids = folds[fold_index]

        # Следующий fold используется только
        # для ранней остановки.
        val_ids = folds[
            (fold_index + 1)
            % NUMBER_OF_FOLDS
        ]

        train_ids = (
            all_experiment_ids
            - test_ids
            - val_ids
        )

        print("\n" + "=" * 70)
        print(
            f"FOLD {fold_index + 1} "
            f"ИЗ {NUMBER_OF_FOLDS}"
        )
        print(
            f"Train: {sorted(train_ids)}"
        )
        print(
            f"Val:   {sorted(val_ids)}"
        )
        print(
            f"Test:  {sorted(test_ids)}"
        )
        print("=" * 70)

        # Подробный вывод подготовки скрываем,
        # чтобы журнал оставался читаемым.
        with redirect_stdout(io.StringIO()):
            prepared = prepare_adapt_data(
                train_ids=train_ids,
                val_ids=val_ids,
                test_ids=test_ids,
            )

        train_counts = np.bincount(
            prepared["y_train"],
            minlength=3,
        )

        val_counts = np.bincount(
            prepared["y_val"],
            minlength=3,
        )

        test_counts = np.bincount(
            prepared["y_test"],
            minlength=3,
        )

        print(
            f"Train классы: "
            f"{train_counts.tolist()}"
        )
        print(
            f"Val классы:   "
            f"{val_counts.tolist()}"
        )
        print(
            f"Test классы:  "
            f"{test_counts.tolist()}"
        )

        (
            fold_metrics,
            fold_history,
            targets,
            predictions,
            test_groups,
        ) = train_one_fold(
            prepared,
            fold_index + 1,
        )

        fold_metrics["train_ids"] = (
            ",".join(sorted(train_ids))
        )
        fold_metrics["val_ids"] = (
            ",".join(sorted(val_ids))
        )
        fold_metrics["test_ids"] = (
            ",".join(sorted(test_ids))
        )

        print(
            "Трёхклассовый результат: "
            f"Macro-F1="
            f"{fold_metrics['multiclass_macro_f1']:.4f}, "
            f"balanced accuracy="
            f"{fold_metrics['multiclass_balanced_accuracy']:.4f}"
        )

        print(
            "Бинарное обнаружение: "
            f"F1={fold_metrics['binary_f1']:.4f}, "
            f"recall={fold_metrics['binary_recall']:.4f}, "
            f"balanced accuracy="
            f"{fold_metrics['binary_balanced_accuracy']:.4f}"
        )

        fold_rows.append(fold_metrics)
        history_rows.extend(fold_history)

        oof_targets.append(targets)
        oof_predictions.append(predictions)
        oof_groups.append(test_groups)
        oof_folds.append(
            np.full(
                len(targets),
                fold_index + 1,
                dtype=np.int64,
            )
        )

    oof_targets = np.concatenate(oof_targets)
    oof_predictions = np.concatenate(
        oof_predictions
    )
    oof_groups = np.concatenate(oof_groups)
    oof_folds = np.concatenate(oof_folds)

    if len(oof_targets) != len(y_all):
        raise ValueError(
            "Число вневыборочных предсказаний "
            "не совпало с числом окон"
        )

    global_metrics = calculate_metrics(
        oof_targets,
        oof_predictions,
    )

    print("\n" + "=" * 70)
    print("ИТОГОВЫЕ ВНЕВЫБОРОЧНЫЕ РЕЗУЛЬТАТЫ")
    print("=" * 70)

    print("\nТрёхклассовая задача:")
    print(
        f"  Accuracy: "
        f"{global_metrics['multiclass_accuracy']:.4f}"
    )
    print(
        f"  Balanced accuracy: "
        f"{global_metrics['multiclass_balanced_accuracy']:.4f}"
    )
    print(
        f"  Macro-F1: "
        f"{global_metrics['multiclass_macro_f1']:.4f}"
    )
    print(
        "  Матрица ошибок:"
    )
    print(
        np.asarray(
            global_metrics[
                "multiclass_confusion_matrix"
            ]
        )
    )

    print("\nБинарное обнаружение отказа:")
    print(
        f"  Accuracy: "
        f"{global_metrics['binary_accuracy']:.4f}"
    )
    print(
        f"  Balanced accuracy: "
        f"{global_metrics['binary_balanced_accuracy']:.4f}"
    )
    print(
        f"  Precision: "
        f"{global_metrics['binary_precision']:.4f}"
    )
    print(
        f"  Recall: "
        f"{global_metrics['binary_recall']:.4f}"
    )
    print(
        f"  F1: "
        f"{global_metrics['binary_f1']:.4f}"
    )
    print(
        "  Матрица ошибок:"
    )
    print(
        np.asarray(
            global_metrics[
                "binary_confusion_matrix"
            ]
        )
    )

    print("\nОтчёт по трём классам:")
    print(
        classification_report(
            oof_targets,
            oof_predictions,
            labels=[0, 1, 2],
            target_names=[
                "Норма",
                "Отказ датчика",
                "Отказ компонента",
            ],
            digits=4,
            zero_division=0,
        )
    )

    experiment_rows = []

    for experiment_id in sorted(
        set(oof_groups),
        key=int,
    ):
        mask = oof_groups == experiment_id

        experiment_metrics = calculate_metrics(
            oof_targets[mask],
            oof_predictions[mask],
        )

        fault_classes = sorted(
            set(oof_targets[mask]) - {0}
        )

        experiment_rows.append({
            "experiment_id": experiment_id,
            "fold": int(
                np.unique(oof_folds[mask])[0]
            ),
            "window_count": int(mask.sum()),
            "fault_class": (
                fault_classes[0]
                if fault_classes
                else ""
            ),
            "multiclass_accuracy": (
                experiment_metrics[
                    "multiclass_accuracy"
                ]
            ),
            "multiclass_macro_f1": (
                experiment_metrics[
                    "multiclass_macro_f1"
                ]
            ),
            "binary_accuracy": (
                experiment_metrics[
                    "binary_accuracy"
                ]
            ),
            "binary_f1": (
                experiment_metrics[
                    "binary_f1"
                ]
            ),
            "binary_recall": (
                experiment_metrics[
                    "binary_recall"
                ]
            ),
        })

    experiment_multiclass_f1 = np.array([
        row["multiclass_macro_f1"]
        for row in experiment_rows
    ])

    experiment_binary_f1 = np.array([
        row["binary_f1"]
        for row in experiment_rows
    ])

    experiment_binary_recall = np.array([
        row["binary_recall"]
        for row in experiment_rows
    ])

    experiment_summary = {
        "multiclass_macro_f1_mean": float(
            experiment_multiclass_f1.mean()
        ),
        "multiclass_macro_f1_std": float(
            experiment_multiclass_f1.std(
                ddof=1
            )
        ),
        "binary_f1_mean": float(
            experiment_binary_f1.mean()
        ),
        "binary_f1_std": float(
            experiment_binary_f1.std(
                ddof=1
            )
        ),
        "binary_recall_mean": float(
            experiment_binary_recall.mean()
        ),
        "binary_recall_std": float(
            experiment_binary_recall.std(
                ddof=1
            )
        ),
    }

    print("\nСредние метрики по экспериментам:")
    print(
        "  Трёхклассовый Macro-F1: "
        f"{experiment_summary['multiclass_macro_f1_mean']:.4f} "
        f"± "
        f"{experiment_summary['multiclass_macro_f1_std']:.4f}"
    )
    print(
        "  Бинарный F1: "
        f"{experiment_summary['binary_f1_mean']:.4f} "
        f"± "
        f"{experiment_summary['binary_f1_std']:.4f}"
    )
    print(
        "  Бинарный recall: "
        f"{experiment_summary['binary_recall_mean']:.4f} "
        f"± "
        f"{experiment_summary['binary_recall_std']:.4f}"
    )

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    write_csv(
        reports_dir
        / "adapt_chapter2_no_spm_cv_fold_metrics.csv",
        fold_rows,
    )

    write_csv(
        reports_dir
        / "adapt_chapter2_no_spm_cv_history.csv",
        history_rows,
    )

    write_csv(
        reports_dir
        / "adapt_chapter2_no_spm_cv_experiment_metrics.csv",
        experiment_rows,
    )

    prediction_rows = []

    for fold, experiment_id, target, prediction in zip(
        oof_folds,
        oof_groups,
        oof_targets,
        oof_predictions,
    ):
        prediction_rows.append({
            "fold": int(fold),
            "experiment_id": experiment_id,
            "target": int(target),
            "prediction": int(prediction),
            "binary_target": int(target != 0),
            "binary_prediction": int(
                prediction != 0
            ),
        })

    write_csv(
        reports_dir
        / "adapt_chapter2_no_spm_cv_predictions.csv",
        prediction_rows,
    )

    summary = {
        "number_of_folds": NUMBER_OF_FOLDS,
        "number_of_experiments": len(
            all_experiment_ids
        ),
        "number_of_oof_windows": len(
            oof_targets
        ),
        "global_metrics": global_metrics,
        "experiment_summary": experiment_summary,
    }

    with (
        reports_dir
        / "adapt_chapter2_no_spm_cv_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\nРезультаты сохранены:")
    print("  reports/adapt_chapter2_no_spm_cv_fold_metrics.csv")
    print("  reports/adapt_chapter2_no_spm_cv_history.csv")
    print("  reports/adapt_chapter2_no_spm_cv_experiment_metrics.csv")
    print("  reports/adapt_chapter2_no_spm_cv_predictions.csv")
    print("  reports/adapt_chapter2_no_spm_cv_summary.json")


if __name__ == "__main__":
    main()