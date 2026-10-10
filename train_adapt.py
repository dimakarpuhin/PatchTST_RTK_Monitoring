import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from model import create_model
from prepare_adapt_data import prepare_adapt_data


def create_loader(X, y, batch_size, shuffle):
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )

    generator = torch.Generator()
    generator.manual_seed(Config.RANDOM_SEED)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
        num_workers=0,
    )


def evaluate(model, loader, criterion, device):
    model.eval()

    losses = []
    targets_all = []
    predictions_all = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            logits, _ = model(
                data,
                use_masking=False,
            )

            loss = criterion(logits, targets)
            predictions = logits.argmax(dim=1)

            losses.append(loss.item())
            targets_all.extend(
                targets.cpu().numpy()
            )
            predictions_all.extend(
                predictions.cpu().numpy()
            )

    targets_all = np.asarray(targets_all)
    predictions_all = np.asarray(predictions_all)

    macro_f1 = f1_score(
        targets_all,
        predictions_all,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    accuracy = accuracy_score(
        targets_all,
        predictions_all,
    )

    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "targets": targets_all,
        "predictions": predictions_all,
    }


def main():
    Config.set_seed()

    prepared = prepare_adapt_data()

    X_train = prepared["X_train"]
    y_train = prepared["y_train"]
    X_val = prepared["X_val"]
    y_val = prepared["y_val"]
    X_test = prepared["X_test"]
    y_test = prepared["y_test"]
    groups_test = prepared["groups_test"]

    # Конфигурация первого диагностического запуска.
    Config.WINDOW_LENGTH = X_train.shape[1]
    Config.NUM_CHANNELS = X_train.shape[2]
    Config.NUM_CLASSES = 3

    Config.PATCH_LEN = 4
    Config.PATCH_STRIDE = 2

    # Размер определяется числом оставшихся каналов.
    # 130 кратно 65 каналам и двум головам внимания.
    Config.D_MODEL = (
        2 * Config.NUM_CHANNELS
    )
    Config.NUM_HEADS = 2
    Config.NUM_LAYERS = 2

    Config.DROPOUT = 0.3
    Config.MASK_PROB = 0.15
    Config.USE_ADAPTIVE_ENCODING = True
    Config.USE_CHANNEL_ATTENTION = True

    Config.BATCH_SIZE = 16
    Config.LEARNING_RATE = 3e-4
    Config.NUM_EPOCHS = 60
    Config.EARLY_STOPPING_PATIENCE = 12

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

    history = []
    best_state = None
    best_val_f1 = -1.0
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    print("\nНачало обучения")
    print(f"Устройство: {device}")
    print(f"Число патчей: {Config.get_num_patches()}")
    print(f"Число параметров: "
          f"{sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, Config.NUM_EPOCHS + 1):
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

            loss = criterion(logits, targets)
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

        train_f1 = f1_score(
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

        scheduler.step(validation["macro_f1"])

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_result = {
            "epoch": epoch,
            "train_loss": float(
                np.mean(train_losses)
            ),
            "train_macro_f1": float(train_f1),
            "val_loss": validation["loss"],
            "val_accuracy": validation["accuracy"],
            "val_macro_f1": validation["macro_f1"],
            "learning_rate": current_lr,
        }

        history.append(epoch_result)

        print(
            f"Эпоха {epoch:02d}: "
            f"train loss={epoch_result['train_loss']:.4f}, "
            f"train F1={train_f1:.4f}, "
            f"val loss={validation['loss']:.4f}, "
            f"val F1={validation['macro_f1']:.4f}, "
            f"val acc={validation['accuracy']:.4f}"
        )

        improved = (
            validation["macro_f1"] > best_val_f1
            or (
                validation["macro_f1"]
                == best_val_f1
                and validation["loss"] < best_val_loss
            )
        )

        if improved:
            best_val_f1 = validation["macro_f1"]
            best_val_loss = validation["loss"]
            best_epoch = epoch
            best_state = copy.deepcopy(
                model.state_dict()
            )
            patience_counter = 0
        else:
            patience_counter += 1

        if (
            patience_counter
            >= Config.EARLY_STOPPING_PATIENCE
        ):
            print(
                "Ранняя остановка. "
                f"Лучшая эпоха: {best_epoch}"
            )
            break

    if best_state is None:
        raise RuntimeError(
            "Не удалось сохранить состояние модели"
        )

    # Возвращаем лучшую эпоху.
    model.load_state_dict(best_state)

    test_result = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    balanced_accuracy = balanced_accuracy_score(
        test_result["targets"],
        test_result["predictions"],
    )

    matrix = confusion_matrix(
        test_result["targets"],
        test_result["predictions"],
        labels=[0, 1, 2],
    )

    print("\nРезультаты тестирования:")
    print(f"Лучшая эпоха: {best_epoch}")
    print(
        f"Accuracy: "
        f"{test_result['accuracy']:.4f}"
    )
    print(
        f"Balanced accuracy: "
        f"{balanced_accuracy:.4f}"
    )
    print(
        f"Macro-F1: "
        f"{test_result['macro_f1']:.4f}"
    )

    print("\nМатрица ошибок:")
    print(matrix)
    print("Строки — истинные классы")
    print("Столбцы — предсказанные классы")

    print("\nОтчёт по классам:")
    print(
        classification_report(
            test_result["targets"],
            test_result["predictions"],
            labels=[0, 1, 2],
            target_names=[
                "Норма",
                "Отказ датчика",
                "Отказ компонента",
            ],
            zero_division=0,
            digits=4,
        )
    )

    print("\nРезультаты по экспериментам:")

    for experiment_id in sorted(set(groups_test)):
        mask = groups_test == experiment_id

        experiment_accuracy = accuracy_score(
            test_result["targets"][mask],
            test_result["predictions"][mask],
        )

        experiment_f1 = f1_score(
            test_result["targets"][mask],
            test_result["predictions"][mask],
            average="macro",
            zero_division=0,
        )

        print(
            f"  {experiment_id}: "
            f"accuracy={experiment_accuracy:.4f}, "
            f"macro-F1={experiment_f1:.4f}, "
            f"окон={int(mask.sum())}"
        )

    reports_dir = Path("reports")
    models_dir = Path("models")

    reports_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    model_path = (
        models_dir
        / "adapt_patchtst_15exp_pilot.pth"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_val_f1,
            "channels": prepared["channels"],
        },
        model_path,
    )

    with (
        reports_dir
        / "adapt_15exp_pilot_history.csv"
    ).open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=history[0].keys(),
        )
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_result["accuracy"],
        "test_balanced_accuracy": float(
            balanced_accuracy
        ),
        "test_macro_f1": test_result["macro_f1"],
        "confusion_matrix": matrix.tolist(),
    }

    with (
        reports_dir
        / "adapt_15exp_pilot_metrics.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nМодель сохранена: {model_path}")
    print(
        "Метрики сохранены: "
        "reports/adapt_15exp_pilot_metrics.json"
    )


if __name__ == "__main__":
    main()