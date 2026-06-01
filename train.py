# train.py
# Обучение модифицированной архитектуры PatchTST
# Реализация функции потерь (2.8): L_total = L_CE + λ₁‖W‖₂² + λ₂ L_contrast

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import os
import time
from datetime import datetime

from config import Config
from model import create_model
from synthetic_data import SyntheticDataGenerator, RTKDataset


class ContrastiveLoss(nn.Module):
    """
    Контрастивная компонента функции потерь (InfoNCE)
    Сближает эмбеддинги одного класса и отдаляет разных классов
    
    Реализует стандартную InfoNCE (Supervised Contrastive Learning)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch, num_patches, d_model] - эмбеддинги патчей
            labels: [batch] - метки классов
        Returns:
            contrastive_loss: скаляр
        """
        batch, num_patches, d_model = embeddings.shape
        
        # Усреднение эмбеддингов по патчам
        z = embeddings.mean(dim=1)  # [batch, d_model]
        
        # Нормализация (для стабильности)
        z = nn.functional.normalize(z, dim=1)
        
        # Матрица сходства
        similarity = torch.matmul(z, z.T) / self.temperature  # [batch, batch]
        
        # Маска для положительных пар (одинаковые классы)
        mask_positive = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # Исключаем диагональ (сравнение с самим собой)
        eye_mask = torch.eye(batch, device=embeddings.device)
        mask_positive = mask_positive - eye_mask
        
        # Проверка на наличие положительных пар
        if mask_positive.sum() == 0:
            return torch.tensor(0.0, device=embeddings.device)
        
        exp_sim = torch.exp(similarity)
        
        # Числитель: сумма по положительным парам
        pos_sum = (exp_sim * mask_positive).sum(dim=1)
        
        # Знаменатель: сумма по ВСЕМ парам (кроме диагонали)
        all_sum = (exp_sim * (1 - eye_mask)).sum(dim=1)
        
        # InfoNCE loss
        loss = -torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))
        
        # Усредняем по батчу (исключая нулевые потери)
        loss = loss[loss > 0].mean()
        
        if torch.isnan(loss):
            return torch.tensor(0.0, device=embeddings.device)
        
        return loss


class TotalLoss(nn.Module):
    """
    Полная функция потерь
    Формула (2.8): L_total = L_CE + λ₁‖W‖₂² + λ₂ L_contrast
    """
    def __init__(self, lambda1=1e-4, lambda2=0.1, temperature=0.07):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        
    def forward(self, predictions, targets, embeddings, model_parameters):
        """
        Args:
            predictions: [batch, num_classes] - выход модели
            targets: [batch] - истинные метки
            embeddings: [batch, num_patches, d_model] - эмбеддинги патчей
            model_parameters: параметры модели для L2-регуляризации
        Returns:
            total_loss, ce, l2_reg, contrast
        """
        # Кросс-энтропийная потеря L_CE
        ce = self.ce_loss(predictions, targets)
        
        # L2-регуляризация λ₁‖W‖₂² (квадрат нормы)
        l2_reg = self.lambda1 * sum(p.norm(2)**2 for p in model_parameters)
        
        # Контрастивная потеря λ₂ L_contrast
        contrast = self.lambda2 * self.contrastive_loss(embeddings, targets)
        
        total_loss = ce + l2_reg + contrast
        
        return total_loss, ce, l2_reg, contrast


class Trainer:
    """Класс для обучения модели"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        
        # Оптимизатор Adam (L2-регуляризация уже в функции потерь)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=0
        )
        
        # Планировщик скорости обучения (ReduceLROnPlateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5      
        )
        
        # Функция потерь (формула 2.8)
        self.criterion = TotalLoss(
            lambda1=config.LAMBDA_1,
            lambda2=config.LAMBDA_2,
            temperature=config.TEMPERATURE
        )
        
        # История обучения
        self.history = {
            'train_loss': [],
            'train_ce': [],
            'train_l2': [],
            'train_contrast': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Создание директории для сохранения моделей
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
        os.makedirs(config.LOG_PATH, exist_ok=True)
        
        # Лучшая метрика
        self.best_val_acc = 0
        self.best_val_loss = float('inf')
        
    def train_epoch(self, train_loader):
        """Обучение одной эпохи"""
        self.model.train()
        total_loss = 0
        total_ce = 0
        total_l2 = 0
        total_contrast = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(self.device)
            targets = targets.to(self.device)
            
            # Прямой проход (с маскированием на обучении)
            self.optimizer.zero_grad()
            predictions, embeddings = self.model(data, use_masking=True)
            
            # Вычисление потерь (формула 2.8)
            loss, ce, l2, contrast = self.criterion(
                predictions, targets, embeddings, self.model.parameters()
            )
            
            # Обратный проход
            loss.backward()
            
            # Клиппинг градиентов для стабильности
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Статистика
            total_loss += loss.item()
            total_ce += ce.item()
            total_l2 += l2.item()
            total_contrast += contrast.item()
            
            _, predicted = predictions.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Прогресс
            if (batch_idx + 1) % 20 == 0:
                print(f'    Батч {batch_idx + 1}/{len(train_loader)} | '
                      f'Loss: {loss.item():.4f} | Acc: {100.*correct/total:.2f}%')
        
        avg_loss = total_loss / len(train_loader)
        avg_ce = total_ce / len(train_loader)
        avg_l2 = total_l2 / len(train_loader)
        avg_contrast = total_contrast / len(train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, avg_ce, avg_l2, avg_contrast, accuracy
    
    def validate(self, val_loader):
        """Валидация"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(self.device)
                targets = targets.to(self.device)
                
                # Прямой проход (без маскирования на валидации)
                predictions, embeddings = self.model(data, use_masking=False)
                
                # Вычисление потерь
                loss, ce, l2, contrast = self.criterion(
                    predictions, targets, embeddings, self.model.parameters()
                )
                
                total_loss += loss.item()
                
                _, predicted = predictions.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train(self, train_loader, val_loader, num_epochs):
        """Полный цикл обучения"""
        print("\n" + "=" * 60)
        print("НАЧАЛО ОБУЧЕНИЯ МОДЕЛИ")
        print("=" * 60)
        print(f"Устройство: {self.device}")
        print(f"Количество эпох: {num_epochs}")
        print(f"Размер батча: {self.config.BATCH_SIZE}")
        print(f"Начальная скорость обучения: {self.config.LEARNING_RATE}")
        print(f"λ₁ (L2): {self.config.LAMBDA_1}")
        print(f"λ₂ (контраст): {self.config.LAMBDA_2}")
        print(f"Температура τ: {self.config.TEMPERATURE}")
        print(f"Вероятность маскирования: {self.config.MASK_PROB}")
        print("=" * 60)
        
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Обучение
            train_loss, train_ce, train_l2, train_contrast, train_acc = self.train_epoch(train_loader)
            
            # Валидация
            val_loss, val_acc = self.validate(val_loader)
            
            # Обновление планировщика
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_ce'].append(train_ce)
            self.history['train_l2'].append(train_l2)
            self.history['train_contrast'].append(train_contrast)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            # Вывод результатов эпохи
            print(f"\nЭпоха {epoch}/{num_epochs} | Время: {epoch_time:.1f}с | LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, L2: {train_l2:.4f}, Contrast: {train_contrast:.4f})")
            print(f"  Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            # Сохранение лучшей модели (по валидационной точности)
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_model(self.config.MODEL_SAVE_PATH)
                patience_counter = 0
                print(f"  >>> Новая лучшая модель сохранена (Acc: {val_acc:.2f}%, Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nРанняя остановка на эпохе {epoch}. Лучшая Val Acc: {self.best_val_acc:.2f}%")
                break
        
        total_time = time.time() - start_time
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
        print(f"Общее время: {total_time / 60:.1f} минут")
        print(f"Лучшая точность на валидации: {self.best_val_acc:.2f}%")
        print(f"Лучшая потеря на валидации: {self.best_val_loss:.4f}")
        print(f"Модель сохранена: {self.config.MODEL_SAVE_PATH}")
        print("=" * 60)
        
        return self.best_val_acc
    
    def save_model(self, path):
        """Сохранение модели и конфигурации"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'config': {
                'WINDOW_LENGTH': self.config.WINDOW_LENGTH,
                'NUM_CHANNELS': self.config.NUM_CHANNELS,
                'NUM_CLASSES': self.config.NUM_CLASSES,
                'PATCH_LEN': self.config.PATCH_LEN,
                'PATCH_STRIDE': self.config.PATCH_STRIDE,
                'D_MODEL': self.config.D_MODEL,
                'NUM_HEADS': self.config.NUM_HEADS,
                'NUM_LAYERS': self.config.NUM_LAYERS,
                'LAMBDA_1': self.config.LAMBDA_1,
                'LAMBDA_2': self.config.LAMBDA_2,
                'TEMPERATURE': self.config.TEMPERATURE,
                'MASK_PROB': self.config.MASK_PROB
            }
        }, path)
        print(f"Модель сохранена: {path}")
    
    def load_model(self, path):
        """Загрузка модели"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Модель загружена: {path}")
        print(f"Лучшая точность валидации: {self.best_val_acc:.2f}%")
        return checkpoint


def prepare_data(config, samples_per_class=24):
    """
    Подготовка данных для обучения
    
    Args:
        config: объект конфигурации
        samples_per_class: количество синтетических образцов на класс
    """
    print("\n" + "=" * 50)
    print("ПОДГОТОВКА ДАННЫХ")
    print("=" * 50)
    
    # Генерация синтетических данных
    generator = SyntheticDataGenerator(config)
    X, y = generator.generate_dataset(samples_per_class=samples_per_class)
    
    # Разделение на обучающую и валидационную выборки (80% / 20%)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=config.RANDOM_SEED
    )
    
    # Создание датасетов
    train_dataset = RTKDataset(X_train, y_train, config, augment=True)
    val_dataset = RTKDataset(X_val, y_val, config, augment=False)
    
    # DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Обучающая выборка: {len(train_dataset)} образцов")
    print(f"Валидационная выборка: {len(val_dataset)} образцов")
    print(f"Размер батча: {config.BATCH_SIZE}")
    print(f"Количество батчей на эпоху: {len(train_loader)}")
    
    return train_loader, val_loader


def prepare_large_dataset(config, samples_per_class=2000):
    """
    Подготовка большого датасета для отладки (глава 2)
    
    Args:
        config: объект конфигурации
        samples_per_class: количество синтетических образцов на класс (по умолчанию 2000)
    """
    print("\n" + "=" * 50)
    print("ПОДГОТОВКА БОЛЬШОГО ДАТАСЕТА (ГЛАВА 2)")
    print("=" * 50)
    
    # Генерация большого синтетического датасета
    generator = SyntheticDataGenerator(config)
    X, y = generator.generate_large_dataset(samples_per_class=samples_per_class)
    
    # Разделение на обучающую, валидационную и тестовую выборки
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=config.RANDOM_SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=config.RANDOM_SEED
    )
    
    # Создание датасетов
    train_dataset = RTKDataset(X_train, y_train, config, augment=True)
    val_dataset = RTKDataset(X_val, y_val, config, augment=False)
    test_dataset = RTKDataset(X_test, y_test, config, augment=False)
    
    # DataLoader'ы
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    
    print(f"Обучающая выборка: {len(train_dataset)} образцов")
    print(f"Валидационная выборка: {len(val_dataset)} образцов")
    print(f"Тестовая выборка: {len(test_dataset)} образцов")
    
    return train_loader, val_loader, test_loader


def main():
    """Основная функция"""
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ НЕЙРОСЕТЕВОГО МОДУЛЯ PATCHTST")
    print("Специальность 2.3.5 - Математическое и программное обеспечение")
    print("=" * 60)
    
    # Фиксация random seed для воспроизводимости
    Config.set_seed()
    
    # Создание необходимых директорий
    Config.ensure_dirs()
    
    # Вывод конфигурации
    Config.print_config()
    Config.validate()
    
    # Параметры синтетических данных
    samples_per_class = getattr(Config, 'SYNTHETIC_SAMPLES_PER_CLASS', 24)
    
    # Подготовка данных
    train_loader, val_loader = prepare_data(Config, samples_per_class=samples_per_class)
    
    # Создание модели
    model = create_model(Config)
    
    # Создание тренера
    trainer = Trainer(model, Config)
    
    # Обучение
    best_acc = trainer.train(train_loader, val_loader, Config.NUM_EPOCHS)
    
    # Сохранение истории в CSV
    import pandas as pd
    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(f"{Config.LOG_PATH}/training_history.csv", index=False)
    print(f"\nИстория обучения сохранена: {Config.LOG_PATH}/training_history.csv")
    
    # Вывод итоговой статистики
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ СТАТИСТИКА")
    print("=" * 60)
    print(f"Лучшая точность на валидации: {trainer.best_val_acc:.2f}%")
    print(f"Финальная точность на валидации: {trainer.history['val_acc'][-1]:.2f}%")
    print(f"Финальная потеря на валидации: {trainer.history['val_loss'][-1]:.4f}")
    
    return trainer


def main_large():
    """
    Основная функция для обучения на большом датасете (глава 2)
    """
    print("\n" + "=" * 60)
    print("ОБУЧЕНИЕ НА БОЛЬШОМ ДАТАСЕТЕ (ГЛАВА 2)")
    print("=" * 60)
    
    # Фиксация random seed
    Config.set_seed()
    Config.ensure_dirs()
    
    # Вывод конфигурации
    Config.print_config()
    Config.validate()
    
    # Параметры большого датасета
    samples_per_class = getattr(Config, 'LARGE_SAMPLES_PER_CLASS', 2000)
    
    # Подготовка данных
    train_loader, val_loader, test_loader = prepare_large_dataset(
        Config, samples_per_class=samples_per_class
    )
    
    # Создание модели
    model = create_model(Config)
    
    # Создание тренера
    trainer = Trainer(model, Config)
    
    # Обучение
    best_acc = trainer.train(train_loader, val_loader, Config.NUM_EPOCHS)
    
    # Оценка на тестовой выборке
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"\nОценка на тестовой выборке:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc: {test_acc:.2f}%")
    
    # Сохранение истории
    import pandas as pd
    history_df = pd.DataFrame(trainer.history)
    history_df.to_csv(f"{Config.LOG_PATH}/training_history_large.csv", index=False)
    
    return trainer


if __name__ == '__main__':
    # Запуск обучения на небольшом датасете (глава 3)
    trainer = main()
    
    # Раскомментировать для обучения на большом датасете (глава 2)
    # trainer_large = main_large()