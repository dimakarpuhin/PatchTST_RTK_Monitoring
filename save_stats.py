# save_stats.py
import numpy as np
from config import Config
from sklearn.preprocessing import StandardScaler

# Загружаем данные
X_train = np.load(f'{Config.PM_PROCESSED_PATH}/X_train.npy')
X_train_flat = X_train.reshape(-1, Config.NUM_CHANNELS)

# Обучаем scaler
scaler = StandardScaler()
scaler.fit(X_train_flat)

# Сохраняем
np.savez('models/patchtst_model_stats.npz', 
         mean=scaler.mean_, 
         std=scaler.scale_)

print("✅ Статистики нормализации сохранены в models/patchtst_model_stats.npz")
