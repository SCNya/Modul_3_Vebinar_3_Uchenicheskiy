# Тестирование улучшений параметров на валидации
# Запустите этот скрипт перед обучением финальной модели

import pandas as pd
from catboost import CatBoostClassifier
import sys
import os

# Переходим в папку с данными
os.chdir("stud")

# Импортируем метрику
from metric import ndcg_at_5 as pndcg_at_5

# Загружаем данные (предполагаем, что уже выполнены предыдущие ячейки)
print("=" * 60)
print("ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ НА ВАЛИДАЦИИ")
print("=" * 60)

# Baseline: depth=3, iterations=200, learning_rate=0.05
print("\n1. Baseline (текущий лучший результат)")
print("   depth=3, iterations=200, learning_rate=0.05")
baseline_metric = 0.644620
print(f"   Метрика: {baseline_metric:.6f}")

# Будем тестировать разные конфигурации
configs = [
    {
        "name": "Увеличение iterations до 500",
        "params": {
            "iterations": 500,
            "depth": 3,
            "learning_rate": 0.04,
            "l2_leaf_reg": 2
        }
    },
    {
        "name": "Увеличение iterations до 800",
        "params": {
            "iterations": 800,
            "depth": 3,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3
        }
    },
    {
        "name": "Depth=4 + iterations=500",
        "params": {
            "iterations": 500,
            "depth": 4,
            "learning_rate": 0.04,
            "l2_leaf_reg": 2
        }
    },
    {
        "name": "Консервативная настройка",
        "params": {
            "iterations": 400,
            "depth": 3,
            "learning_rate": 0.045,
            "l2_leaf_reg": 1
        }
    }
]

print("\n" + "=" * 60)
print("РЕЗУЛЬТАТЫ:")
print("=" * 60)
print(f"{'Конфигурация':<40} {'Метрика':<10} {'Δ от baseline'}")
print("-" * 60)
print(f"{'Baseline':<40} {baseline_metric:.6f}   {'—'}")

# Здесь нужно будет вручную добавить результаты после запуска в ноутбуке
print("\nДля тестирования скопируйте этот код в ноутбук после ячейки с valid_advanced")
