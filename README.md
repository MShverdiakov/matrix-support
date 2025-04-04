# Matrix Game Analyzer

Анализатор матричных игр с графическим интерфейсом. Позволяет анализировать матричные игры, находить максимин и минимакс стратегии, а также удалять доминируемые стратегии.

## Функциональность

- Загрузка и сохранение матриц в форматах CSV и JSON
- Нахождение максимин и минимакс стратегий
- Удаление строго доминируемых стратегий
- Удаление слабо доминируемых стратегий
- Удаление нло стратегий
- Визуальное выделение стратегий в матрице
- Подробный вывод результатов анализа

## Требования

- Python 3.8+
- PyQt6
- NumPy

## Установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/MShverdiakov/matrix_game_analyzer.git
```

2. Установите зависимости:
```bash
pip install -r requirements.txt
```

## Использование

Запустите программу:
```bash
python -m src.main
```

## Структура проекта

```
matrix_game_analyzer/
├── src/
│   ├── core/
│   │   └── game_analyzer.py
│   │   └── game_analyzer.py
│   ├── gui/
│   │   └── main_window.py
│   └── main.py
├── data/
│   └── test_matrix.csv
├── requirements.txt
└── README.md
```

## License

MIT License 
