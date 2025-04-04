from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QLabel, QSpinBox, QFileDialog, QMessageBox,
                             QScrollArea, QTextEdit)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
import numpy as np
from src.core.game_analyzer import GameAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализатор матричных игр")
        self.setMinimumSize(800, 600)
        
        # Создаем центральный виджет и главный layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Создаем верхнюю панель с размерами и кнопками загрузки/сохранения
        top_panel = QHBoxLayout()
        
        # Размеры матрицы
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Строк:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 10)
        self.rows_spin.setValue(2)
        self.rows_spin.setFixedWidth(60)
        self.rows_spin.valueChanged.connect(self.update_matrix_size)
        size_layout.addWidget(self.rows_spin)
        
        size_layout.addWidget(QLabel("Столбцов:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 10)
        self.cols_spin.setValue(2)
        self.cols_spin.setFixedWidth(60)
        self.cols_spin.valueChanged.connect(self.update_matrix_size)
        size_layout.addWidget(self.cols_spin)
        
        top_panel.addLayout(size_layout)
        top_panel.addStretch()
        
        # Кнопки загрузки и сохранения
        load_button = QPushButton("Загрузить")
        load_button.clicked.connect(self.load_matrix)
        top_panel.addWidget(load_button)
        
        save_button = QPushButton("Сохранить")
        save_button.clicked.connect(self.save_matrix)
        top_panel.addWidget(save_button)
        
        main_layout.addLayout(top_panel)
        
        # Создаем таблицу для матрицы
        self.matrix_table = QTableWidget()
        self.matrix_table.setMinimumSize(400, 300)
        main_layout.addWidget(self.matrix_table)
        
        # Создаем панель с кнопками анализа
        analysis_buttons = QHBoxLayout()
        
        maximin_button = QPushButton("Максимин")
        maximin_button.clicked.connect(self.find_maximin)
        analysis_buttons.addWidget(maximin_button)
        
        minimax_button = QPushButton("Минимакс")
        minimax_button.clicked.connect(self.find_minimax)
        analysis_buttons.addWidget(minimax_button)
        
        remove_strict_button = QPushButton("Удалить строго доминируемые")
        remove_strict_button.clicked.connect(self.remove_strictly_dominated)
        analysis_buttons.addWidget(remove_strict_button)
        
        remove_weak_button = QPushButton("Удалить слабо доминируемые")
        remove_weak_button.clicked.connect(self.remove_weakly_dominated)
        analysis_buttons.addWidget(remove_weak_button)
        
        remove_unmentioned_button = QPushButton("Удалить нло стратегии")
        remove_unmentioned_button.clicked.connect(self.remove_unmentioned_strategies)
        analysis_buttons.addWidget(remove_unmentioned_button)
        
        main_layout.addLayout(analysis_buttons)
        
        # Создаем область для отображения результатов с прокруткой
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(200)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        scroll_area.setWidget(self.results_text)
        
        main_layout.addWidget(scroll_area)
        
        # Инициализируем анализатор
        self.analyzer = GameAnalyzer(np.array([[0]]))
        
        # Устанавливаем начальное состояние
        self.update_matrix_size()
        
    def update_matrix_size(self):
        """Обновляет размер таблицы на основе значений спинбоксов."""
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        self.matrix_table.setRowCount(rows)
        self.matrix_table.setColumnCount(cols)
        
        # Устанавливаем заголовки
        self.matrix_table.setHorizontalHeaderLabels([f"Стратегия {i+1}" for i in range(cols)])
        self.matrix_table.setVerticalHeaderLabels([f"Стратегия {i+1}" for i in range(rows)])
        
        # Заполняем таблицу нулями
        for i in range(rows):
            for j in range(cols):
                if not self.matrix_table.item(i, j):
                    item = QTableWidgetItem("0")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.matrix_table.setItem(i, j, item)
    
    def get_matrix_from_table(self) -> np.ndarray:
        """Извлекает значения матрицы из таблицы."""
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        matrix = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                item = self.matrix_table.item(i, j)
                matrix[i, j] = float(item.text() if item else 0)
        
        return matrix
    
    def set_matrix_to_table(self, matrix: np.ndarray):
        """Устанавливает значения матрицы в таблицу."""
        rows, cols = matrix.shape
        self.rows_spin.setValue(rows)
        self.cols_spin.setValue(cols)
        self.update_matrix_size()
        
        for i in range(rows):
            for j in range(cols):
                item = QTableWidgetItem(str(matrix[i, j]))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.matrix_table.setItem(i, j, item)
    
    def highlight_cells(self, rows=None, cols=None, color=QColor(255, 255, 0, 100)):
        """Выделяет ячейки указанным цветом."""
        # Сначала сбрасываем все выделения
        for i in range(self.matrix_table.rowCount()):
            for j in range(self.matrix_table.columnCount()):
                self.matrix_table.item(i, j).setBackground(QColor(255, 255, 255))
        
        # Выделяем указанные строки и столбцы
        if rows is not None:
            for i in rows:
                for j in range(self.matrix_table.columnCount()):
                    self.matrix_table.item(i, j).setBackground(color)
        
        if cols is not None:
            for j in cols:
                for i in range(self.matrix_table.rowCount()):
                    self.matrix_table.item(i, j).setBackground(color)
    
    def find_maximin(self):
        """Находит максимин стратегии и выделяет их."""
        matrix = self.get_matrix_from_table()
        self.analyzer = GameAnalyzer(matrix)
        
        # Находим максимин
        maximin_value, maximin_strategy = self.analyzer.find_maximin()
        
        # Находим все строки с максимальным значением
        row_minima = np.min(matrix, axis=1)
        maximin_rows = np.where(row_minima == maximin_value)[0]
        
        # Выделяем строки
        self.highlight_cells(rows=maximin_rows)
        
        # Форматируем результаты
        results = "Анализ максимин стратегий:\n\n"
        results += f"Значение максимина: {maximin_value}\n"
        results += f"Максимин стратегии: {[i+1 for i in maximin_rows]}\n"
        results += f"Значения в строках: {[row_minima[i] for i in maximin_rows]}\n"
        
        self.results_text.setPlainText(results)
    
    def find_minimax(self):
        """Находит минимакс стратегии и выделяет их."""
        matrix = self.get_matrix_from_table()
        self.analyzer = GameAnalyzer(matrix)
        
        # Находим минимакс
        minimax_value, minimax_strategy = self.analyzer.find_minimax()
        
        # Находим все столбцы с минимальным значением
        column_maxima = np.max(matrix, axis=0)
        minimax_cols = np.where(column_maxima == minimax_value)[0]
        
        # Выделяем столбцы
        self.highlight_cells(cols=minimax_cols)
        
        # Форматируем результаты
        results = "Анализ минимакс стратегий:\n\n"
        results += f"Значение минимакса: {minimax_value}\n"
        results += f"Минимакс стратегии: {[i+1 for i in minimax_cols]}\n"
        results += f"Значения в столбцах: {[column_maxima[i] for i in minimax_cols]}\n"
        
        self.results_text.setPlainText(results)
    
    def remove_strictly_dominated(self):
        """Удаляет строго доминируемые стратегии."""
        matrix = self.get_matrix_from_table()
        self.analyzer = GameAnalyzer(matrix)
        
        # Находим доминируемые стратегии
        strictly_dominated_rows, _ = self.analyzer.find_dominated_rows(matrix)
        strictly_dominated_cols, _ = self.analyzer.find_dominated_columns(matrix)
        
        if not strictly_dominated_rows and not strictly_dominated_cols:
            self.results_text.setPlainText("Нет строго доминируемых стратегий для удаления.")
            return
        
        # Удаляем доминируемые стратегии
        matrix = self.analyzer.rationalize()
        
        # Обновляем таблицу
        self.set_matrix_to_table(matrix)
        
        # Форматируем результаты
        results = "Удаление строго доминируемых стратегий:\n\n"
        if strictly_dominated_rows:
            results += f"Удалены строго доминируемые строки: {[i+1 for i in strictly_dominated_rows]}\n"
        if strictly_dominated_cols:
            results += f"Удалены строго доминируемые столбцы: {[i+1 for i in strictly_dominated_cols]}\n"
        
        self.results_text.setPlainText(results)
    
    def remove_weakly_dominated(self):
        """Удаляет слабо и строго доминируемые стратегии."""
        matrix = self.get_matrix_from_table()
        self.analyzer = GameAnalyzer(matrix)
        
        # Находим все доминируемые стратегии (и строго, и слабо)
        strictly_dominated_rows, weakly_dominated_rows = self.analyzer.find_dominated_rows(matrix)
        strictly_dominated_cols, weakly_dominated_cols = self.analyzer.find_dominated_columns(matrix)
        
        # Объединяем списки доминируемых стратегий
        all_dominated_rows = list(set(strictly_dominated_rows + weakly_dominated_rows))
        all_dominated_cols = list(set(strictly_dominated_cols + weakly_dominated_cols))
        
        if not all_dominated_rows and not all_dominated_cols:
            self.results_text.setPlainText("Нет доминируемых стратегий для удаления.")
            return
        
        # Создаем маски для удаления
        row_mask = np.ones(len(matrix), dtype=bool)
        for i in all_dominated_rows:
            row_mask[i] = False
            
        col_mask = np.ones(len(matrix[0]), dtype=bool)
        for i in all_dominated_cols:
            col_mask[i] = False
        
        # Удаляем доминируемые стратегии
        new_matrix = matrix[row_mask][:, col_mask]
        
        # Обновляем таблицу
        self.set_matrix_to_table(new_matrix)
        
        # Форматируем результаты
        results = "Удаление доминируемых стратегий:\n\n"
        if strictly_dominated_rows:
            results += f"Удалены строго доминируемые строки: {[i+1 for i in strictly_dominated_rows]}\n"
        if strictly_dominated_cols:
            results += f"Удалены строго доминируемые столбцы: {[i+1 for i in strictly_dominated_cols]}\n"
        if weakly_dominated_rows:
            results += f"Удалены слабо доминируемые строки: {[i+1 for i in weakly_dominated_rows]}\n"
        if weakly_dominated_cols:
            results += f"Удалены слабо доминируемые столбцы: {[i+1 for i in weakly_dominated_cols]}\n"
        
        self.results_text.setPlainText(results)
    
    def remove_unmentioned_strategies(self):
        """Удаляет нло стратегии."""
        matrix = self.get_matrix_from_table()
        self.analyzer = GameAnalyzer(matrix)
        
        # Удаляем нло стратегии
        matrix = self.analyzer.remove_unmentioned_row(matrix)
        matrix = self.analyzer.remove_unmentioned_columns(matrix)
        
        # Преобразуем список в numpy array
        matrix = np.array(matrix)
        
        # Обновляем таблицу
        self.set_matrix_to_table(matrix)
        
        # Форматируем результаты
        results = "Удаление нло стратегий:\n\n"
        results += "Удалены стратегии, которые не содержат максимумов в соответствующих строках/столбцах.\n"
        
        self.results_text.setPlainText(results)
    
    def load_matrix(self):
        """Загружает матрицу из файла."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Загрузить матрицу", "", "CSV файлы (*.csv);;JSON файлы (*.json)"
        )
        if filename:
            try:
                format = 'csv' if filename.endswith('.csv') else 'json'
                self.analyzer = GameAnalyzer.load_from_file(filename, format)
                self.set_matrix_to_table(self.analyzer.matrix)
                QMessageBox.information(self, "Успех", "Матрица успешно загружена!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось загрузить матрицу: {str(e)}")
    
    def save_matrix(self):
        """Сохраняет матрицу в файл."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Сохранить матрицу", "", "CSV файлы (*.csv);;JSON файлы (*.json)"
        )
        if filename:
            try:
                matrix = self.get_matrix_from_table()
                self.analyzer = GameAnalyzer(matrix)
                format = 'csv' if filename.endswith('.csv') else 'json'
                self.analyzer.save_to_file(filename, format)
                QMessageBox.information(self, "Успех", "Матрица успешно сохранена!")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка", f"Не удалось сохранить матрицу: {str(e)}") 