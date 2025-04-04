import numpy as np
from typing import Tuple, List, Optional

class GameAnalyzer:
    def __init__(self, matrix: np.ndarray):
        """
        Initialize the game analyzer with a payoff matrix.
        
        Args:
            matrix: A 2D numpy array representing the payoff matrix
        """
        self.matrix = matrix
        self.rows, self.cols = matrix.shape

    def find_maximin(self) -> Tuple[float, int]:
        """
        Find the maximin strategy for the row player.
        
        Returns:
            Tuple of (maximin value, strategy index)
        """
        row_minima = np.min(self.matrix, axis=1)
        maximin_value = np.max(row_minima)
        maximin_strategy = np.argmax(row_minima)
        return maximin_value, maximin_strategy

    def find_minimax(self) -> Tuple[float, int]:
        """
        Find the minimax strategy for the column player.
        
        Returns:
            Tuple of (minimax value, strategy index)
        """
        column_maxima = np.max(self.matrix, axis=0)
        minimax_value = np.min(column_maxima)
        minimax_strategy = np.argmin(column_maxima)
        return minimax_value, minimax_strategy

    def is_dominated_row(self, matrix, row1, row2):
        """Проверяет, доминирует ли строка row1 строку row2"""
        # Для строк (первый игрок) - большее значение лучше
        # Строгая доминация: все элементы в row1 должны быть больше, чем в row2
        # Слабая доминация: все элементы в row1 должны быть больше или равны, чем в row2, и хотя бы один строго больше
        strictly_dominated = all(matrix[row1][j] > matrix[row2][j] for j in range(len(matrix[0])))
        weakly_dominated = (all(matrix[row1][j] >= matrix[row2][j] for j in range(len(matrix[0]))) and
                           any(matrix[row1][j] > matrix[row2][j] for j in range(len(matrix[0]))))
        return strictly_dominated, weakly_dominated

    def is_dominated_column(self, matrix, col1, col2):
        """Проверяет, доминирует ли столбец col1 столбец col2"""
        # Для столбцов (второй игрок) - меньшее значение лучше
        # Строгая доминация: все элементы в col1 должны быть меньше, чем в col2
        # Слабая доминация: все элементы в col1 должны быть меньше или равны, чем в col2, и хотя бы один строго меньше
        strictly_dominated = all(matrix[i][col1] > matrix[i][col2] for i in range(len(matrix)))
        weakly_dominated = (all(matrix[i][col1] >= matrix[i][col2] for i in range(len(matrix))) and
                           any(matrix[i][col1] > matrix[i][col2] for i in range(len(matrix))))
        return strictly_dominated, weakly_dominated

    def find_dominated_rows(self, matrix):
        """Находит доминируемые строки"""
        rows = len(matrix)
        strictly_dominated = []
        weakly_dominated = []
        
        for i in range(rows):
            for j in range(rows):
                if i != j:
                    strict, weak = self.is_dominated_row(matrix, j, i)
                    if strict:
                        strictly_dominated.append(i)
                    elif weak:
                        weakly_dominated.append(i)
        
        return list(set(strictly_dominated)), list(set(weakly_dominated))

    def find_dominated_columns(self, matrix):
        """Находит доминируемые столбцы"""
        cols = len(matrix[0])
        strictly_dominated = []
        weakly_dominated = []
        
        for i in range(cols):
            for j in range(cols):
                if i != j:
                    strict, weak = self.is_dominated_column(matrix, j, i)
                    if strict:
                        strictly_dominated.append(i)
                    elif weak:
                        weakly_dominated.append(i)
        
        return list(set(strictly_dominated)), list(set(weakly_dominated))

    def remove_unmentioned_row(self, matrix):
        """Удаляет строки, которые не содержат максимумов в столбцах"""
        rows = len(matrix)
        cols = len(matrix[0])

        # Шаг 1: Найти максимумы в каждом столбце и сохранить номера строк
        mentioned_rows = set()
        for col in range(cols):
            max_value = max(row[col] for row in matrix)  # Находим максимум в столбце
            for row_index in range(rows):
                if matrix[row_index][col] == max_value:
                    mentioned_rows.add(row_index)  # Запоминаем строку, содержащую максимум

        # Шаг 2: Найти все строки, которые не упоминались в максимумах
        unmentioned_rows = [row_index for row_index in range(rows) if row_index not in mentioned_rows]

        if not unmentioned_rows:
            return matrix  # Если все строки были упомянуты, ничего не удаляем

        # Удаляем все строки, которые не упоминались
        new_matrix = [row for i, row in enumerate(matrix) if i not in unmentioned_rows]
        return new_matrix

    def remove_unmentioned_columns(self, matrix):
        """Удаляет столбцы, которые не содержат максимумов в строках"""
        rows = len(matrix)
        cols = len(matrix[0])

        # Шаг 1: Найти максимумы в каждой строке и сохранить номера столбцов
        mentioned_cols = set()
        for row in matrix:
            max_value = max(row)  # Находим максимум в строке
            for col_index in range(cols):
                if row[col_index] == max_value:
                    mentioned_cols.add(col_index)  # Запоминаем столбец, содержащий максимум

        # Шаг 2: Найти все столбцы, которые не упоминались в максимумах
        unmentioned_cols = [col_index for col_index in range(cols) if col_index not in mentioned_cols]

        if not unmentioned_cols:
            return matrix  # Если все столбцы были упомянуты, ничего не удаляем

        # Удаляем все столбцы, которые не упоминались
        new_matrix = [[row[col] for col in range(cols) if col not in unmentioned_cols] for row in matrix]
        return new_matrix

    def find_best_response(self, opponent_strategy: np.ndarray) -> Tuple[int, float]:
        """
        Find the best response to a mixed strategy of the opponent.
        
        Args:
            opponent_strategy: Array of probabilities for each column strategy
            
        Returns:
            Tuple of (best response strategy index, expected payoff)
        """
        expected_payoffs = np.dot(self.matrix, opponent_strategy)
        best_strategy = np.argmax(expected_payoffs)
        best_payoff = expected_payoffs[best_strategy]
        return best_strategy, best_payoff

    def rationalize(self) -> np.ndarray:
        """
        Remove dominated strategies from the game matrix.
        
        Returns:
            New matrix with dominated strategies removed
        """
        strictly_dominated_rows, _ = self.find_dominated_rows(self.matrix)
        strictly_dominated_cols, _ = self.find_dominated_columns(self.matrix)
        
        if not strictly_dominated_rows and not strictly_dominated_cols:
            return self.matrix
        
        # Create masks for non-dominated rows and columns
        row_mask = np.ones(self.rows, dtype=bool)
        for i in strictly_dominated_rows:
            row_mask[i] = False
            
        col_mask = np.ones(self.cols, dtype=bool)
        for i in strictly_dominated_cols:
            col_mask[i] = False
            
        # Return matrix with dominated strategies removed
        return self.matrix[row_mask][:, col_mask]

    def generate_random_matrix(self, size: Tuple[int, int], min_val: float = -10, max_val: float = 10) -> np.ndarray:
        """
        Generate a random game matrix for testing.
        
        Args:
            size: Tuple of (rows, columns)
            min_val: Minimum value in the matrix
            max_val: Maximum value in the matrix
            
        Returns:
            Random game matrix
        """
        return np.random.uniform(min_val, max_val, size=size)

    def save_to_file(self, filename: str, format: str = 'csv') -> None:
        """
        Save the game matrix to a file.
        
        Args:
            filename: Name of the file to save to
            format: File format ('csv' or 'json')
        """
        # Round matrix to 2 decimal places
        rounded_matrix = np.round(self.matrix, decimals=2)
        
        if format == 'csv':
            np.savetxt(filename, rounded_matrix, delimiter=',', fmt='%.2f')
        elif format == 'json':
            import json
            data = {
                'matrix': rounded_matrix.tolist(),
                'shape': rounded_matrix.shape
            }
            with open(filename, 'w') as f:
                json.dump(data, f)

    @classmethod
    def load_from_file(cls, filename: str, format: str = 'csv') -> 'GameAnalyzer':
        """
        Load a game matrix from a file.
        
        Args:
            filename: Name of the file to load from
            format: File format ('csv' or 'json')
            
        Returns:
            GameAnalyzer instance with loaded matrix
        """
        if format == 'csv':
            matrix = np.loadtxt(filename, delimiter=',')
        elif format == 'json':
            import json
            with open(filename, 'r') as f:
                data = json.load(f)
                matrix = np.array(data['matrix'])
        return cls(matrix) 