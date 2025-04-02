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

    def find_dominated_rows(self) -> Tuple[List[int], List[int]]:
        """
        Find all dominated row strategies.
        
        Returns:
            Tuple of (strictly dominated rows, weakly dominated rows)
        """
        strictly_dominated = set()
        weakly_dominated = set()
        
        # For each row, check if it is dominated by any other row
        for i in range(self.rows):
            for j in range(self.rows):
                if i != j:
                    # Check if row i is strictly dominated by row j
                    if all(self.matrix[i, k] < self.matrix[j, k] for k in range(self.cols)):
                        strictly_dominated.add(i)
                    
                    # Check if row i is weakly dominated by row j
                    if (all(self.matrix[i, k] <= self.matrix[j, k] for k in range(self.cols)) and
                        any(self.matrix[i, k] < self.matrix[j, k] for k in range(self.cols))):
                        weakly_dominated.add(i)
        
        return list(strictly_dominated), list(weakly_dominated)
        
    def find_dominated_columns(self) -> Tuple[List[int], List[int]]:
        """
        Find all dominated column strategies.
        
        Returns:
            Tuple of (strictly dominated columns, weakly dominated columns)
        """
        strictly_dominated = set()
        weakly_dominated = set()
        
        # For each column, check if it is dominated by any other column
        for i in range(self.cols):
            for j in range(self.cols):
                if i != j:
                    # Check if column i is strictly dominated by column j
                    if all(self.matrix[k, i] > self.matrix[k, j] for k in range(self.rows)):
                        strictly_dominated.add(i)
                    
                    # Check if column i is weakly dominated by column j
                    if (all(self.matrix[k, i] >= self.matrix[k, j] for k in range(self.rows)) and
                        any(self.matrix[k, i] > self.matrix[k, j] for k in range(self.rows))):
                        weakly_dominated.add(i)
        
        return list(strictly_dominated), list(weakly_dominated)
    
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
        strictly_dominated_rows, _ = self.find_dominated_rows()
        strictly_dominated_cols, _ = self.find_dominated_columns()
        
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