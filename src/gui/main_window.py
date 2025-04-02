from PyQt6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QTableWidget, QTableWidgetItem,
                             QLabel, QSpinBox, QFileDialog, QMessageBox,
                             QScrollArea, QTextEdit)
from PyQt6.QtCore import Qt
import numpy as np
from src.core.game_analyzer import GameAnalyzer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Matrix Game Analyzer")
        self.setMinimumSize(600, 500)
        
        # Initialize game analyzer
        self.game_analyzer = GameAnalyzer(np.zeros((2, 2)))  # Initialize with empty matrix
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(5)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Matrix size controls
        size_layout = QHBoxLayout()
        size_layout.setSpacing(5)
        
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 10)
        self.rows_spin.setValue(2)
        self.rows_spin.setFixedWidth(60)
        self.rows_spin.valueChanged.connect(self.update_matrix_size)
        
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 10)
        self.cols_spin.setValue(2)
        self.cols_spin.setFixedWidth(60)
        self.cols_spin.valueChanged.connect(self.update_matrix_size)
        
        size_layout.addStretch()
        size_layout.addWidget(QLabel("Rows:"))
        size_layout.addWidget(self.rows_spin)
        size_layout.addWidget(QLabel("Columns:"))
        size_layout.addWidget(self.cols_spin)
        size_layout.addStretch()
        layout.addLayout(size_layout)
        
        # Matrix table
        self.matrix_table = QTableWidget()
        self.matrix_table.setRowCount(2)
        self.matrix_table.setColumnCount(2)
        self.matrix_table.setFixedHeight(200)
        layout.addWidget(self.matrix_table)
        
        # Buttons layout (horizontal)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(5)
        
        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(80)
        load_btn.clicked.connect(self.load_matrix)
        
        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(80)
        save_btn.clicked.connect(self.save_matrix)
        
        analyze_btn = QPushButton("Analyze")
        analyze_btn.setFixedWidth(80)
        analyze_btn.clicked.connect(self.analyze_game)
        
        random_btn = QPushButton("Random")
        random_btn.setFixedWidth(80)
        random_btn.clicked.connect(self.generate_random_matrix)
        
        buttons_layout.addStretch()
        buttons_layout.addWidget(load_btn)
        buttons_layout.addWidget(save_btn)
        buttons_layout.addWidget(analyze_btn)
        buttons_layout.addWidget(random_btn)
        buttons_layout.addStretch()
        layout.addLayout(buttons_layout)
        
        # Results display - now using QTextEdit with scrolling
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("QTextEdit { font-size: 10pt; }")
        self.results_text.setMinimumHeight(150)
        layout.addWidget(self.results_text)
        
        # Initialize matrix size
        self.update_matrix_size()
    
    def update_matrix_size(self):
        """Update the matrix table size based on spin box values."""
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        self.matrix_table.setRowCount(rows)
        self.matrix_table.setColumnCount(cols)
        
        # Set default values
        for i in range(rows):
            for j in range(cols):
                if not self.matrix_table.item(i, j):
                    self.matrix_table.setItem(i, j, QTableWidgetItem("0"))
    
    def get_matrix_from_table(self) -> np.ndarray:
        """Extract matrix values from the table."""
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        matrix = np.zeros((rows, cols))
        
        for i in range(rows):
            for j in range(cols):
                item = self.matrix_table.item(i, j)
                matrix[i, j] = float(item.text() if item else 0)
        
        return matrix
    
    def set_matrix_to_table(self, matrix: np.ndarray):
        """Set matrix values in the table."""
        rows, cols = matrix.shape
        self.rows_spin.setValue(rows)
        self.cols_spin.setValue(cols)
        self.update_matrix_size()
        
        for i in range(rows):
            for j in range(cols):
                self.matrix_table.setItem(i, j, QTableWidgetItem(str(matrix[i, j])))
    
    def load_matrix(self):
        """Load matrix from file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Matrix", "", "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if filename:
            try:
                format = 'csv' if filename.endswith('.csv') else 'json'
                self.game_analyzer = GameAnalyzer.load_from_file(filename, format)
                self.set_matrix_to_table(self.game_analyzer.matrix)
                QMessageBox.information(self, "Success", "Matrix loaded successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load matrix: {str(e)}")
    
    def save_matrix(self):
        """Save matrix to file."""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Matrix", "", "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if filename:
            try:
                matrix = self.get_matrix_from_table()
                self.game_analyzer = GameAnalyzer(matrix)
                format = 'csv' if filename.endswith('.csv') else 'json'
                self.game_analyzer.save_to_file(filename, format)
                QMessageBox.information(self, "Success", "Matrix saved successfully!")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save matrix: {str(e)}")
    
    def analyze_game(self):
        """Perform game analysis and display results."""
        try:
            matrix = self.get_matrix_from_table()
            self.game_analyzer = GameAnalyzer(matrix)
            
            # Perform analysis
            maximin_value, maximin_strategy = self.game_analyzer.find_maximin()
            minimax_value, minimax_strategy = self.game_analyzer.find_minimax()
            strictly_dominated_rows, weakly_dominated_rows = self.game_analyzer.find_dominated_rows()
            strictly_dominated_cols, weakly_dominated_cols = self.game_analyzer.find_dominated_columns()
            
            # Format results with 1-indexed positions
            results = f"""Analysis Results:

Maximin Strategy:
- Value: {maximin_value:.2f}
- Strategy Index: {maximin_strategy + 1}

Minimax Strategy:
- Value: {minimax_value:.2f}
- Strategy Index: {minimax_strategy + 1}

Dominated Strategies:
- Strictly Dominated Rows: {[x + 1 for x in strictly_dominated_rows]}
- Strictly Dominated Columns: {[x + 1 for x in strictly_dominated_cols]}
- Weakly Dominated Rows: {[x + 1 for x in weakly_dominated_rows]}
- Weakly Dominated Columns: {[x + 1 for x in weakly_dominated_cols]}
"""
            
            self.results_text.setPlainText(results)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
    
    def generate_random_matrix(self):
        """Generate a random matrix."""
        try:
            rows = self.rows_spin.value()
            cols = self.cols_spin.value()
            # Generate random numbers and round to 2 decimal places
            matrix = np.round(np.random.uniform(-10, 10, size=(rows, cols)), 2)
            self.set_matrix_to_table(matrix)
            QMessageBox.information(self, "Success", "Random matrix generated successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate random matrix: {str(e)}") 