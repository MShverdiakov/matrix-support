# Matrix Game Analyzer

A Python application for analyzing matrix games, including finding maximin and minimax strategies, dominated strategies, and rationalization.

## Features

- Load and save game matrices in CSV and JSON formats
- Find maximin and minimax strategies
- Identify strictly and weakly dominated strategies
- Rationalize game matrix by removing dominated strategies
- Generate random game matrices for testing
- GUI interface for easy interaction

## Requirements

- Python 3.8+
- NumPy
- PyQt6

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/matrix_game_analizer.git
cd matrix_game_analizer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
python -m src.main
```

### File Formats

#### CSV Format
```
1,2,3
2,4,1
1,3,2
```

#### JSON Format
```json
{
    "matrix": [[1,2,3], [2,4,1], [1,3,2]],
    "shape": [3,3]
}
```

## License

MIT License 