# Contributing to AI Customer Behavior Analysis

Thank you for your interest in contributing to this project! 🎉

## 📋 How to Contribute

### 1. Fork the Repository
```bash
git fork https://github.com/YOUR_USERNAME/ai-customer-behavior-project.git
```

### 2. Clone Your Fork
```bash
git clone https://github.com/YOUR_USERNAME/ai-customer-behavior-project.git
cd ai-customer-behavior-project
```

### 3. Set Up Development Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Create a Branch
```bash
git checkout -b feature/your-feature-name
```

### 5. Make Changes
- Follow the existing code style
- Add docstrings to new functions
- Update tests if applicable

### 6. Test Your Changes
```bash
# Run notebooks to verify everything works
jupyter notebook
```

### 7. Commit and Push
```bash
git add .
git commit -m "Add: description of your changes"
git push origin feature/your-feature-name
```

### 8. Open a Pull Request
Go to GitHub and create a Pull Request from your branch.

---

## 📁 Project Structure

```
ai-customer-behavior-project/
├── data/           # Data files (raw & processed)
├── notebooks/      # Jupyter notebooks (numbered sequence)
├── src/            # Reusable Python modules
├── outputs/        # Generated outputs (figures, models)
└── requirements.txt
```

---

## 🎨 Code Style Guidelines

- Use **snake_case** for functions and variables
- Use **PascalCase** for classes
- Add **docstrings** to all functions
- Keep functions focused and under 50 lines when possible
- Use **type hints** for function parameters

Example:
```python
def calculate_score(value: float, weight: float = 1.0) -> float:
    """
    Calculate weighted score.
    
    Parameters:
        value: Base value to score
        weight: Score multiplier (default 1.0)
        
    Returns:
        Calculated weighted score
    """
    return value * weight
```

---

## 🐛 Reporting Issues

Please include:
1. Description of the issue
2. Steps to reproduce
3. Expected vs actual behavior
4. Python version and OS

---

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.
