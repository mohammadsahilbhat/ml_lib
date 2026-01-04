# ml_lib

A machine learning library built from scratch using NumPy.

## Installation

```bash
git clone https://github.com/mohammadsahilbhat/ml_lib.git
cd ml_lib
pip install .

```  




## Development Mode
```bash
pip install -e .
```
### for requirements go to requrirement.txt\

# Usage
```bash
from ml_lib.linear_models import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
preds = model.predict(X_test)
```