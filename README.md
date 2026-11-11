# ECEN4293-Final-Project
Valorant Map Win-Probability Engine

Final project for ECEN 4293, Numerical Methods for Python, Oklahoma State University.


## Setup

```bash
python -m venv .venv
.venv/Scripts/activate   # .venv/bin/activate on macOS/Linux
pip install -e ".[dev]"
```

Raw data (`data/raw/*.csv`) is not committed; place the chiro.gg CSV exports there yourself.

## Tests

```bash
pytest
```
