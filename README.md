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

## Usage

After `pip install -e ".[dev]"` the `winprob` command is available (or use `python -m winprob`).

```bash
# win probability from a game state; Tier 2 (buy type + loss streak) is the default
winprob predict --score 5-7                      # pistol round next, so no buy types needed
winprob predict --score 6-5 --buy-a "Full buy" --buy-b Eco --streak-b 1 --compare
winprob predict --tier 1 --score 5-7             # score and side only; ignores the economy
winprob predict --tier 3 --score 5-5 --bank-a 19000 --bank-b 19000     --agents-a chamber,sova,neon,jett,omen --ready-a chamber --map haven

# round-win rate team A needs to reach a target win probability (bisection or secant)
winprob solve --score 3-8 --target 0.5 --method secant

# Monte Carlo error vs N: prints the log-log slope, saves a plot
winprob converge --out convergence.png

# fit the round-win logistic model on the pro data in data/raw/
winprob fit

# one match: win probability after each round, plot, and a round-by-round step-through
winprob match vlr-100001-84408 --out match.png --step
```

Impossible states (a score past the win condition, negative or over-cap credits,
more than 5 ults, ...) are rejected with an explanation and exit code 2.

## Tests

```bash
pytest
```
