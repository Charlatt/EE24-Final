import matplotlib.pyplot as plt
from main import find_mle, simulate
import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from scipy.stats import poisson
from simulation import results, simulate_one_team, expected_points_one_team


BASE = os.environ.get("KAGGLE_DATA_PATH", "/Users/bradyk/.cache/kagglehub/datasets/excel4soccer/espn-soccer-data/versions/527/base_data")

fixtures = pd.read_csv(f"{BASE}/fixtures.csv")
teams = pd.read_csv(f"{BASE}/teams.csv")
leagues = pd.read_csv(f"{BASE}/leagues.csv")
status = pd.read_csv(f"{BASE}/status.csv")

# Print 
arsenal = "Arsenal"
home_opps = []
home_win_probs = []
away_opps = []
away_win_probs = []

for opponent in teams:
    if opponent == arsenal:
        continue

    arsenal_home_win, opp_win, draw_prob = simulate(arsenal,opponent,find_mle)
    home_opps.append(opponent)
    home_win_probs.append(arsenal_home_win)

    opp_away_win, arsenal_away_win, draw_prob = simulate(opponent,arsenal,find_mle)
    away_opps.append(opponent)
    away_win_probs.append(arsenal_away_win)


# Graph 1: Arsenal home win probabilities
plt.figure(figsize=(12, 6))
plt.bar(home_opps, home_win_probs)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Win Probability")
plt.xlabel("Opponent")
plt.title("Arsenal Home Win Probability vs Each Opponent")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()

# Graph 2: Arsenal away win probabilities
plt.figure(figsize=(12, 6))
plt.bar(away_opps, away_win_probs)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Win Probability")
plt.xlabel("Opponent")
plt.title("Arsenal Away Win Probability vs Each Opponent")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
