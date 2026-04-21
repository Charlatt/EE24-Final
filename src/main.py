import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from scipy.stats import poisson
from simulation import simulate
from currentSeason import simulate_remaining
from stats import unpack
from stats import log_likelihood
from currentSeason import url_26
from currentSeason import plot_simulation_results


# Fix - let kagglehub find it automatically
import kagglehub
path = kagglehub.dataset_download("excel4soccer/espn-soccer-data")
BASE = os.path.join(path, "base_data")

# BASE = os.environ.get("KAGGLE_DATA_PATH", "/Users/loker/.cache/kagglehub/datasets/excel4soccer/espn-soccer-data/versions/526/base_data")

fixtures = pd.read_csv(f"{BASE}/fixtures.csv")
teams = pd.read_csv(f"{BASE}/teams.csv")
leagues = pd.read_csv(f"{BASE}/leagues.csv")
status = pd.read_csv(f"{BASE}/status.csv")

# Find the correct seasonType for 2024-25 EPL
pl_rows = leagues[leagues["leagueId"] == 700].copy()

print("Premier League rows:")
print(pl_rows[["year", "seasonName", "seasonSlug", "seasonType", "leagueId"]])

SEASON_TYPE = 12654
SEASON_TYPE_26 = 13481 # 13481 2025-26 Premier League Season

# Convert dates
fixtures["date"] = pd.to_datetime(fixtures["date"])

# Filter to 2024-25 Premier League
pl = fixtures[
    (fixtures["leagueId"] == 700) &
    (fixtures["seasonType"] == SEASON_TYPE)
].copy()

# Keep only completed matches (28)
pl = pl[pl["statusId"] == 28].copy()

# Merge team names
teams_small = teams[["teamId", "displayName"]].copy()

pl = pl.merge(
    teams_small,
    left_on = "homeTeamId",
    right_on = "teamId",
    how = "left"
).rename(columns  ={"displayName": "home_team"})

pl = pl.merge(
    teams_small,
    left_on = "awayTeamId",
    right_on = "teamId",
    how = "left",
    suffixes = ("", "_away")
).rename(columns = {"displayName": "away_team"})

# Rename scores and keep only needed columns
pl = pl.rename(columns = {
    "homeTeamScore": "home_goals",
    "awayTeamScore": "away_goals"
})

pl = pl[[
    "date",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals"
]].copy()

## *** 2025-2026 MLE

# Name map from standings to team_index_26 names
# Print teams_26 to verify these match exactly
name_map_sim = {
    "Arsenal":                "Arsenal",
    "Manchester City":        "Manchester City",
    "Manchester United":      "Manchester United",
    "Aston Villa":            "Aston Villa",
    "Liverpool":              "Liverpool",
    "Chelsea":                "Chelsea",
    "Brentford":              "Brentford",
    "Bournemouth":            "Bournemouth",
    "Brighton":               "Brighton",
    "Everton":                "Everton",
    "Sunderland":             "Sunderland",
    "Fulham":                 "Fulham",
    "Crystal Palace":         "Crystal Palace",
    "Newcastle United":       "Newcastle United",
    "Leeds":           "Leeds",
    "Nottingham Forest":      "Nottingham Forest",
    "West Ham":               "West Ham",
    "Tottenham":              "Tottenham",
    "Burnley":                "Burnley",
    "Wolverhampton Wanderers":"Wolverhampton Wanderers",
}

raw_26_all = pd.read_csv(url_26)

# Rows without scores = future fixtures
# Remaining fixtures from matchday 34 onwards (after April 20, 2026)


title_count, top4_count, relegated_count = simulate_remaining(n=10000)



plot_simulation_results(title_count, top4_count, relegated_count, n=10000)

# Converts teams to numeric indices
teams = sorted(set(pl["home_team"]).union(pl["away_team"]))
team_index = {team: i for i, team in enumerate(teams)}
num_teams = len(teams)

# Main Code


# Converts indices of teams + goals scored into array
home_index = np.array([team_index[t] for t in pl["home_team"]])
away_index = np.array([team_index[t] for t in pl["away_team"]])
home_goals = np.array(pl["home_goals"]).astype(int)
away_goals = np.array(pl["away_goals"]).astype(int)

# Initial guess of all zeroes
theta0 = np.zeros(2 * num_teams + 1)

# Determines reuslt based on arguments, optimization method, and improvmenet minimum
result = minimize(
    log_likelihood,
    theta0,
    args = (home_index, away_index, home_goals, away_goals, num_teams),
    method = "L-BFGS-B",
    options = {"maxiter": 5000, "ftol": 1e-12}
)

theta_hat = result.x # Best theta value
a_hat, d_hat, h_hat = unpack(theta_hat, num_teams) 

# Returns lambda for home and away team with mle estimation
# Lambda is the expecgted number of goals each team scores
def find_mle(home_team, away_team):
    home = team_index[home_team]
    away = team_index[away_team]

    lambda_home = np.exp(a_hat[home] - d_hat[away] + h_hat)
    lambda_away = np.exp(a_hat[away] - d_hat[home])

    return lambda_home, lambda_away

# Find win probabilites and expected goals scored on 24/25 season
# result = simulate("Arsenal", "Chelsea", 10000, find_mle)
# for key, value in result.items():
#     print(f"{key}: {value:.3f}")




