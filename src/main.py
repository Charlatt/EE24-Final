import pandas as pd
import numpy as np
import os
from scipy.optimize import minimize
from scipy.stats import poisson
from simulation import simulate

BASE = os.environ.get("KAGGLE_DATA_PATH", "/Users/WILL/.cache/kagglehub/datasets/excel4soccer/espn-soccer-data/versions/526/base_data")

fixtures = pd.read_csv(f"{BASE}/fixtures.csv")
teams = pd.read_csv(f"{BASE}/teams.csv")
leagues = pd.read_csv(f"{BASE}/leagues.csv")
status = pd.read_csv(f"{BASE}/status.csv")

# Find the correct seasonType for 2024-25 EPL
pl_rows = leagues[leagues["leagueId"] == 700].copy()

SEASON_TYPE = 12654

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

# Converts teams to numeric indices
teams = sorted(set(pl["home_team"]).union(pl["away_team"]))
team_index = {team: i for i, team in enumerate(teams)}
num_teams = len(teams)

# Main Code

# Returns attack strength, defensive strength, and home advantage
def unpack(theta):
    a = theta[:num_teams] # positions 0-19
    d = theta[num_teams:2 * num_teams] # positions 20-39
    h = theta[-1] # position 40
    return a, d, h

# Return log-likelihood
def log_likelihood(theta, home_index, away_index, home_goals, away_goals):
    a, d, h = unpack(theta)
    
    # Compute lambdas for every match at once
    lambda_home = np.exp(a[home_index] - d[away_index] + h)
    lambda_away = np.exp(a[away_index] - d[home_index])
    
    # Log-likelihood over all 380 matches
    ll = (poisson.logpmf(home_goals, lambda_home).sum() +
          poisson.logpmf(away_goals, lambda_away).sum())
    
    return -ll

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
    args = (home_index, away_index, home_goals, away_goals),
    method = "L-BFGS-B",
    options = {"maxiter": 5000, "ftol": 1e-12}
)

theta_hat = result.x # Best theta value
a_hat, d_hat, h_hat = unpack(theta_hat) 

# Returns lambda for home and away team with mle estimation
# Lambda is the expecgted number of goals each team scores
def find_mle(home_team, away_team):
    home = team_index[home_team]
    away = team_index[away_team]

    lambda_home = np.exp(a_hat[home] - d_hat[away] + h_hat)
    lambda_away = np.exp(a_hat[away] - d_hat[home])

    return lambda_home, lambda_away

# Find win probabilites and expected goals scored on 24/25 season
result = simulate("Arsenal", "Chelsea", 10000, find_mle)
for key, value in result.items():
    print(f"{key}: {value:.3f}")




