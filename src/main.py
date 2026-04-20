import pandas as pd
import numpy as np

BASE = "/Users/WILL/.cache/kagglehub/datasets/excel4soccer/espn-soccer-data/versions/526/base_data"

fixtures = pd.read_csv(f"{BASE}/fixtures.csv")
teams = pd.read_csv(f"{BASE}/teams.csv")
leagues = pd.read_csv(f"{BASE}/leagues.csv")
status = pd.read_csv(f"{BASE}/status.csv")

# Find the correct seasonType for 2024-25 EPL
pl_rows = leagues[leagues["leagueId"] == 700].copy()
print("Premier League rows:")
print(pl_rows[["year", "seasonName", "seasonSlug", "seasonType", "leagueId"]])

SEASON_TYPE = 12654

# Convert dates
fixtures["date"] = pd.to_datetime(fixtures["date"])

# Filter to 2024-25 Premier League
pl = fixtures[
    (fixtures["leagueId"] == 700) &
    (fixtures["seasonType"] == SEASON_TYPE)
].copy()

# Keep only completed matches (I think 28?)
pl = pl[pl["statusId"] == 28].copy()

# Merge team names
teams_small = teams[["teamId", "displayName"]].copy()

pl = pl.merge(
    teams_small,
    left_on="homeTeamId",
    right_on="teamId",
    how="left"
).rename(columns={"displayName": "home_team"})

pl = pl.merge(
    teams_small,
    left_on="awayTeamId",
    right_on="teamId",
    how="left",
    suffixes=("", "_away")
).rename(columns={"displayName": "away_team"})

# Rename scores and keep only needed columns
pl = pl.rename(columns={
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

# Basic checks
# print(pl.head())
# print("Matches:", len(pl))
# print("Unique teams:", len(set(pl["home_team"]).union(set(pl["away_team"]))))
# print("Average home goals:", pl["home_goals"].mean())
# print("Average away goals:", pl["away_goals"].mean())


# Main Code
# Create long-format dataset
home = pl[['home_team', 'home_goals', 'away_goals']].copy()
home.columns = ['team', 'goals_scored', 'goals_conceded']

away = pl[['away_team', 'away_goals', 'home_goals']].copy()
away.columns = ['team', 'goals_scored', 'goals_conceded']

teams_long = pd.concat([home, away])

# Compute averages
team_stats = teams_long.groupby('team').mean()

# League averages
league_avg_scored = teams_long['goals_scored'].mean()
league_avg_conceded = teams_long['goals_conceded'].mean()

# Compute strengths
team_stats['O'] = np.log(team_stats['goals_scored'] / league_avg_scored)
team_stats['D'] = np.log(team_stats['goals_conceded'] / league_avg_conceded)

print(team_stats.head())

mean_home = pl['home_goals'].mean()
mean_away = pl['away_goals'].mean()

H = np.log(mean_home / mean_away)

print("Home advantage H:", H)

def predict_lambda(home_team, away_team):
    O_home = team_stats.loc[home_team, 'O']
    D_home = team_stats.loc[home_team, 'D']
    O_away = team_stats.loc[away_team, 'O']
    D_away = team_stats.loc[away_team, 'D']
    
    lambda_home = mean_home * np.exp(O_home - D_away)
    lambda_away = mean_away * np.exp(O_away - D_home)
    
    return lambda_home, lambda_away

lh, la = predict_lambda("Arsenal", "Chelsea")
print("Expected goals:", lh, la)

def simulate_match(home_team, away_team):
    lh, la = predict_lambda(home_team, away_team)
    
    home_goals = np.random.poisson(lh)
    away_goals = np.random.poisson(la)
    
    return home_goals, away_goals

def simulate_many(home_team, away_team, n=5000):
    results = [simulate_match(home_team, away_team) for _ in range(n)]
    
    home = np.array([r[0] for r in results])
    away = np.array([r[1] for r in results])
    
    return {
        "home_win": np.mean(home > away),
        "draw": np.mean(home == away),
        "away_win": np.mean(home < away)
    }

simulate_many("Arsenal", "Manchester City")


pl[['lambda_home', 'lambda_away']] = pl.apply(
    lambda row: pd.Series(predict_lambda(row['home_team'], row['away_team'])),
    axis=1
)

print("Predicted home:", pl['lambda_home'].mean())
print("Actual home:", pl['home_goals'].mean())

mse_home = np.mean((pl['home_goals'] - pl['lambda_home'])**2)
mse_away = np.mean((pl['away_goals'] - pl['lambda_away'])**2)

print("MSE home:", mse_home)
print("MSE away:", mse_away)

# Now need to add MLE