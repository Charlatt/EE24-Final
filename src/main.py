import pandas as pd
import numpy as np

BASE = "/Users/WILL/.cache/kagglehub/datasets/excel4soccer/espn-soccer-data/versions/526/base_data"

fixtures = pd.read_csv(f"{BASE}/fixtures.csv")
teams = pd.read_csv(f"{BASE}/teams.csv")
leagues = pd.read_csv(f"{BASE}/leagues.csv")
status = pd.read_csv(f"{BASE}/status.csv")

# --------------------------------------------------
# 1. Find the correct seasonType for 2024-25 EPL
# --------------------------------------------------
pl_rows = leagues[leagues["leagueId"] == 700].copy()
print("Premier League rows:")
print(pl_rows[["year", "seasonName", "seasonSlug", "seasonType", "leagueId"]])

# Replace this after inspecting the print output
SEASON_TYPE = 12654

# --------------------------------------------------
# 2. Convert dates
# --------------------------------------------------
fixtures["date"] = pd.to_datetime(fixtures["date"])

# --------------------------------------------------
# 3. Filter to 2024-25 Premier League
# --------------------------------------------------
pl = fixtures[
    (fixtures["leagueId"] == 700) &
    (fixtures["seasonType"] == SEASON_TYPE)
].copy()

# --------------------------------------------------
# 4. Keep only completed matches
# Replace 28 if your status table shows another code
# --------------------------------------------------
pl = pl[pl["statusId"] == 28].copy()

# --------------------------------------------------
# 5. Merge team names
# --------------------------------------------------
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

# --------------------------------------------------
# 6. Rename scores and keep only needed columns
# --------------------------------------------------
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

# --------------------------------------------------
# 7. Basic sanity checks
# --------------------------------------------------
print(pl.head())
print("Matches:", len(pl))
print("Unique teams:", len(set(pl["home_team"]).union(set(pl["away_team"]))))
print("Average home goals:", pl["home_goals"].mean())
print("Average away goals:", pl["away_goals"].mean())