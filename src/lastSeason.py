import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from stats import log_likelihood, unpack, find_mle
from simulation import results, expected_points_one_team
from graphs import generate_graphs

# ESPN dataset season identifier for 2024-25 Premier League
SEASON_TYPE = 12654


def lastSeasonSetup():
    """
    Download the ESPN soccer dataset, filter to completed 2024-25 Premier
    League matches, fit the Dixon-Coles Poisson model via MLE, then run
    partOne, partThree, and generate_graphs with the fitted parameters.
    """
    import kagglehub
    path = kagglehub.dataset_download("excel4soccer/espn-soccer-data")
    BASE = os.path.join(path, "base_data")

    fixtures = pd.read_csv(f"{BASE}/fixtures.csv")
    teams    = pd.read_csv(f"{BASE}/teams.csv")
    leagues  = pd.read_csv(f"{BASE}/leagues.csv")

    fixtures["date"] = pd.to_datetime(fixtures["date"])

    # Filter to completed (statusId == 28) 2024-25 Premier League fixtures
    pl = fixtures[
        (fixtures["leagueId"]   == 700) &
        (fixtures["seasonType"] == SEASON_TYPE)
    ].copy()
    pl = pl[pl["statusId"] == 28].copy()

    # Join team names onto home and away sides
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

    pl = pl.rename(columns={
        "homeTeamScore": "home_goals",
        "awayTeamScore": "away_goals"
    })

    pl = pl[["date", "home_team", "away_team", "home_goals", "away_goals"]].copy()

    # Build a sorted team list and map each team name to a numeric index
    team_list  = sorted(set(pl["home_team"]).union(pl["away_team"]))
    team_index = {team: i for i, team in enumerate(team_list)}
    num_teams  = len(team_list)

    # Convert to arrays for vectorised log-likelihood computation
    home_index = np.array([team_index[t] for t in pl["home_team"]])
    away_index = np.array([team_index[t] for t in pl["away_team"]])
    home_goals = np.array(pl["home_goals"]).astype(int)
    away_goals = np.array(pl["away_goals"]).astype(int)

    # theta = [a_0..a_n, d_0..d_n, h]  (attack, defense, home advantage)
    theta0 = np.zeros(2 * num_teams + 1)

    result = minimize(
        log_likelihood,
        theta0,
        args=(home_index, away_index, home_goals, away_goals, num_teams),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12}
    )

    theta_hat            = result.x
    a_hat, d_hat, h_hat  = unpack(theta_hat, num_teams)

    partOne(team_list, team_index, a_hat, d_hat, h_hat)
    partThree(team_list, team_index, a_hat, d_hat, h_hat)
    generate_graphs(team_list, team_index, a_hat, d_hat, h_hat)


def partOne(teams, team_index, a_hat, d_hat, h_hat):
    """
    Print simulated win probabilities and expected goals for three
    classic Premier League rivalries using 2024-25 fitted parameters.
    """
    match1 = results("Tottenham Hotspur", "Arsenal", a_hat, d_hat, h_hat, team_index)
    for key, value in match1.items():
        print(f"{key}: {value:.2f}")

    match2 = results("Manchester United", "Manchester City", a_hat, d_hat, h_hat, team_index)
    for key, value in match2.items():
        print(f"{key}: {value:.2f}")

    match3 = results("Liverpool", "Everton", a_hat, d_hat, h_hat, team_index)
    for key, value in match3.items():
        print(f"{key}: {value:.2f}")


def partThree(teams, team_index, a_hat, d_hat, h_hat):
    """
    Print a predicted final table ranked by each team's expected points
    across a full home-and-away season using 2024-25 fitted parameters.
    """
    print("\nExpected table:")
    table = [
        (team, expected_points_one_team(team, teams, a_hat, d_hat, h_hat, team_index))
        for team in teams
    ]
    table.sort(key=lambda x: x[1], reverse=True)
    for team, pts in table:
        print(f"{team}: {pts:.1f}")