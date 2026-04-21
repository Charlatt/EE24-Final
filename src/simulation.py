import numpy as np
import random

n = 10000

def results(home_team, away_team, find_mle):
    lambda_home, lambda_away = find_mle(home_team, away_team)
    
    home = np.random.poisson(lambda_home, n)
    away = np.random.poisson(lambda_away, n)

    return {
        f"{home_team} win probability": np.mean(home > away),
        f"{away_team} win probability": np.mean(home < away),
        "Draw probability": np.mean(home == away),
        f"{home_team} expected goals: ":  lambda_home,
        f"{away_team} expected goals":  lambda_away,
    }


def simulate(home_team, away_team, find_mle):
    lambda_home, lambda_away = find_mle(home_team, away_team)
    
    home = np.random.poisson(lambda_home, n)
    away = np.random.poisson(lambda_away, n)

    home_prob = np.mean(home > away)
    away_prob = np.mean(home < away)
    draw_prob = np.mean(home == away)

    return home_prob, away_prob, draw_prob


def expected_points_one_team(input_team, all_teams, find_mle):
    total_points = 0

    for opponent in all_teams:
        if input_team == opponent:
            continue

        # input team at home
        team_prob, opponent_prob, draw_prob = simulate(input_team, opponent, find_mle)
        total_points += 3 * team_prob + draw_prob

        # input team away
        opponent_prob, team_prob, draw_prob = simulate(opponent, input_team, find_mle)
        total_points += 3 * team_prob + draw_prob

    return total_points






