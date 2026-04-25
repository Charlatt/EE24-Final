import numpy as np
from stats import find_mle

# Number of Monte Carlo samples used for all probability estimates
N_SIMULATIONS = 10000


def results(home_team, away_team, a_hat, d_hat, h_hat, team_index):
    """
    Simulate a single match and return a dict of win probabilities
    and expected goals for both teams.
    """
    # Get Poisson rate parameters (expected goals) for each team
    lambda_home, lambda_away = find_mle(home_team, away_team, team_index, a_hat, d_hat, h_hat)

    # Draw N_SIMULATIONS match outcomes from the Poisson distributions
    home = np.random.poisson(lambda_home, N_SIMULATIONS)
    away = np.random.poisson(lambda_away, N_SIMULATIONS)

    return {
        f"{home_team} win probability":  np.mean(home > away),
        f"{away_team} win probability":  np.mean(home < away),
        "Draw probability":              np.mean(home == away),
        f"{home_team} expected goals":   lambda_home,
        f"{away_team} expected goals":   lambda_away,
    }


def simulate(home_team, away_team, a_hat, d_hat, h_hat, team_index):
    """
    Simulate a single match and return (home_prob, away_prob, draw_prob).
    Lighter than results() — used internally when only probabilities are needed.
    """
    lambda_home, lambda_away = find_mle(home_team, away_team, team_index, a_hat, d_hat, h_hat)

    home = np.random.poisson(lambda_home, N_SIMULATIONS)
    away = np.random.poisson(lambda_away, N_SIMULATIONS)

    home_prob = np.mean(home > away)
    away_prob = np.mean(home < away)
    draw_prob = np.mean(home == away)

    return home_prob, away_prob, draw_prob


def expected_points_one_team(input_team, all_teams, a_hat, d_hat, h_hat, team_index):
    """
    Estimate the total expected points for input_team across a full
    home-and-away season against every other team in all_teams.
    Each match contributes 3 * win_prob + draw_prob expected points.
    """
    total_points = 0

    for opponent in all_teams:
        if input_team == opponent:
            continue

        # input team at home
        team_prob, _, draw_prob = simulate(input_team, opponent, a_hat, d_hat, h_hat, team_index)
        total_points += 3 * team_prob + draw_prob

        # input team away
        _, team_prob, draw_prob = simulate(opponent, input_team, a_hat, d_hat, h_hat, team_index)
        total_points += 3 * team_prob + draw_prob

    return total_points