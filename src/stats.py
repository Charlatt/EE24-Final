from scipy.stats import poisson
import numpy as np

# Returns attack strength, defensive strength, and home advantage
def unpack(theta, number_teams):
    a = theta[:number_teams] # positions 0-19
    d = theta[number_teams:2 * number_teams] # positions 20-39
    h = theta[-1] # position 40
    return a, d, h

# Return log-likelihood
def log_likelihood(theta, home_index, away_index, home_goals, away_goals, number_teams):
    a, d, h = unpack(theta, number_teams)
    
    # Compute lambdas for every match at once
    lambda_home = np.exp(a[home_index] - d[away_index] + h)
    lambda_away = np.exp(a[away_index] - d[home_index])
    
    # Log-likelihood over all 380 matches
    ll = (poisson.logpmf(home_goals, lambda_home).sum() +
          poisson.logpmf(away_goals, lambda_away).sum())
    
    return -ll

def find_mle(home_team, away_team, team_index, a_hat, d_hat, h_hat):
    home = team_index[home_team]
    away = team_index[away_team]

    lambda_home = np.exp(a_hat[home] - d_hat[away] + h_hat)
    lambda_away = np.exp(a_hat[away] - d_hat[home])

    return lambda_home, lambda_away