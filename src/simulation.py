import numpy as np

# home team, away team, number of trials
def simulate(home_team, away_team, n, find_mle):
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

def simulate_one_team(team, find_mle, all_teams):
    for (i in (2 * all_teams.length())):
        if (team = all_teams[i]): 
            continue
        else:
            # team as home team
            lambda_home, lambda_away = find_mle(home_team, away_team)
            
            # team as away team
            
