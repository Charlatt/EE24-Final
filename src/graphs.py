import matplotlib.pyplot as plt
from simulation import simulate


def generate_graphs(teams, team_index, a_hat, d_hat, h_hat):
    """
    Plot Arsenal's simulated win probability against every other team,
    once as the home side and once as the away side, and save both charts.
    """
    arsenal = "Arsenal"
    home_opps, home_win_probs = [], []
    away_opps, away_win_probs = [], []

    for opponent in teams:
        if opponent == arsenal:
            continue

        # Arsenal at home
        arsenal_home_win, _, _ = simulate(arsenal, opponent, a_hat, d_hat, h_hat, team_index)
        home_opps.append(opponent)
        home_win_probs.append(arsenal_home_win)

        # Arsenal away
        _, arsenal_away_win, _ = simulate(opponent, arsenal, a_hat, d_hat, h_hat, team_index)
        away_opps.append(opponent)
        away_win_probs.append(arsenal_away_win)

    # Graph 1: Arsenal home win probabilities
    plt.figure(figsize=(12, 6))
    plt.bar(home_opps, home_win_probs)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Win Probability")
    plt.xlabel("Opponent")
    plt.title("Arsenal Home Win Probability vs Each Opponent (2024-25)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("arsenal_home_win_probs.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to arsenal_home_win_probs.png")

    # Graph 2: Arsenal away win probabilities
    plt.figure(figsize=(12, 6))
    plt.bar(away_opps, away_win_probs)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Win Probability")
    plt.xlabel("Opponent")
    plt.title("Arsenal Away Win Probability vs Each Opponent (2024-25)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig("arsenal_away_win_probs.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved to arsenal_away_win_probs.png")