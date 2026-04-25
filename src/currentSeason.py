import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import numpy as np
from stats import log_likelihood, unpack, find_mle

url_26 = "https://datahub.io/core/english-premier-league/_r/-/season-2526.csv"

name_map_26 = {
    "Man United":    "Manchester United",
    "Man City":      "Manchester City",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves":        "Wolverhampton Wanderers",
    "Newcastle":     "Newcastle United",
    "Spurs":         "Tottenham",
}

def normalize_name(name):
    return name_map_26.get(name, name)


def currentSeasonSetup():
    raw_26 = pd.read_csv(url_26)

    pl_26 = raw_26.rename(columns={
        "Date":     "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG":     "home_goals",
        "FTAG":     "away_goals"
    })

    pl_26["date"] = pd.to_datetime(pl_26["date"], dayfirst=True, format="mixed")
    pl_26["home_team"] = pl_26["home_team"].apply(normalize_name)
    pl_26["away_team"] = pl_26["away_team"].apply(normalize_name)

    pl_26 = pl_26[[
        "date", "home_team", "away_team", "home_goals", "away_goals"
    ]].dropna().copy()

    pl_26["home_goals"] = pl_26["home_goals"].astype(int)
    pl_26["away_goals"] = pl_26["away_goals"].astype(int)

    print(f"Completed matches: {len(pl_26)}")
    print(f"Most recent match: {pl_26['date'].max()}")

    extra_matches = pd.DataFrame([
        ("Brentford",             "Fulham",                   0, 0),
        ("Leeds",                 "Wolverhampton Wanderers",  3, 0),
        ("Newcastle United",      "Bournemouth",              1, 2),
        ("Tottenham",             "Brighton",                 2, 2),
        ("Chelsea",               "Manchester United",        0, 1),
        ("Aston Villa",           "Sunderland",               4, 3),
        ("Everton",               "Liverpool",                1, 2),
        ("Nottingham Forest",     "Burnley",                  4, 1),
        ("Manchester City",       "Arsenal",                  2, 1),
        ("Crystal Palace",        "West Ham",                 0, 0),
    ], columns=["home_team", "away_team", "home_goals", "away_goals"])

    pl_26 = pd.concat([pl_26, extra_matches], ignore_index=True)

    teams_26 = sorted(set(pl_26["home_team"]).union(pl_26["away_team"]))
    team_index_26 = {team: i for i, team in enumerate(teams_26)}
    num_teams_26 = len(teams_26)

    print(f"Unique teams: {teams_26}")

    home_index_26 = np.array([team_index_26[t] for t in pl_26["home_team"]])
    away_index_26 = np.array([team_index_26[t] for t in pl_26["away_team"]])
    home_goals_26 = np.array(pl_26["home_goals"]).astype(int)
    away_goals_26 = np.array(pl_26["away_goals"]).astype(int)

    theta0_26 = np.zeros(2 * num_teams_26 + 1)

    result_26 = minimize(
        log_likelihood,
        theta0_26,
        args=(home_index_26, away_index_26, home_goals_26, away_goals_26, num_teams_26),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-12}
    )

    theta_hat_26 = result_26.x
    a_hat_26, d_hat_26, h_hat_26 = unpack(theta_hat_26, num_teams_26)

    print("Optimization success:", result_26.success)
    print("Home advantage:", h_hat_26)

    print("\nTeam strengths (2025-26):")
    print(f"{'Team':<25} {'Attack':>8} {'Defense':>8}")
    print("-" * 43)
    for team in sorted(teams_26, key=lambda t: -a_hat_26[team_index_26[t]]):
        i = team_index_26[team]
        print(f"{team:<25} {a_hat_26[i]:>8.3f} {d_hat_26[i]:>8.3f}")

    title_count, top4_count, relegated_count = simulate_remaining(
        team_index_26, a_hat_26, d_hat_26, h_hat_26
    )
    plot_simulation_results(title_count, top4_count, relegated_count, n=10000)


current_standings = {
    "Arsenal":                 {"pts": 70, "gp": 33},
    "Manchester City":         {"pts": 67, "gp": 32},
    "Manchester United":       {"pts": 58, "gp": 33},
    "Aston Villa":             {"pts": 58, "gp": 33},
    "Liverpool":               {"pts": 55, "gp": 33},
    "Chelsea":                 {"pts": 48, "gp": 33},
    "Brentford":               {"pts": 48, "gp": 33},
    "Bournemouth":             {"pts": 48, "gp": 33},
    "Brighton":                {"pts": 47, "gp": 33},
    "Everton":                 {"pts": 47, "gp": 33},
    "Sunderland":              {"pts": 46, "gp": 33},
    "Fulham":                  {"pts": 45, "gp": 33},
    "Crystal Palace":          {"pts": 42, "gp": 31},
    "Newcastle United":        {"pts": 42, "gp": 33},
    "Leeds":                   {"pts": 39, "gp": 33},
    "Nottingham Forest":       {"pts": 36, "gp": 33},
    "West Ham":                {"pts": 32, "gp": 32},
    "Tottenham":               {"pts": 31, "gp": 33},
    "Burnley":                 {"pts": 20, "gp": 33},
    "Wolverhampton Wanderers": {"pts": 17, "gp": 33},
}

remaining_fixtures = pd.DataFrame([
    # MD34 - April 21-22
    ("Brighton",               "Chelsea"),
    ("Bournemouth",            "Leeds"),
    # MD34 continued
    ("Arsenal",                "Nottingham Forest"),
    # MD35 - April 25-27
    ("Bournemouth",            "Leeds"),
    ("Arsenal",                "Newcastle United"),
    ("Brighton",               "Chelsea"),
    ("Burnley",                "Manchester City"),
    ("Fulham",                 "Aston Villa"),
    ("Liverpool",              "Crystal Palace"),
    ("Manchester United",      "Brentford"),
    ("Sunderland",             "Nottingham Forest"),
    ("West Ham",               "Everton"),
    ("Wolverhampton Wanderers","Tottenham"),
    # MD36 - May 2-4
    ("Bournemouth",            "Crystal Palace"),
    ("Arsenal",                "Fulham"),
    ("Aston Villa",            "Tottenham"),
    ("Brentford",              "West Ham"),
    ("Chelsea",                "Nottingham Forest"),
    ("Everton",                "Manchester City"),
    ("Leeds",                  "Burnley"),
    ("Manchester United",      "Liverpool"),
    ("Newcastle United",       "Brighton"),
    ("Wolverhampton Wanderers","Sunderland"),
    # MD37 - May 9-10
    ("Brighton",               "Wolverhampton Wanderers"),
    ("Burnley",                "Aston Villa"),
    ("Crystal Palace",         "Everton"),
    ("Fulham",                 "Bournemouth"),
    ("Liverpool",              "Chelsea"),
    ("Manchester City",        "Brentford"),
    ("Nottingham Forest",      "Newcastle United"),
    ("Sunderland",             "Manchester United"),
    ("West Ham",               "Arsenal"),
    # MD38 - May 17
    ("Bournemouth",            "Manchester City"),
    ("Arsenal",                "Burnley"),
    ("Aston Villa",            "Liverpool"),
    ("Brentford",              "Crystal Palace"),
    ("Chelsea",                "Tottenham"),
    ("Everton",                "Sunderland"),
    ("Leeds",                  "Brighton"),
    ("Manchester United",      "Nottingham Forest"),
    ("Newcastle United",       "West Ham"),
    ("Wolverhampton Wanderers","Fulham"),
    # Final day - May 24
    ("Brighton",               "Manchester United"),
    ("Burnley",                "Wolverhampton Wanderers"),
    ("Crystal Palace",         "Arsenal"),
    ("Fulham",                 "Newcastle United"),
    ("Liverpool",              "Brentford"),
    ("Manchester City",        "Aston Villa"),
    ("Nottingham Forest",      "Bournemouth"),
    ("Sunderland",             "Chelsea"),
    ("Tottenham",              "Everton"),
    ("West Ham",               "Leeds"),
], columns=["HomeTeam", "AwayTeam"])


def simulate_remaining(team_index_26, a_hat_26, d_hat_26, h_hat_26, n=10000):
    teams_sim = list(current_standings.keys())

    title_count     = {t: 0 for t in teams_sim}
    top4_count      = {t: 0 for t in teams_sim}
    relegated_count = {t: 0 for t in teams_sim}

    for _ in range(n):
        pts = {t: current_standings[t]["pts"] for t in teams_sim}

        for _, row in remaining_fixtures.iterrows():
            home = row["HomeTeam"]
            away = row["AwayTeam"]

            lh, la = find_mle(home, away, team_index_26, a_hat_26, d_hat_26, h_hat_26)

            hg = np.random.poisson(lh)
            ag = np.random.poisson(la)

            if hg > ag:
                pts[home] += 3
            elif hg == ag:
                pts[home] += 1
                pts[away] += 1
            else:
                pts[away] += 3

        ranked = sorted(teams_sim, key=lambda t: pts[t], reverse=True)

        title_count[ranked[0]] += 1
        for t in ranked[:4]:
            top4_count[t] += 1
        for t in ranked[-3:]:
            relegated_count[t] += 1

    print("\n TITLE WINNER PROBABILITY")
    print(f"{'Team':<25} {'Probability':>12}")
    print("-" * 39)
    for team, count in sorted(title_count.items(), key=lambda x: -x[1]):
        if count/n >= 0.001:
            print(f"{team:<25} {count/n*100:>11.1f}%")

    print("\n TOP 4 PROBABILITY")
    print(f"{'Team':<25} {'Probability':>12}")
    print("-" * 39)
    for team, count in sorted(top4_count.items(), key=lambda x: -x[1]):
        if count/n >= 0.001:
            print(f"{team:<25} {count/n*100:>11.1f}%")

    print("\n  RELEGATION PROBABILITY")
    print(f"{'Team':<25} {'Probability':>12}")
    print("-" * 39)
    for team, count in sorted(relegated_count.items(), key=lambda x: -x[1]):
        if count/n >= 0.001:
            print(f"{team:<25} {count/n*100:>11.1f}%")

    return title_count, top4_count, relegated_count


def plot_simulation_results(title_count, top4_count, relegated_count, n, threshold=0.001):
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.patch.set_facecolor("#ffffff")

    pink_dark  = "#c2185b"
    pink_mid   = "#e91e8c"
    pink_light = "#f48fb1"
    bg         = "#ffffff"

    def draw_chart(ax, count_dict, title, color):
        data = {t: c/n*100 for t, c in count_dict.items() if c/n >= threshold}
        data = dict(sorted(data.items(), key=lambda x: -x[1]))

        teams = list(data.keys())
        probs = list(data.values())

        ax.set_facecolor(bg)
        bars = ax.bar(teams, probs, color=color, edgecolor="white", linewidth=0.5)

        for bar, prob in zip(bars, probs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{prob:.1f}%",
                va="bottom", ha="center",
                fontsize=9, color=pink_dark, fontweight="bold"
            )

        ax.set_ylim(0, 115)
        ax.set_title(title, fontsize=13, fontweight="bold", color=pink_dark, pad=12)
        ax.set_ylabel("Probability (%)", fontsize=10, color=pink_dark)
        ax.tick_params(axis="x", colors=pink_dark, rotation=45)
        ax.tick_params(axis="y", colors=pink_dark)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color(pink_light)
        ax.spines["bottom"].set_color(pink_light)
        for label in ax.get_xticklabels():
            label.set_color(pink_dark)
            label.set_fontsize(9)
            label.set_ha("right")
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=pink_light, linestyle="--", linewidth=0.7)

    draw_chart(axes[0], title_count,     "Title Winner",  pink_mid)
    draw_chart(axes[1], top4_count,      "Top 4 Finish",  pink_light)
    draw_chart(axes[2], relegated_count, "Relegation",    pink_dark)

    fig.suptitle("2025-26 Premier League Season Simulation",
                 fontsize=16, fontweight="bold", color=pink_dark, y=1.01)

    plt.tight_layout()
    plt.savefig("simulation_results.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print("Saved to simulation_results.png")