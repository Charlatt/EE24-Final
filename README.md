# EE24-Final
# Premier League Match Simulator

A Poisson-based match simulation model for the English Premier League. Uses Maximum Likelihood Estimation (MLE) to fit team attack and defense strengths from historical results, then simulates matches and full seasons using those parameters.

## How It Works

The model uses a log-linear Poisson regression approach to football prediction. Each team is assigned two parameters — an **attack strength** and a **defense strength** — plus a shared **home advantage** parameter. Goals scored by the home and away teams in a given match are modelled as independent Poisson random variables:

```
λ_home = exp(attack_home − defense_away + home_advantage)
λ_away = exp(attack_away − defense_home)
```

These parameters are fitted by minimising the negative log-likelihood over all completed matches using L-BFGS-B optimisation. Once fitted, match outcomes are simulated by drawing from the resulting Poisson distributions 10,000 times.

## Project Structure

```
src/
├── main.py           # Entry point — runs both seasons in sequence
├── stats.py          # MLE core: log_likelihood, unpack, find_mle
├── simulation.py     # Match simulation and expected points functions
├── lastSeason.py     # 2024-25 season: data loading, fitting, and analysis
├── currentSeason.py  # 2025-26 season: data loading, fitting, and simulation
└── graphs.py         # Arsenal win-probability bar charts (2024-25)
```

## File Descriptions

### `main.py`
Entry point. Calls `lastSeasonSetup()` followed by `currentSeasonSetup()` to run the full pipeline for both seasons.

### `stats.py`
Core statistical functions used by both seasons:
- `log_likelihood` — computes the negative log-likelihood of observed scorelines given a parameter vector theta
- `unpack` — splits theta into attack array, defense array, and home advantage scalar
- `find_mle` — returns the fitted Poisson lambda values (expected goals) for a given home/away matchup

### `simulation.py`
Monte Carlo simulation functions built on top of the fitted parameters:
- `results` — simulates a single match and returns win probabilities and expected goals for both teams
- `simulate` — lighter version of `results`, returns only `(home_prob, away_prob, draw_prob)`
- `expected_points_one_team` — estimates a team's total expected points across a full home-and-away season

### `lastSeason.py`
Handles the 2024-25 Premier League season using the ESPN soccer dataset from Kaggle:
- `lastSeasonSetup` — downloads the dataset, filters to completed PL matches, fits the model, and runs all analysis
- `partOne` — prints simulated probabilities and expected goals for three classic derbies (North London, Manchester, Merseyside)
- `partThree` — prints a predicted final table ranked by expected points

### `currentSeason.py`
Handles the 2025-26 Premier League season using a live CSV feed from DataHub:
- `currentSeasonSetup` — fetches results, appends any recent matches not yet in the feed, fits the model, and prints team strengths
- `simulate_remaining` — runs 10,000 season simulations from the current standings across all remaining fixtures, printing title, top-4, and relegation probabilities
- `plot_simulation_results` — renders and saves a three-panel bar chart of the simulation output

### `graphs.py`
- `generate_graphs` — plots Arsenal's simulated win probability against every other 2024-25 team, once as home side and once as away side, and saves both charts as PNGs

## Data Sources

| Season | Source | Details |

| 2024-25 | [ESPN Soccer Data](https://www.kaggle.com/datasets/excel4soccer/espn-soccer-data) via `kagglehub` | Downloaded automatically on first run |
| 2025-26 | [DataHub](https://datahub.io/core/english-premier-league) | Fetched live from a public CSV URL |

## Setup

Install dependencies:

```bash
pip install pandas numpy scipy matplotlib kagglehub
```

A Kaggle account and API token are required for the `kagglehub` download. Follow the [Kaggle API setup guide](https://github.com/Kaggle/kaggle-api#api-credentials) if you haven't done this before.

## Usage

```bash
python main.py
```

This will:
1. Download and process the 2024-25 ESPN dataset
2. Fit the Poisson model and print team strengths
3. Print derby match probabilities and a predicted final table
4. Generate and save Arsenal win-probability graphs
5. Fetch and process the 2025-26 DataHub results
6. Fit the model for the current season and print team strengths
7. Simulate the remaining fixtures and print title/top-4/relegation probabilities
8. Save the simulation results chart as `simulation_results.png`

## Output Files

| File | Description |

| `arsenal_home_win_probs.png` | Arsenal home win probability vs each 2024-25 opponent |
| `arsenal_away_win_probs.png` | Arsenal away win probability vs each 2024-25 opponent |
| `simulation_results.png` | 2025-26 title, top-4, and relegation probability charts |


