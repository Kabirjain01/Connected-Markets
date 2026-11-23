# Connected Markets: Predicting Stock Returns with Graph Neural Networks

APS360 course project by Kabir Jain (University of Toronto).

Companies are linked: when a large customer has a bad quarter, its suppliers often fall days later.
This project tests whether a model that knows how companies are connected predicts next-week stock
returns better than one that looks at each company alone.

## What the model does

- **Input:** the last 20 trading days of daily returns and volume changes for 100 large U.S. stocks,
  plus a company graph (sector links and last year's correlation links).
- **Model:** an LSTM summarizes each stock's recent history, then two relational graph attention layers
  let each stock use information from the companies it is linked to (78,273 parameters).
- **Output:** a score that ranks all 100 stocks for the coming week.
- **Baselines:** momentum and reversal rules, ridge regression (returns only, and returns + volume),
  and the same LSTM without the graph.

## Repository contents

| File | What it is |
|---|---|
| `Connected_Markets_progress.ipynb` | Main notebook: data download and cleaning, graph construction, baselines, GNN training and evaluation |
| `progress_report.pdf` | Progress report |
| `report/` | LaTeX source of the progress report (course template), references and figures |

## How to run

1. Open `Connected_Markets_progress.ipynb` in Google Colab.
2. Runtime → Change runtime type → **T4 GPU**.
3. Runtime → **Run all**, and allow Google Drive access.

The notebook downloads the data (Wikipedia S&P 500 list and Yahoo Finance prices), builds the graph,
trains every model and saves figures and `results.json` to `MyDrive/Connected-Markets/`.
The first full run takes about 5 minutes on a T4 GPU. Later runs reload the saved data and models.

## Data splits

| Split | Years | Use |
|---|---|---|
| Train | 2010 to 2019 | model fitting |
| Validation | 2020 to 2021 | early stopping and comparison |
| Test | 2022 to 2026 | sealed until the final report |

## Progress report results (validation, 2020 to 2021)

| Model | Validation IC |
|---|---|
| Ridge, returns only | -0.038 |
| Ridge, returns + volume | -0.039 |
| LSTM, no graph | +0.006 |
| GNN, sector + correlation links | +0.012 |
| GNN, correlation links only | +0.020 |

IC is the average weekly rank correlation between predicted and actual returns (0 means no skill).
The graph versions score higher than the LSTM, but the gain is not yet statistically reliable or
consistent: the GNNs beat the LSTM in about half of the validation weeks. See `progress_report.pdf`
for the full analysis.

## Next steps

- Build supply-chain links from major-customer disclosures in SEC 10-K filings and report verified links per year
- Add directed customer-to-supplier links that pass the customer's previous-week return and volume
  shock through a learned lag gate
- Add returns relative to the market and sector as inputs, and train with a ranking loss
- Final evaluation on the sealed test years and on new data collected after the model is frozen
