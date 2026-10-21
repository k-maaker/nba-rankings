# NBA Betting Market Rankings

A self-updating recreation of inpredictable's NBA betting market rankings:
what the betting market thinks of every team, worked out from closing point
spreads and over/unders.

Live page: https://k-maaker.github.io/nba-rankings/

## How it updates

1. BigDataBall drops a new season file (e.g. `10-21-2026-nba-season-team-feed.xlsx`)
   into the shared Dropbox folder overnight.
2. cron-job.org presses this repo's "Run workflow" button every morning
   (GitHub's own scheduler runs hours late, so it isn't used).
3. The workflow downloads the Dropbox folder, keeps the newest NBA file,
   adds tonight's lines from ESPN, rebuilds the rankings and publishes
   `docs/index.html`.

## The model

`team points = league average + offense(team) - defense(opponent) ± home court/2 ∓ back-to-back/2`

- Every game's closing spread and total are split into expected points for
  each team. Each is a row in a weighted regression, weighted
  `1 / (games since + C_LINE)`, so recent lines count most.
- Actual scores are added too, at a much lower weight
  `1 / (games since + C_RESULT)`.
- Last season's games count as `OFFSEASON_GAP` (200) games older, so they anchor
  October and are quickly outweighed by this season's lines.
- Home court and the back-to-back penalty are estimated from the lines.
- The three constants were picked by backtesting the 2023-24, 2024-25 and
  2025-26 seasons: the values that best predicted the next day's closing
  spreads.

## Files

| File | What it is |
|---|---|
| `nba_rankings.py` | The model and the web page |
| `fetch_bigdataball.py` | Downloads the Dropbox folder and picks the newest NBA file |
| `data/games.csv` | Every game: 3 past seasons plus this one (scores, closing spread/total, back-to-backs) |
| `data/history.csv` | Each team's rating each day (trend lines and "last week") |
| `docs/index.html` | The published page |
| `.github/workflows/update.yml` | The morning job |

## Upkeep

- **GitHub token** used by cron-job.org expires on the date you chose. When
  it does, make a new one and paste it into the cron-job.org job's
  Authorization header (`Bearer github_pat_...`).
- **Dropbox link** is stored as the repo secret `DROPBOX_FOLDER_URL`. If
  BigDataBall re-shares the folder, update the secret.
- If the page looks stale, check the Actions tab for a red X.
