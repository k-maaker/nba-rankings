#!/usr/bin/env python3
"""
NBA Betting Market Rankings, a recreation of inpredictable's NBA rankings.

What the market thinks of every team, reverse-engineered from closing point
spreads and totals with a recency-weighted regression:

    team points = league avg + offense(team) - defense(opponent)
                  +/- home court/2  -/+ back-to-back/2

  * Market lines (closing spread + total) are split into expected points
    for each side; each side is one regression row, weighted
    1 / (elapsed games + C_LINE).
  * Actual scores are added as extra rows, weighted
    1 / (elapsed games + C_RESULT), so the most recent result counts for
    much less than the most recent line.
  * "Elapsed games" counts how many games that team has played since. Games
    from last season count OFFSEASON_GAP extra games old, so last season acts
    as a whole-season-average prior: it carries teams that haven't played yet
    in October and is quickly outweighed once this season's lines arrive.
  * Home court and the back-to-back penalty are estimated inside the
    regression (the market's own values), not fixed by hand.
  * The three constants above were chosen by backtesting on the 2023-24,
    2024-25 and 2025-26 BigDataBall files: the values that best predicted
    the next day's closing spreads.

Outputs
  GPF  = points favored vs. a league-average team on a neutral court
  oGPF = offensive part of GPF (includes pace)
  dGPF = defensive part of GPF (positive = good defense)
  GOU  = expected over/under vs. a league-average team

Data
  data/games.csv          every game we know about (3 past seasons + this one)
  data/bigdataball/*.xlsx today's BigDataBall season file (pulled from Dropbox)
  data/history.csv        one row per team per day, for trends and "last week"
  ESPN's public scoreboard supplies tonight's lines (BigDataBall only has
  finished games), which the market model treats as the freshest info.

Usage
  python nba_rankings.py              # ingest, fetch tonight's lines, rebuild page
  python nba_rankings.py --no-fetch   # skip ESPN (offline rebuild)
  python nba_rankings.py --asof 2026-01-15   # rebuild as of a past date (testing)
"""

import argparse
import datetime as dt
import json
import os
import re
import sys
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "data")
OUT_DIR = os.path.join(BASE, "docs")          # GitHub Pages serves /docs
GAMES_CSV = os.path.join(DATA_DIR, "games.csv")
HISTORY_CSV = os.path.join(DATA_DIR, "history.csv")
BDB_DIR = os.path.join(DATA_DIR, "bigdataball")
EASTERN = ZoneInfo("America/New_York")

# Tuned by backtest on 2024-25 and 2025-26 (each fit only on data known
# before the day it predicts). Mean absolute error vs. the actual closing
# spread: 2.31 points over both seasons, 2.09 in each season's first month.
# The line/result constants landed on the same values inpredictable uses.
C_LINE = 0.25          # market-line weight: 1 / (elapsed + C_LINE)
C_RESULT = 3.5         # game-result weight: 1 / (elapsed + C_RESULT)
OFFSEASON_GAP = 200    # last season's games count this many extra games old

SPARK_N = 30
SCOREBOARD_URL = ("https://site.api.espn.com/apis/site/v2/sports/basketball/"
                  "nba/scoreboard?dates={date}")

# abbr, full name, BigDataBall short name, other spellings (ESPN etc.)
TEAMS = [
    ("ATL", "Atlanta Hawks", "Atlanta", []),
    ("BOS", "Boston Celtics", "Boston", []),
    ("BKN", "Brooklyn Nets", "Brooklyn", ["BRK", "BRO"]),
    ("CHA", "Charlotte Hornets", "Charlotte", ["CHO"]),
    ("CHI", "Chicago Bulls", "Chicago", []),
    ("CLE", "Cleveland Cavaliers", "Cleveland", []),
    ("DAL", "Dallas Mavericks", "Dallas", []),
    ("DEN", "Denver Nuggets", "Denver", []),
    ("DET", "Detroit Pistons", "Detroit", []),
    ("GSW", "Golden State Warriors", "Golden State", ["GS"]),
    ("HOU", "Houston Rockets", "Houston", []),
    ("IND", "Indiana Pacers", "Indiana", []),
    ("LAC", "Los Angeles Clippers", "LA Clippers", []),
    ("LAL", "Los Angeles Lakers", "LA Lakers", []),
    ("MEM", "Memphis Grizzlies", "Memphis", []),
    ("MIA", "Miami Heat", "Miami", []),
    ("MIL", "Milwaukee Bucks", "Milwaukee", []),
    ("MIN", "Minnesota Timberwolves", "Minnesota", []),
    ("NOP", "New Orleans Pelicans", "New Orleans", ["NO", "NOR"]),
    ("NYK", "New York Knicks", "New York", ["NY"]),
    ("OKC", "Oklahoma City Thunder", "Oklahoma City", []),
    ("ORL", "Orlando Magic", "Orlando", []),
    ("PHI", "Philadelphia 76ers", "Philadelphia", []),
    ("PHX", "Phoenix Suns", "Phoenix", ["PHO"]),
    ("POR", "Portland Trail Blazers", "Portland", []),
    ("SAC", "Sacramento Kings", "Sacramento", []),
    ("SAS", "San Antonio Spurs", "San Antonio", ["SA"]),
    ("TOR", "Toronto Raptors", "Toronto", []),
    ("UTA", "Utah Jazz", "Utah", ["UTAH"]),
    ("WAS", "Washington Wizards", "Washington", ["WSH"]),
]
ABBRS = [t[0] for t in TEAMS]
TEAM_NAMES = {t[0]: t[1] for t in TEAMS}
NAME_TO_ABBR = {}
for _a, _full, _short, _alts in TEAMS:
    for _n in [_a, _full, _short, *_alts]:
        NAME_TO_ABBR[_n.upper()] = _a
NAME_TO_ABBR["LOS ANGELES CLIPPERS"] = "LAC"
NAME_TO_ABBR["L.A. CLIPPERS"] = "LAC"
NAME_TO_ABBR["L.A. LAKERS"] = "LAL"


def team_abbr(name) -> str | None:
    if name is None:
        return None
    return NAME_TO_ABBR.get(str(name).strip().upper())


def season_of(date_str: str) -> int:
    """NBA season label = the year it starts (Oct 2026 - Jun 2027 -> 2026)."""
    d = dt.date.fromisoformat(date_str)
    return d.year if d.month >= 8 else d.year - 1


GAME_COLS = ["game_id", "date", "season", "dataset", "home", "away", "neutral",
             "home_score", "away_score", "spread_home", "total",
             "home_b2b", "away_b2b"]


# ----------------------------------------------------------------------------
# BigDataBall workbook -> one row per game
# ----------------------------------------------------------------------------
def _norm_header(h) -> str:
    return re.sub(r"\s+", " ", str(h or "").replace("\n", " ")).strip().upper()


def parse_bigdataball_xlsx(path: str) -> pd.DataFrame:
    from openpyxl import load_workbook

    wb = load_workbook(path, read_only=True, data_only=True)
    skip = {"METADATA", "TEAMS", "CONVERT DATE FORMAT"}
    sheet = next((s for s in wb.sheetnames if s.upper() not in skip),
                 wb.sheetnames[0])
    rows = list(wb[sheet].iter_rows(values_only=True))
    if not rows:
        return pd.DataFrame(columns=GAME_COLS)
    hdr = [_norm_header(h) for h in rows[0]]

    def find(*names, startswith=None):
        for n in names:
            if n in hdr:
                return hdr.index(n)
        if startswith:
            for i, h in enumerate(hdr):
                if h.startswith(startswith):
                    return i
        return None

    c_ds = find("BIGDATABALL DATASET", startswith="BIGDATABALL")
    c_id = find("GAME-ID", "GAME ID")
    c_date = find("DATE")
    c_team = find("TEAM", "TEAMS")
    c_ven = find(startswith="VENUE")
    c_f = find("F", "PTS")
    c_rest = find("TEAM REST DAYS", startswith="TEAM REST")
    c_cs = find("CLOSING SPREAD")
    c_ct = find("CLOSING TOTAL", "CLOSING O/U")
    c_os = find("OPENING SPREAD")
    c_ot = find("OPENING TOTAL", "OPENING O/U")
    if None in (c_id, c_date, c_team, c_ven, c_f):
        raise ValueError(f"unrecognized BigDataBall layout in {os.path.basename(path)}")

    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def iso(v):
        if isinstance(v, dt.datetime):
            return v.date().isoformat()
        if isinstance(v, dt.date):
            return v.isoformat()
        m, d, y = str(v).strip().split("/")
        return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

    by_game: dict[str, list] = {}
    for r in rows[1:]:
        if not r or r[c_id] in (None, ""):
            continue
        abbr = team_abbr(r[c_team])
        if abbr is None:
            continue  # exhibition / unknown team
        ven = str(r[c_ven] or "").strip().upper()[:1]  # R / H / N
        spread = num(r[c_cs]) if c_cs is not None else None
        if spread is None and c_os is not None:
            spread = num(r[c_os])
        total = num(r[c_ct]) if c_ct is not None else None
        if total is None and c_ot is not None:
            total = num(r[c_ot])
        rest = str(r[c_rest]).upper() if c_rest is not None else ""
        by_game.setdefault(str(r[c_id]), []).append({
            "date": iso(r[c_date]), "team": abbr, "venue": ven,
            "pts": num(r[c_f]), "spread": spread, "total": total,
            "b2b": 1 if "B2B" in rest else 0,
            "dataset": str(r[c_ds]) if c_ds is not None else "",
        })

    out = []
    for gid, sides in by_game.items():
        if len(sides) != 2:
            continue
        a, b = sides
        if a["venue"] == "H":
            a, b = b, a
        # now b is home (or, for neutral sites, the second-listed team)
        home, away = b, a
        neutral = int(home["venue"] == "N" or away["venue"] == "N")
        spread_home = home["spread"]
        if spread_home is None and away["spread"] is not None:
            spread_home = -away["spread"]
        total = home["total"] if home["total"] is not None else away["total"]
        out.append({
            "game_id": gid, "date": home["date"], "season": season_of(home["date"]),
            "dataset": home["dataset"], "home": home["team"], "away": away["team"],
            "neutral": neutral,
            "home_score": home["pts"], "away_score": away["pts"],
            "spread_home": spread_home, "total": total,
            "home_b2b": home["b2b"], "away_b2b": away["b2b"],
        })
    df = pd.DataFrame(out, columns=GAME_COLS)
    return df.sort_values(["date", "game_id"]).reset_index(drop=True)


def load_games() -> pd.DataFrame:
    if not os.path.exists(GAMES_CSV):
        return pd.DataFrame(columns=GAME_COLS)
    return pd.read_csv(GAMES_CSV, dtype={"game_id": str})[GAME_COLS]


def upsert_games(games: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new.empty:
        return games
    keep = games[~games.game_id.isin(set(new.game_id))]
    out = pd.concat([keep, new], ignore_index=True)
    return out.sort_values(["date", "game_id"]).reset_index(drop=True)


def ingest_bigdataball_dir(games: pd.DataFrame) -> pd.DataFrame:
    if not os.path.isdir(BDB_DIR):
        print("No data/bigdataball folder; using stored games only.")
        return games
    files = [f for f in sorted(os.listdir(BDB_DIR))
             if f.lower().endswith((".xlsx", ".xlsm"))]
    if not files:
        print("No BigDataBall file in data/bigdataball/; using stored games only.")
    for fn in files:
        try:
            new = parse_bigdataball_xlsx(os.path.join(BDB_DIR, fn))
        except Exception as e:  # noqa: BLE001
            print(f"!! ERROR reading {fn}: {e}", file=sys.stderr)
            continue
        before = len(games)
        games = upsert_games(games, new)
        last = new.date.max() if len(new) else "n/a"
        print(f"BigDataBall {fn}: {len(new)} games (latest {last}); "
              f"games table {before} -> {len(games)}")
    return games


# ----------------------------------------------------------------------------
# ESPN: tonight's lines (not stored; BigDataBall replaces them once played)
# ----------------------------------------------------------------------------
def http_get_json(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _espn_spread_home(odds: dict, home: str, away: str):
    details = (odds.get("details") or "").strip()
    if details.upper() in ("EVEN", "PK", "PICK", "PICK'EM"):
        return 0.0
    m = re.match(r"^([A-Za-z]+)\s+([+-]?\d+(\.\d+)?)$", details)
    if m:
        fav, n = team_abbr(m.group(1)), -abs(float(m.group(2)))
        if fav == home:
            return n
        if fav == away:
            return -n
    if odds.get("spread") is not None:
        try:
            return float(odds["spread"])
        except (TypeError, ValueError):
            pass
    return None


def fetch_tonight(day: dt.date, games: pd.DataFrame) -> pd.DataFrame:
    """Tonight's not-yet-played games that have a posted spread and total."""
    try:
        js = http_get_json(SCOREBOARD_URL.format(date=day.strftime("%Y%m%d")))
    except Exception as e:  # noqa: BLE001
        print(f"!! ESPN fetch failed ({e}); continuing without tonight's lines.")
        return pd.DataFrame(columns=GAME_COLS)

    last_played = {}
    done = games.dropna(subset=["home_score"])
    for col in ("home", "away"):
        for t, d in done.groupby(col)["date"].max().items():
            last_played[t] = max(last_played.get(t, ""), d)
    yesterday = (day - dt.timedelta(days=1)).isoformat()

    rows = []
    for ev in js.get("events", []):
        comp = (ev.get("competitions") or [{}])[0]
        if (ev.get("status") or {}).get("type", {}).get("completed"):
            continue
        sides = comp.get("competitors") or []
        h = next((s for s in sides if s.get("homeAway") == "home"), None)
        a = next((s for s in sides if s.get("homeAway") == "away"), None)
        if not h or not a:
            continue
        home = team_abbr(h["team"].get("abbreviation"))
        away = team_abbr(a["team"].get("abbreviation"))
        if home is None or away is None:
            continue  # preseason vs. international club, All-Star, etc.
        spread = total = None
        for odds in comp.get("odds") or []:
            if spread is None:
                spread = _espn_spread_home(odds, home, away)
            if total is None and odds.get("overUnder") is not None:
                try:
                    total = float(odds["overUnder"])
                except (TypeError, ValueError):
                    pass
        if spread is None or total is None:
            continue
        rows.append({
            "game_id": f"espn_{ev.get('id')}", "date": day.isoformat(),
            "season": season_of(day.isoformat()), "dataset": "upcoming",
            "home": home, "away": away,
            "neutral": int(bool(comp.get("neutralSite"))),
            "home_score": np.nan, "away_score": np.nan,
            "spread_home": spread, "total": total,
            "home_b2b": int(last_played.get(home) == yesterday),
            "away_b2b": int(last_played.get(away) == yesterday),
        })
    print(f"ESPN: {len(rows)} of tonight's games have lines.")
    return pd.DataFrame(rows, columns=GAME_COLS)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
N_T = len(ABBRS)
TIDX = {a: i for i, a in enumerate(ABBRS)}
N_P = 3 + 2 * N_T   # mu, hca, b2b, offense x30, defense x30


def _elapsed(g: pd.DataFrame, cur_season: int, gap: float):
    """Elapsed games for the home and away side of every row in g (sorted)."""
    n = len(g)
    teams = np.concatenate([g.home.map(TIDX).to_numpy(), g.away.map(TIDX).to_numpy()])
    order = np.concatenate([np.arange(n), np.arange(n)])
    idx = np.lexsort((order, teams))
    t_sorted = teams[idx]
    starts = np.r_[0, np.flatnonzero(np.diff(t_sorted)) + 1]
    counts = np.diff(np.r_[starts, len(t_sorted)])
    grp = np.repeat(np.arange(len(starts)), counts)
    pos = np.arange(len(t_sorted)) - starts[grp]
    el_sorted = counts[grp] - 1 - pos
    el = np.empty(2 * n)
    el[idx] = el_sorted
    seasons = np.concatenate([g.season.to_numpy(), g.season.to_numpy()]).astype(float)
    el = el + gap * (cur_season - seasons)
    return el[:n], el[n:]


def fit(g: pd.DataFrame, cur_season: int, c_line=None, c_result=None, gap=None):
    """Weighted least squares. g: games sorted by date (completed + tonight's).
    Returns parameter vector [mu, hca, b2b, o..., d...]."""
    c_line = C_LINE if c_line is None else c_line
    c_result = C_RESULT if c_result is None else c_result
    gap = OFFSEASON_GAP if gap is None else gap

    n = len(g)
    hi = g.home.map(TIDX).to_numpy()
    ai = g.away.map(TIDX).to_numpy()
    s = np.where(g.neutral.to_numpy() == 1, 0.0, 0.5)
    hb = g.home_b2b.to_numpy(dtype=float)
    ab = g.away_b2b.to_numpy(dtype=float)
    el_h, el_a = _elapsed(g, cur_season, gap)

    def side_rows(o_i, d_i, sgn_hca, own_b2b, opp_b2b):
        X = np.zeros((n, N_P))
        X[:, 0] = 1.0
        X[:, 1] = sgn_hca * s
        X[:, 2] = -0.5 * own_b2b + 0.5 * opp_b2b
        X[np.arange(n), 3 + o_i] = 1.0
        X[np.arange(n), 3 + N_T + d_i] = -1.0
        return X

    Xh = side_rows(hi, ai, 1.0, hb, ab)
    Xa = side_rows(ai, hi, -1.0, ab, hb)

    sp = g.spread_home.to_numpy(dtype=float)
    tot = g.total.to_numpy(dtype=float)
    has_line = ~(np.isnan(sp) | np.isnan(tot))
    exp_h = (tot - sp) / 2.0
    exp_a = (tot + sp) / 2.0

    hs = g.home_score.to_numpy(dtype=float)
    as_ = g.away_score.to_numpy(dtype=float)
    has_res = ~(np.isnan(hs) | np.isnan(as_))

    Xs, ys, ws = [], [], []
    m = has_line
    Xs += [Xh[m], Xa[m]]
    ys += [exp_h[m], exp_a[m]]
    ws += [1.0 / (el_h[m] + c_line), 1.0 / (el_a[m] + c_line)]
    if c_result is not None and c_result > 0:
        m = has_res
        Xs += [Xh[m], Xa[m]]
        ys += [hs[m], as_[m]]
        ws += [1.0 / (el_h[m] + c_result), 1.0 / (el_a[m] + c_result)]
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)

    # Solve the weighted normal equations with sum(o)=0, sum(d)=0 constraints.
    Xw = X * w[:, None]
    A = Xw.T @ X
    b = Xw.T @ y
    A += np.eye(N_P) * 1e-9
    C = np.zeros((2, N_P))
    C[0, 3:3 + N_T] = 1.0
    C[1, 3 + N_T:] = 1.0
    K = np.block([[A, C.T], [C, np.zeros((2, 2))]])
    sol = np.linalg.solve(K, np.concatenate([b, [0.0, 0.0]]))
    return sol[:N_P]


def predict_spread_home(beta, home, away, neutral, home_b2b, away_b2b):
    o, d = beta[3:3 + N_T], beta[3 + N_T:]
    hi = np.array([TIDX[t] for t in home])
    ai = np.array([TIDX[t] for t in away])
    margin = ((o[hi] + d[hi]) - (o[ai] + d[ai])
              + beta[1] * np.where(np.asarray(neutral) == 1, 0.0, 1.0)
              - beta[2] * np.asarray(home_b2b) + beta[2] * np.asarray(away_b2b))
    return -margin


def model_window(games: pd.DataFrame, asof: str) -> pd.DataFrame:
    """Games the model sees as of `asof`: this season and last season, played
    before asof (rows dated asof or later are dropped; add tonight's lines
    separately)."""
    cur = season_of(asof)
    g = games[(games.date < asof) & (games.season >= cur - 1)]
    return g


def build_rankings(games: pd.DataFrame, asof: str, tonight: pd.DataFrame | None = None):
    cur = season_of(asof)
    g = model_window(games, asof)
    if tonight is not None and len(tonight):
        g = pd.concat([g, tonight], ignore_index=True)
    g = g.sort_values(["date", "game_id"]).reset_index(drop=True)
    beta = fit(g, cur)
    mu, hca, b2b = beta[0], beta[1], beta[2]
    o, d = beta[3:3 + N_T], beta[3 + N_T:]

    this = games[(games.season == cur) & (games.date < asof)].dropna(subset=["home_score"])
    reg = this[~this.dataset.str.contains("Playoff|Play-In", case=False, na=False)]
    wins = {t: 0 for t in ABBRS}
    losses = {t: 0 for t in ABBRS}
    played = {t: 0 for t in ABBRS}
    for r in reg.itertuples():
        win, lose = (r.home, r.away) if r.home_score > r.away_score else (r.away, r.home)
        wins[win] += 1
        losses[lose] += 1
        played[r.home] += 1
        played[r.away] += 1
    # SRS: average margin adjusted for opponents (least squares, mean zero);
    # shown once a team has played 5 games.
    srs = {t: np.nan for t in ABBRS}
    if len(reg):
        A = np.zeros((len(reg) + 1, N_T))
        A[np.arange(len(reg)), reg.home.map(TIDX).to_numpy()] = 1.0
        A[np.arange(len(reg)), reg.away.map(TIDX).to_numpy()] = -1.0
        A[-1, :] = 1.0
        yv = np.r_[(reg.home_score - reg.away_score).to_numpy(dtype=float), 0.0]
        sol = np.linalg.lstsq(A, yv, rcond=None)[0]
        srs = {t: (sol[TIDX[t]] if played[t] >= 5 else np.nan) for t in ABBRS}

    out = pd.DataFrame({
        "team": ABBRS, "gpf": o + d, "ogpf": o, "dgpf": d, "gou": 2 * mu + o - d,
        "srs": [srs[t] for t in ABBRS],
        "w": [wins[t] for t in ABBRS], "l": [losses[t] for t in ABBRS],
    }).sort_values("gpf", ascending=False).reset_index(drop=True)
    out["rank"] = out.index + 1
    for col in ("ogpf", "dgpf", "gou", "srs"):
        out[f"{col}_rank"] = out[col].rank(ascending=False, method="min")
    info = {
        "mu": mu, "hca": hca, "b2b": b2b,
        "lines_this_season": int(this.dropna(subset=["spread_home"]).shape[0]),
        "games_this_season": int(len(this)),
        "through": this.date.max() if len(this) else None,
        "tonight": 0 if tonight is None else int(len(tonight)),
    }
    return out, info


# ----------------------------------------------------------------------------
# History + page
# ----------------------------------------------------------------------------
def append_history(rk: pd.DataFrame, asof: str) -> pd.DataFrame:
    cols = ["date", "team", "rank", "gpf", "ogpf", "dgpf", "gou"]
    hist = (pd.read_csv(HISTORY_CSV) if os.path.exists(HISTORY_CSV)
            else pd.DataFrame(columns=cols))
    hist = hist[hist.date != asof]
    snap = rk[["team", "rank", "gpf", "ogpf", "dgpf", "gou"]].copy()
    snap.insert(0, "date", asof)
    hist = pd.concat([hist, snap], ignore_index=True).sort_values(["date", "rank"])
    hist.to_csv(HISTORY_CSV, index=False)
    return hist


def sparkline(vals, w=84, h=26) -> str:
    vals = list(vals)[-SPARK_N:]
    if len(vals) < 2:
        return '<span class="nospark">&ndash;</span>'
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    pts = [f"{3 + i * (w - 6) / (len(vals) - 1):.1f},"
           f"{h - 4 - (v - lo) / rng * (h - 8):.1f}" for i, v in enumerate(vals)]
    lx, ly = pts[-1].split(",")
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'aria-hidden="true"><polyline points="{" ".join(pts)}"/>'
            f'<circle cx="{lx}" cy="{ly}" r="2.2"/></svg>')


def last_week_rank(hist, team, asof):
    target = (dt.date.fromisoformat(asof) - dt.timedelta(days=7)).isoformat()
    old = hist[(hist.team == team) & (hist.date <= target)]
    return int(old.sort_values("date").iloc[-1]["rank"]) if len(old) else None


def fmt_rank(v):
    return "" if pd.isna(v) else f"({int(v)})"


def render_html(rk, hist, asof, info) -> str:
    rows = []
    for r in rk.itertuples():
        lw = last_week_rank(hist, r.team, asof)
        move = ""
        if lw is not None and lw != r.rank:
            up = lw > r.rank
            move = (f'<span class="mv {"up" if up else "dn"}">'
                    f'{"▲" if up else "▼"}{abs(lw - r.rank)}</span>')
        series = hist[hist.team == r.team].sort_values("date")["gpf"]
        srs = "&ndash;" if pd.isna(r.srs) else f"{r.srs:+.1f} <span class=sub>{fmt_rank(r.srs_rank)}</span>"
        rows.append(f"""
      <tr>
        <td class="rk">{r.rank}</td>
        <td class="tm"><span class="ab">{r.team}</span><span class="nm">{TEAM_NAMES[r.team]}</span></td>
        <td class="lw">{"&ndash;" if lw is None else lw} {move}</td>
        <td class="num main">{r.gpf:+.1f}</td>
        <td class="sp">{sparkline(series)}</td>
        <td class="num">{r.ogpf:+.1f} <span class="sub">{fmt_rank(r.ogpf_rank)}</span></td>
        <td class="num">{r.dgpf:+.1f} <span class="sub">{fmt_rank(r.dgpf_rank)}</span></td>
        <td class="num">{r.gou:.1f} <span class="sub">{fmt_rank(r.gou_rank)}</span></td>
        <td class="num">{srs}</td>
        <td class="num">{r.w}-{r.l}</td>
      </tr>""")

    asof_txt = dt.date.fromisoformat(asof).strftime("%B %-d, %Y")
    if info["through"]:
        thru = dt.date.fromisoformat(info["through"]).strftime("%b %-d")
        thru_txt = f"results through {thru} ({info['games_this_season']} games this season)"
    else:
        thru_txt = "no games played yet this season"
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NBA Betting Market Rankings</title>
<style>
  :root {{ --ink:#14212e; --sub:#6b7a89; --line:#dfe6ec; --acc:#c8102e;
           --bg:#f7f9fb; --card:#ffffff; --hover:#fff4f5; --up:#1a7f37; --dn:#c1341f; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink:#e6edf3; --sub:#8b98a5; --line:#2a3440; --acc:#ff5a6e;
             --bg:#0e141b; --card:#151d26; --hover:#1d2733; --up:#3fb950; --dn:#f85149; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.45 "Helvetica Neue", Arial, sans-serif; }}
  .wrap {{ max-width:1000px; margin:0 auto; padding:28px 16px 60px; }}
  h1 {{ font-size:26px; margin:0 0 2px; letter-spacing:-.01em; }}
  .asof {{ color:var(--sub); margin:0 0 18px; }}
  .scroll {{ overflow-x:auto; border:1px solid var(--line); border-radius:10px; background:var(--card); }}
  table {{ width:100%; border-collapse:collapse; }}
  th {{ text-align:right; font-size:12px; text-transform:uppercase; letter-spacing:.06em;
        color:var(--sub); padding:10px; border-bottom:2px solid var(--line); }}
  th.l {{ text-align:left; }}
  td {{ padding:7px 10px; border-bottom:1px solid var(--line); text-align:right; white-space:nowrap; }}
  tr:last-child td {{ border-bottom:none; }}
  tr:hover td {{ background:var(--hover); }}
  .rk {{ color:var(--sub); width:34px; }}
  .tm {{ text-align:left; }}
  .ab {{ font-weight:700; display:inline-block; width:40px; }}
  .nm {{ color:var(--sub); font-size:13px; }}
  .lw {{ color:var(--sub); }}
  .mv {{ font-size:12px; }} .mv.up {{ color:var(--up); }} .mv.dn {{ color:var(--dn); }}
  .num.main {{ font-weight:700; font-size:16px; }}
  .sub {{ color:var(--sub); font-size:11px; }}
  .sp {{ padding:2px 6px; }}
  .spark {{ display:block; }} .spark polyline {{ fill:none; stroke:var(--acc); stroke-width:1.6; }}
  .spark circle {{ fill:var(--acc); }} .nospark {{ color:var(--sub); }}
  .foot {{ color:var(--sub); font-size:13px; margin-top:16px; }}
  .foot b {{ color:var(--ink); }} .foot a {{ color:var(--sub); }}
  @media (max-width:720px) {{ .nm, .sp {{ display:none; }} }}
</style></head>
<body><div class="wrap">
  <h1>NBA Betting Market Rankings</h1>
  <p class="asof">As of {asof_txt} &middot; {thru_txt} &middot; implied by closing point spreads and totals</p>
  <div class="scroll"><table>
    <thead><tr>
      <th>Rk</th><th class="l">Team</th><th>LstWk</th><th>GPF</th><th class="sp">Trend</th>
      <th>oGPF</th><th>dGPF</th><th>GOU</th><th>SRS</th><th>W-L</th>
    </tr></thead>
    <tbody>{''.join(rows)}
    </tbody>
  </table></div>
  <p class="foot">
    <b>GPF</b>: points favored vs. a league-average team on a neutral court.
    <b>oGPF / dGPF</b>: offensive and defensive parts of GPF (dGPF &gt; 0 = good defense).
    <b>GOU</b>: expected over/under vs. a league-average team.
    <b>SRS</b>: this season's scoring margin adjusted for opponents.
    Built from {info['lines_this_season']} of this season's closing lines
    plus {info['tonight']} of tonight's lines, with last season fading out as a prior.
    Market home court: <b>{info['hca']:.2f}</b> pts. Back-to-back penalty: <b>{info['b2b']:.2f}</b> pts.
    League-average team score: <b>{info['mu']:.1f}</b>.
    Method per <a href="https://www.inpredictable.com/2012/11/nba-team-rankings-now-available.html">inpredictable</a>.
  </p>
</div></body></html>"""


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true", help="skip ESPN tonight's lines")
    ap.add_argument("--asof", help="rebuild as of YYYY-MM-DD (testing)")
    args = ap.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    games = ingest_bigdataball_dir(load_games())
    games.to_csv(GAMES_CSV, index=False)

    today = dt.date.fromisoformat(args.asof) if args.asof else dt.datetime.now(EASTERN).date()
    asof = today.isoformat()
    tonight = None if args.no_fetch else fetch_tonight(today, games)

    rk, info = build_rankings(games, asof, tonight)
    hist = append_history(rk, asof)
    with open(os.path.join(OUT_DIR, "index.html"), "w") as f:
        f.write(render_html(rk, hist, asof, info))
    rk.to_csv(os.path.join(OUT_DIR, "rankings.csv"), index=False)

    print(rk[["rank", "team", "gpf", "ogpf", "dgpf", "gou", "w", "l"]].round(2).to_string(index=False))
    print(f"\nhca={info['hca']:.2f}  b2b={info['b2b']:.2f}  mu={info['mu']:.1f}  "
          f"lines this season={info['lines_this_season']}  tonight={info['tonight']}")


if __name__ == "__main__":
    main()
