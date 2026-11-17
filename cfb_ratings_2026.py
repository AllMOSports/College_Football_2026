"""
College Football Ratings 2026  (v3)
-----------------------------------
Offense/defense rating engine fed by ESPN's college-football scoreboard feed.

What changed from v2 (football_ratings_2025.py engine):
  1. Real shrinkage. REGULARIZATION_K now acts as K "pseudo-games" played at
     each team's prior rating, so a team with few games is pulled toward its
     prior instead of being fully fit to them. (In v2, K only slowed the fit
     down -- the final ratings were the same whatever K was.)
  2. Home-field advantage is learned from the data (neutral-site games are
     excluded from it) instead of being ignored.
  3. The fit runs until the ratings stop moving (TOLERANCE), not a fixed
     1,000 iterations, and reports whether it converged.
  4. FCS-vs-FCS games are included (ESPN group 81), so FCS teams are rated on
     a full schedule instead of one game.
  5. Optional recency weighting (RECENCY_HALF_LIFE_DAYS) and an optional
     preseason prior (PRIOR_RATINGS_PATH, e.g. last season's output).
  6. Any failed ESPN pull stops the run with an error, so a run can't
     silently publish ratings that are missing a week.
  7. DraftKings (ESPN's listed provider) spreads/totals are read from the
     feed. Upcoming games get model-vs-market picks with a confidence tier,
     each pick is logged BEFORE kickoff, and logged picks are graded once the
     game is final, giving a running record by confidence tier.
  8. Output adds ESPN team id + abbreviation, W-L record, strength of
     schedule, and the learned home-field value. Bowls/CFP are included.
"""

import requests
import json
import csv
import os
import math
import statistics
import pandas as pd
from datetime import datetime, date
import time

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SEASON_YEAR   = 2026
SEASON_TYPE   = 2            # 2 = regular season
WEEKS         = list(range(1, 16))   # weeks not played yet are harmless
INCLUDE_POSTSEASON = True    # also pull bowls/CFP (seasontype=3, week=1)


def _weeks_from_env(default):
    """
    Optional override from the CFB_WEEKS environment variable (used by the
    GitHub Actions workflow). Accepts "1", "1,2,3", or a range like "1-15".
    """
    raw = os.environ.get("CFB_WEEKS", "").strip()
    if not raw:
        return default
    weeks = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            weeks.extend(range(int(lo), int(hi) + 1))
        elif part:
            weeks.append(int(part))
    return sorted(set(weeks))


WEEKS = _weeks_from_env(WEEKS)

# ESPN "groups": 80 = FBS (includes FBS-vs-FCS games), 81 = FCS.
GROUPS        = [80, 81]

SCOREBOARD_URL = ("https://site.api.espn.com/apis/site/v2/sports/football/"
                  "college-football/scoreboard"
                  "?year={year}&seasontype={stype}&week={week}"
                  "&groups={group}&limit=500")

FAIL_ON_FETCH_ERROR = True   # stop the run if any ESPN pull fails

MAX_POINTS    = 150
OUTPUT_PATH   = f"cfb_ratings_{SEASON_YEAR}.json"
CSV_PATH      = f"cfb_scoreboard_{SEASON_YEAR}.csv"
PICKS_JSON    = f"cfb_picks_{SEASON_YEAR}.json"
PICKS_CSV     = f"cfb_picks_{SEASON_YEAR}.csv"
LOG_CSV       = f"cfb_predictions_log_{SEASON_YEAR}.csv"   # append-only history
BACKTEST_JSON = f"cfb_backtest_{SEASON_YEAR}.json"
RAW_CACHE_DIR = "espn_raw"

# --- Rating engine (v3) ---
LEARNING_RATE         = 0.1
MAX_ITERATIONS        = 6000
TOLERANCE             = 1e-4  # stop when no rating moves more than this
COMPETITIVE_THRESHOLD = 40    # "half-weight" point of the competitiveness curve
REGULARIZATION_K      = 3.0   # pseudo-games at the shrinkage target (real shrinkage)
SHRINK_TOWARD_CONFERENCE = True  # target = conference average, not 0
                                 # (shrinking everyone toward 0 squeezes FBS and
                                 # FCS together and inflates home-field advantage)
MOV_CAP               = 28    # max points of error one game can contribute
HOME_FIELD_START      = 2.5   # starting guess; the fit learns the real value
LEARN_HOME_FIELD      = True  # False = keep HOME_FIELD_START fixed

# Recency: a game's weight halves every RECENCY_HALF_LIFE_DAYS before the
# most recent game in the data. None turns recency weighting off.
RECENCY_HALF_LIFE_DAYS = 70

# Preseason prior: a ratings JSON in this script's own output format (e.g.
# last season's cfb_ratings_2025.json). Teams are pulled toward
# PRIOR_REGRESSION x their old rating instead of toward 0. Missing file = no
# prior (everyone starts at 0).
PRIOR_RATINGS_PATH = f"cfb_ratings_{SEASON_YEAR - 1}.json"
PRIOR_REGRESSION   = 0.5

# Confidence tiers: size of the model-vs-market gap, in points.
CONFIDENCE_TIERS = [(5.0, "High"), (3.0, "Medium"), (1.5, "Low")]  # else "Pass"

# ESPN conferenceId -> conference name for FBS. Anything not in this map is
# labelled NON_FBS_LABEL. Unknown IDs are reported at the end of each run.
FBS_CONFERENCES = {
    "1":   "ACC",
    "4":   "Big 12",
    "5":   "Big Ten",
    "8":   "SEC",
    "9":   "Pac-12",
    "12":  "Conference USA",
    "15":  "MAC",
    "17":  "Mountain West",
    "18":  "FBS Independents",
    "37":  "Sun Belt",
    "151": "American",
}
NON_FBS_LABEL = "FCS / Non-FBS"

# ---------------------------------------------------------------------------
# MANUAL GAMES / CORRECTIONS / EXCLUSIONS
# ---------------------------------------------------------------------------
# Team names must match the names this script prints (ESPN "location" names,
# e.g. "Ohio State", "Miami", "Miami (OH)").
# Format: ("YYYY-MM-DD", "Home Team", home_score, "Away Team", away_score)
#     or  ("YYYY-MM-DD", "Team 1", score1, "Team 2", score2, True)  for neutral
MANUAL_GAMES = [
]

# Format: ("YYYY-MM-DD", "Team A", correct_score_A, "Team B", correct_score_B)
SCORE_CORRECTIONS = [
]

# Format: ("YYYY-MM-DD", "Team A", "Team B")
EXCLUDED_GAMES = [
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer": "https://www.espn.com/",
}

# ---------------------------------------------------------------------------
# HTTP SESSION
# ---------------------------------------------------------------------------

def build_session():
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        from requests.packages.urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=2, connect=2, read=2, backoff_factor=1.5,
                  status_forcelist=[429, 500, 502, 503, 504],
                  raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry, pool_connections=1, pool_maxsize=1)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# ---------------------------------------------------------------------------
# PARSING
# ---------------------------------------------------------------------------

def parse_score(value):
    try:
        score = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return score if 0 <= score <= MAX_POINTS else None


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_odds(comp, home_abbr, away_abbr):
    """
    Read the sportsbook line ESPN attaches to a game.
    Returns {"provider", "home_line", "total"} where home_line is the home
    team's spread (negative = home favored), or None if there is no line.
    """
    odds_list = comp.get("odds") or []
    if not odds_list:
        return None
    o = odds_list[0]
    provider = ((o.get("provider") or {}).get("name")) or "unknown"
    total = _to_float(o.get("overUnder"))
    home_line = None

    details = (o.get("details") or "").strip()
    if details.upper() in ("EVEN", "PK", "PICK", "PICK'EM"):
        home_line = 0.0
    elif details:
        parts = details.rsplit(" ", 1)
        num = _to_float(parts[-1]) if len(parts) == 2 else None
        if num is not None:
            abbr = parts[0].strip().upper()
            if abbr == (home_abbr or "").upper():
                home_line = num
            elif abbr == (away_abbr or "").upper():
                home_line = -num

    if home_line is None:
        # Fallback: numeric spread + favorite flag
        spread = _to_float(o.get("spread"))
        if spread is not None:
            home_fav = (o.get("homeTeamOdds") or {}).get("favorite")
            away_fav = (o.get("awayTeamOdds") or {}).get("favorite")
            if home_fav:
                home_line = -abs(spread)
            elif away_fav:
                home_line = abs(spread)

    if home_line is None and total is None:
        return None
    return {"provider": provider, "home_line": home_line, "total": total}


def parse_scoreboard(data, team_info):
    """
    Turn one ESPN scoreboard payload into:
      finals:   (event_id, date, home_id, home_score, away_id, away_score, neutral)
      upcoming: (event_id, date, home_id, away_id, neutral, odds)  -- not started
      lines:    {event_id: odds}  -- market line for every event that has one
    Also fills team_info[team_id].
    """
    finals, upcoming, lines = [], [], {}
    skipped = {"in_progress_or_other": 0, "bad_score": 0, "malformed": 0}

    for event in data.get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            skipped["malformed"] += 1
            continue
        comp = comps[0]
        eid = str(event.get("id"))
        status = (comp.get("status") or event.get("status") or {}).get("type", {})
        state = status.get("state")

        competitors = comp.get("competitors") or []
        if len(competitors) != 2:
            skipped["malformed"] += 1
            continue

        sides = {}
        for c in competitors:
            team = c.get("team") or {}
            tid = str(team.get("id") or c.get("id") or "")
            if not tid:
                break
            team_info.setdefault(tid, {
                "name":          team.get("location") or team.get("displayName"),
                "full_name":     team.get("displayName"),
                "abbreviation":  team.get("abbreviation"),
                "conference_id": str(team.get("conferenceId", "")),
            })
            sides[c.get("homeAway")] = (tid, c.get("score"),
                                        team.get("abbreviation"))
        if set(sides) != {"home", "away"}:
            skipped["malformed"] += 1
            continue

        (h_id, h_raw, h_abbr), (a_id, a_raw, a_abbr) = sides["home"], sides["away"]
        neutral = bool(comp.get("neutralSite", False))
        game_date = (event.get("date") or comp.get("date") or "")[:10]

        odds = parse_odds(comp, h_abbr, a_abbr)
        if odds:
            lines[eid] = odds

        if state == "pre":
            upcoming.append((eid, game_date, h_id, a_id, neutral, odds))
            continue

        if (not status.get("completed", False)
                or status.get("name", "STATUS_FINAL") != "STATUS_FINAL"):
            skipped["in_progress_or_other"] += 1
            continue

        h_score, a_score = parse_score(h_raw), parse_score(a_raw)
        if h_score is None or a_score is None:
            skipped["bad_score"] += 1
            continue
        finals.append((eid, game_date, h_id, h_score, a_id, a_score, neutral))

    return finals, upcoming, lines, skipped


def make_unique_names(team_info):
    """ESPN short name, falling back to full name / team id on collisions."""
    by_name = {}
    for tid, info in team_info.items():
        by_name.setdefault(info["name"], []).append(tid)
    names = {}
    for name, ids in by_name.items():
        for tid in ids:
            names[tid] = name if len(ids) == 1 else team_info[tid]["full_name"]
    counts = {}
    for n in names.values():
        counts[n] = counts.get(n, 0) + 1
    for tid, n in names.items():
        if counts[n] > 1:
            names[tid] = f"{n} [{tid}]"
    return names

# ---------------------------------------------------------------------------
# SCRAPING
# ---------------------------------------------------------------------------

def fetch_json(session, url):
    resp = session.get(url, timeout=(10, 25), headers=HEADERS)
    resp.raise_for_status()
    return resp.json()


def scrape_season():
    os.makedirs(RAW_CACHE_DIR, exist_ok=True)
    session   = build_session()
    team_info = {}
    finals, upcoming, lines, failed = [], [], {}, []

    pulls = [(SEASON_TYPE, w) for w in WEEKS]
    if INCLUDE_POSTSEASON:
        pulls.append((3, 1))

    t0 = time.perf_counter()
    for stype, week in pulls:
        for group in GROUPS:
            url = SCOREBOARD_URL.format(year=SEASON_YEAR, stype=stype,
                                        week=week, group=group)
            label = f"seasontype {stype} week {week} group {group}"
            print(f"  Fetching {label}...", end=" ", flush=True)
            try:
                data = fetch_json(session, url)
            except (requests.RequestException, ValueError) as e:
                print(f"FAILED ({e})")
                failed.append(label)
                continue

            with open(os.path.join(RAW_CACHE_DIR,
                      f"s{stype}_w{week}_g{group}.json"), "w") as f:
                json.dump(data, f)

            f_, u_, l_, skipped = parse_scoreboard(data, team_info)
            finals.extend(f_)
            upcoming.extend(u_)
            lines.update(l_)
            print(f"{len(f_)} final, {len(u_)} upcoming "
                  f"(skipped: {skipped['in_progress_or_other']} live/other, "
                  f"{skipped['bad_score']} bad score, "
                  f"{skipped['malformed']} malformed)")
            time.sleep(0.5)

    print(f"\n  [TIMING] Fetching took {time.perf_counter() - t0:.1f}s.")
    if failed:
        print(f"  *** {len(failed)} pull(s) failed: {failed}")
        if FAIL_ON_FETCH_ERROR:
            raise SystemExit("Stopping: ratings would be missing games. "
                             "Re-run, or set FAIL_ON_FETCH_ERROR = False.")

    # A game can appear in both group 80 and 81 -> dedup by ESPN event id
    def dedup(rows):
        seen, out = set(), []
        for r in rows:
            if r[0] not in seen:
                seen.add(r[0])
                out.append(r)
        return out
    finals, upcoming = dedup(finals), dedup(upcoming)

    names = make_unique_names(team_info)
    all_games = [(d, names[h], hs, names[a], as_, neu, eid)
                 for eid, d, h, hs, a, as_, neu in finals]
    upcoming = [(eid, d, names[h], names[a], neu, odds)
                for eid, d, h, a, neu, odds in upcoming]

    team_meta, unknown_conf_ids = {}, set()
    for tid, info in team_info.items():
        cid = info["conference_id"]
        conf = FBS_CONFERENCES.get(cid)
        if conf is None:
            conf = NON_FBS_LABEL
            if cid:
                unknown_conf_ids.add(cid)
        team_meta[names[tid]] = {"conference": conf, "team_id": tid,
                                 "conference_id": cid or None,
                                 "abbreviation": info.get("abbreviation")}

    return all_games, upcoming, lines, team_meta, unknown_conf_ids

# ---------------------------------------------------------------------------
# CORRECTIONS / EXCLUSIONS / DEDUP
# Game tuple: (date, home, home_score, away, away_score, neutral, event_id)
# ---------------------------------------------------------------------------

def normalize_manual(games):
    out = []
    for g in games:
        d, t1, s1, t2, s2 = g[:5]
        neutral = bool(g[5]) if len(g) > 5 else False
        out.append((d, t1, s1, t2, s2, neutral, f"manual-{d}-{t1}-{t2}"))
    return out


def apply_score_corrections(all_games, corrections=SCORE_CORRECTIONS):
    lookup = {(d, frozenset([a, b])): {a: sa, b: sb}
              for d, a, sa, b, sb in corrections}
    corrected, fixed = 0, []
    for d, t1, s1, t2, s2, neu, eid in all_games:
        fix = lookup.get((d, frozenset([t1, t2])))
        if fix:
            n1, n2 = fix.get(t1, s1), fix.get(t2, s2)
            corrected += (n1, n2) != (s1, s2)
            fixed.append((d, t1, n1, t2, n2, neu, eid))
        else:
            fixed.append((d, t1, s1, t2, s2, neu, eid))
    print(f"  Corrected {corrected} game score(s)." if corrected
          else "  No SCORE_CORRECTIONS matched (nothing changed).")
    return fixed


def apply_exclusions(all_games, exclusions=EXCLUDED_GAMES):
    keys = {(d, frozenset([a, b])) for d, a, b in exclusions}
    kept = [g for g in all_games if (g[0], frozenset([g[1], g[3]])) not in keys]
    removed = len(all_games) - len(kept)
    print(f"  Removed {removed} excluded game(s)." if removed
          else "  No EXCLUDED_GAMES matched (nothing removed).")
    return kept


def deduplicate_games(all_games):
    seen, unique = set(), []
    for g in all_games:
        key = (g[0], frozenset([g[1], g[3]]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(g)
    dupes = len(all_games) - len(unique)
    print(f"  Removed {dupes} duplicate game(s). {len(unique)} remain." if dupes
          else f"  No duplicates found. {len(unique)} games.")
    return unique


def save_csv(all_games):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Home Team", "Home Score", "Away Team",
                    "Away Score", "Neutral Site", "ESPN Event ID"])
        w.writerows(all_games)
    print(f"Saved {len(all_games)} games to {CSV_PATH}")

# ---------------------------------------------------------------------------
# PRIOR
# ---------------------------------------------------------------------------

def load_prior(path=PRIOR_RATINGS_PATH, regression=PRIOR_REGRESSION):
    """{team: (off_prior, def_prior)} from an earlier output file, regressed."""
    if not path or not os.path.exists(path):
        print(f"  No prior ratings file ({path}) -- every team starts at 0.")
        return {}
    with open(path) as f:
        data = json.load(f)
    prior = {t["school"]: (regression * t["off_rating"],
                           regression * t["def_rating"])
             for t in data.get("teams", [])}
    print(f"  Loaded prior ratings for {len(prior)} teams from {path} "
          f"(x{regression}).")
    return prior

# ---------------------------------------------------------------------------
# RATING ENGINE (v3)
# ---------------------------------------------------------------------------
# Predicted points for a team:
#     league_avg + own OFF - opponent DEF  (+/- home_field/2 unless neutral)
# Each pass nudges every rating toward the value that makes its weighted,
# capped errors balance against K pseudo-games at the team's prior:
#     sum(w * error) = K * (rating - prior)
# so a team with few (or low-weight) games stays close to its prior.

def competitiveness_weight(gap, scale=COMPETITIVE_THRESHOLD):
    return 1.0 / (1.0 + (gap / scale) ** 2)


def recency_weights(dates):
    if not RECENCY_HALF_LIFE_DAYS:
        return [1.0] * len(dates)
    parsed = []
    for d in dates:
        try:
            parsed.append(date.fromisoformat(d))
        except (TypeError, ValueError):
            parsed.append(None)
    valid = [p for p in parsed if p]
    if not valid:
        return [1.0] * len(dates)
    latest = max(valid)
    return [0.5 ** ((latest - p).days / RECENCY_HALF_LIFE_DAYS) if p else 1.0
            for p in parsed]


def calculate_ratings(all_games, prior=None, groups=None):
    """
    groups: {team: conference id}. With SHRINK_TOWARD_CONFERENCE, each team
    is pulled toward its conference's current average (plus its own
    preseason offset, if any) instead of toward 0. The conference averages
    themselves are not shrunk, so the gap between strong and weak
    conferences -- and between FBS and FCS -- is set by the games played.
    """
    prior = prior or {}
    groups = groups or {}
    games = [(t1, t2, s1, s2, neu) for _, t1, s1, t2, s2, neu, _ in all_games]
    rec_w = recency_weights([g[0] for g in all_games])
    teams = sorted({t for g in games for t in (g[0], g[1])})
    if not teams:
        return {}, {}, {}, 0, 0, False

    all_scores = [s for g in games for s in (g[2], g[3])]
    league_avg = sum(all_scores) / len(all_scores)
    print(f"  League average: {league_avg:.2f} points per team per game")

    p_off = {t: prior.get(t, (0.0, 0.0))[0] for t in teams}
    p_def = {t: prior.get(t, (0.0, 0.0))[1] for t in teams}
    off_rating = dict(p_off)
    def_rating = dict(p_def)
    hfa = HOME_FIELD_START

    use_groups = SHRINK_TOWARD_CONFERENCE and bool(groups)
    group_of = {t: (groups.get(t) or "none") for t in teams}
    members = {}
    for t in teams:
        members.setdefault(group_of[t], []).append(t)
    gmean = lambda d, g: sum(d[t] for t in members[g]) / len(members[g])
    # a team's preseason offset from its conference (0 with no prior file)
    p_off_dev = {t: p_off[t] - gmean(p_off, group_of[t]) for t in teams}
    p_def_dev = {t: p_def[t] - gmean(p_def, group_of[t]) for t in teams}
    if use_groups:
        print(f"  Shrinking toward conference averages ({len(members)} groups).")

    print(f"\n  Fitting ({len(teams)} teams, {len(games)} games; K={REGULARIZATION_K}, "
          f"scale={COMPETITIVE_THRESHOLD}, MOV cap={MOV_CAP}, "
          f"recency half-life={RECENCY_HALF_LIFE_DAYS} days)...")
    t0 = time.perf_counter()
    converged = False
    for it in range(1, MAX_ITERATIONS + 1):
        off_err = {t: 0.0 for t in teams}
        def_err = {t: 0.0 for t in teams}
        wsum    = {t: 0.0 for t in teams}
        hfa_num = hfa_den = 0.0

        for (h, a, sh, sa, neu), rw in zip(games, rec_w):
            gap = abs((off_rating[h] + def_rating[h]) -
                      (off_rating[a] + def_rating[a]))
            w = rw * competitiveness_weight(gap)
            half = 0.0 if neu else hfa / 2.0

            pred_h = league_avg + off_rating[h] - def_rating[a] + half
            pred_a = league_avg + off_rating[a] - def_rating[h] - half
            err_h = max(-MOV_CAP, min(MOV_CAP, sh - pred_h))
            err_a = max(-MOV_CAP, min(MOV_CAP, sa - pred_a))

            off_err[h] += w * err_h
            off_err[a] += w * err_a
            def_err[h] -= w * err_a
            def_err[a] -= w * err_h
            wsum[h] += w
            wsum[a] += w
            if not neu:
                hfa_num += w * (err_h - err_a)
                hfa_den += w

        if use_groups:
            g_off = {g: gmean(off_rating, g) for g in members}
            g_def = {g: gmean(def_rating, g) for g in members}
            tgt_off = {t: g_off[group_of[t]] + p_off_dev[t] for t in teams}
            tgt_def = {t: g_def[group_of[t]] + p_def_dev[t] for t in teams}
        else:
            tgt_off, tgt_def = p_off, p_def

        max_step = 0.0
        for t in teams:
            denom = wsum[t] + REGULARIZATION_K
            d_off = LEARNING_RATE * (off_err[t] - REGULARIZATION_K *
                                     (off_rating[t] - tgt_off[t])) / denom
            d_def = LEARNING_RATE * (def_err[t] - REGULARIZATION_K *
                                     (def_rating[t] - tgt_def[t])) / denom
            off_rating[t] += d_off
            def_rating[t] += d_def
            max_step = max(max_step, abs(d_off), abs(d_def))

        if LEARN_HOME_FIELD and hfa_den > 0:
            d_hfa = LEARNING_RATE * (hfa_num / hfa_den)
            hfa += d_hfa
            max_step = max(max_step, abs(d_hfa))

        if it % 500 == 0:
            print(f"  [Fit] iteration {it}: largest change {max_step:.5f}")
        if max_step < TOLERANCE:
            converged = True
            break

    print(f"  [TIMING] Fit took {time.perf_counter() - t0:.1f}s, "
          f"{it} iterations, {'converged' if converged else 'NOT converged'}.")
    print(f"  Learned home-field advantage: {hfa:.2f} points")
    if not converged:
        print("  *** Fit did not converge -- raise MAX_ITERATIONS.")

    ovr_rating = {t: off_rating[t] + def_rating[t] for t in teams}
    return off_rating, def_rating, ovr_rating, league_avg, hfa, converged


def predict(home, away, neutral, off, dfn, league_avg, hfa):
    """Projected (home_pts, away_pts) for a game."""
    half = 0.0 if neutral else hfa / 2.0
    ph = league_avg + off[home] - dfn[away] + half
    pa = league_avg + off[away] - dfn[home] - half
    return ph, pa

# ---------------------------------------------------------------------------
# RECORDS / STRENGTH OF SCHEDULE
# ---------------------------------------------------------------------------

def records_and_sos(all_games, ovr_rating):
    rec, opps = {}, {}
    for _, h, sh, a, sa, _, _ in all_games:
        for t, us, them, opp in ((h, sh, sa, a), (a, sa, sh, h)):
            r = rec.setdefault(t, [0, 0, 0])
            r[0 if us > them else 1 if us < them else 2] += 1
            opps.setdefault(t, []).append(opp)
    sos = {t: statistics.mean(ovr_rating[o] for o in os_) for t, os_ in opps.items()}
    record = {t: (f"{w}-{l}" + (f"-{ti}" if ti else "")) for t, (w, l, ti) in rec.items()}
    games = {t: sum(r) for t, r in rec.items()}
    return record, sos, games

# ---------------------------------------------------------------------------
# RATINGS OUTPUT
# ---------------------------------------------------------------------------

def build_team_entries(off, dfn, ovr, meta, record, sos, gp,
                       conf_filter=None, fbs_only=False):
    pool = list(ovr.keys())
    if conf_filter is not None:
        pool = [t for t in pool if meta.get(t, {}).get("conference") == conf_filter]
    if fbs_only:
        pool = [t for t in pool if meta.get(t, {}).get("conference") != NON_FBS_LABEL]

    rank = lambda d: {t: i + 1 for i, t in
                      enumerate(sorted(pool, key=lambda t: d[t], reverse=True))}
    off_rank, def_rank, sos_rank = rank(off), rank(dfn), rank(sos)
    ovr_sorted = sorted(pool, key=lambda t: ovr[t], reverse=True)

    return [
        {
            "ovr_rank":     i + 1,
            "school":       t,
            "team_id":      meta.get(t, {}).get("team_id"),
            "abbreviation": meta.get(t, {}).get("abbreviation"),
            "conference":   meta.get(t, {}).get("conference", NON_FBS_LABEL),
            "record":       record.get(t, "0-0"),
            "games":        gp.get(t, 0),
            "ovr_rating":   round(ovr[t], 2),
            "off_rating":   round(off[t], 2),
            "off_rank":     off_rank[t],
            "def_rating":   round(dfn[t], 2),
            "def_rank":     def_rank[t],
            "sos":          round(sos[t], 2),
            "sos_rank":     sos_rank[t],
        }
        for i, t in enumerate(ovr_sorted)
    ]


def write_json(path, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def header(league_avg, hfa, converged, extra=None):
    return {
        "last_updated":   datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "season":         SEASON_YEAR,
        "weeks":          WEEKS,
        "league_average": round(league_avg, 2),
        "home_field":     round(hfa, 2),
        "fit_converged":  converged,
        **(extra or {}),
    }


def save_ratings(off, dfn, ovr, league_avg, hfa, converged, meta, all_games):
    record, sos, gp = records_and_sos(all_games, ovr)
    args = (off, dfn, ovr, meta, record, sos, gp)

    entries = build_team_entries(*args)
    write_json(OUTPUT_PATH, {**header(league_avg, hfa, converged), "teams": entries})
    print(f"Saved {len(entries)} teams to {OUTPUT_PATH}")

    fbs = build_team_entries(*args, fbs_only=True)
    fbs_path = f"cfb_ratings_{SEASON_YEAR}_fbs.json"
    write_json(fbs_path, {**header(league_avg, hfa, converged, {"scope": "FBS"}),
                          "teams": fbs})
    print(f"Saved {len(fbs)} FBS teams to {fbs_path}")
    print("Top 10 FBS:")
    for e in fbs[:10]:
        print(f"  {e['ovr_rank']:>2}. {e['school']:<22} {e['record']:<6} "
              f"({e['conference']}) | OVR {e['ovr_rating']:+.2f} "
              f"| OFF {e['off_rating']:+.2f} | DEF {e['def_rating']:+.2f}")

    for conf in sorted({meta.get(t, {}).get("conference", NON_FBS_LABEL) for t in ovr}):
        ce = build_team_entries(*args, conf_filter=conf)
        slug = conf.lower().replace(" / ", "_").replace(" ", "_").replace("-", "")
        write_json(f"cfb_ratings_{SEASON_YEAR}_{slug}.json",
                   {**header(league_avg, hfa, converged, {"conference": conf}),
                    "teams": ce})
        print(f"  {conf}: {len(ce)} teams")

    df = pd.DataFrame([{
        "School": e["school"], "Conference": e["conference"],
        "Record": e["record"], "Games": e["games"],
        "OFF Rating": e["off_rating"], "DEF Rating": e["def_rating"],
        "OVR Rating": e["ovr_rating"], "OFF Rank": e["off_rank"],
        "DEF Rank": e["def_rank"], "OVR Rank": e["ovr_rank"],
        "SOS": e["sos"], "SOS Rank": e["sos_rank"],
        "ESPN Team ID": e["team_id"],
    } for e in entries])
    path = f"cfb_rankings_{SEASON_YEAR}_all.csv"
    df.to_csv(path, index=False)
    print(f"  Rankings CSV: {path}")

# ---------------------------------------------------------------------------
# PICKS, PREDICTION LOG, BACKTEST
# ---------------------------------------------------------------------------

def tier(gap):
    for threshold, label in CONFIDENCE_TIERS:
        if gap >= threshold:
            return label
    return "Pass"


def half(x):
    return round(x * 2) / 2


def fmt_line(team, line):
    if line == 0:
        return f"{team} PK"
    return f"{team} {'+' if line > 0 else ''}{line:g}"


def make_picks(upcoming, off, dfn, league_avg, hfa):
    """Model vs market for every not-yet-started game with both teams rated."""
    picks, unrated = [], 0
    for eid, d, home, away, neutral, odds in upcoming:
        if home not in off or away not in off:
            unrated += 1
            continue
        ph, pa = predict(home, away, neutral, off, dfn, league_avg, hfa)
        model_home_line = -(ph - pa)            # spread convention: neg = home fav
        p = {
            "event_id": eid, "date": d, "home": home, "away": away,
            "neutral": neutral,
            "model_home_pts": round(ph, 1), "model_away_pts": round(pa, 1),
            "model_home_line": round(model_home_line, 1),
            "model_total": round(ph + pa, 1),
            "provider": None, "market_home_line": None, "market_total": None,
            "spread_gap": None, "spread_lean": None, "spread_lean_side": None,
            "spread_conf": None,
            "total_gap": None, "total_lean": None, "total_conf": None,
        }
        if odds:
            p["provider"] = odds["provider"]
            ml, mt = odds["home_line"], odds["total"]
            if ml is not None:
                gap = model_home_line - ml        # neg = model likes home more
                p["market_home_line"] = ml
                p["spread_gap"] = round(abs(gap), 1)
                if gap < 0:
                    p["spread_lean_side"] = "home"
                    p["spread_lean"] = fmt_line(home, ml)
                else:
                    p["spread_lean_side"] = "away"
                    p["spread_lean"] = fmt_line(away, -ml)
                p["spread_conf"] = tier(abs(gap))
            if mt is not None:
                tg = (ph + pa) - mt
                p["market_total"] = mt
                p["total_gap"] = round(abs(tg), 1)
                p["total_lean"] = ("Over " if tg > 0 else "Under ") + f"{mt:g}"
                p["total_conf"] = tier(abs(tg))
        picks.append(p)
    if unrated:
        print(f"  {unrated} upcoming game(s) skipped: a team has no rated games yet.")
    picks.sort(key=lambda p: -(p["spread_gap"] or 0))
    return picks


def save_picks(picks, hfa):
    write_json(PICKS_JSON, {
        "last_updated": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "season": SEASON_YEAR, "home_field": round(hfa, 2),
        "confidence_tiers": {label: f">= {th} pts" for th, label in CONFIDENCE_TIERS},
        "games": picks,
    })
    rows = []
    for p in picks:
        mfav = p["home"] if p["model_home_line"] < 0 else p["away"]
        rows.append({
            "Date": p["date"],
            "Game": f"{p['away']} {'vs' if p['neutral'] else '@'} {p['home']}",
            "Market Spread": (fmt_line(p["home"], p["market_home_line"])
                              if p["market_home_line"] is not None else ""),
            "Model Spread": fmt_line(mfav, -abs(half(p["model_home_line"]))),
            "Spread Lean": p["spread_lean"] or "",
            "Spread Conf.": p["spread_conf"] or "",
            "Market O/U": p["market_total"] if p["market_total"] is not None else "",
            "Model O/U": half(p["model_total"]),
            "Total Lean": p["total_lean"] or "",
            "Total Conf.": p["total_conf"] or "",
            "Spread Gap": p["spread_gap"] if p["spread_gap"] is not None else "",
            "Total Gap": p["total_gap"] if p["total_gap"] is not None else "",
            "Provider": p["provider"] or "",
        })
    pd.DataFrame(rows).to_csv(PICKS_CSV, index=False)
    with_lines = sum(1 for p in picks if p["market_home_line"] is not None)
    print(f"Saved {len(picks)} upcoming games ({with_lines} with a market line) "
          f"to {PICKS_CSV} / {PICKS_JSON}")


LOG_FIELDS = ["event_id", "date", "home", "away", "neutral", "logged_at",
              "provider", "market_home_line", "market_total",
              "model_home_line", "model_total",
              "spread_lean_side", "spread_gap", "spread_conf",
              "total_lean", "total_gap", "total_conf",
              "home_score", "away_score", "spread_result", "total_result"]


def update_prediction_log(picks, all_games):
    """
    Append-only history. A pick is (re)written only while its game hasn't
    started, so the logged line is the last pre-kickoff one -- no hindsight.
    Finished games are graded against the market line that was logged.
    """
    log = {}
    if os.path.exists(LOG_CSV):
        with open(LOG_CSV, newline="") as f:
            for row in csv.DictReader(f):
                log[row["event_id"]] = row

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    for p in picks:
        if p["market_home_line"] is None and p["market_total"] is None:
            continue
        log[p["event_id"]] = {
            "event_id": p["event_id"], "date": p["date"], "home": p["home"],
            "away": p["away"], "neutral": p["neutral"], "logged_at": stamp,
            "provider": p["provider"],
            "market_home_line": p["market_home_line"], "market_total": p["market_total"],
            "model_home_line": p["model_home_line"], "model_total": p["model_total"],
            "spread_lean_side": p["spread_lean_side"], "spread_gap": p["spread_gap"],
            "spread_conf": p["spread_conf"],
            "total_lean": p["total_lean"], "total_gap": p["total_gap"],
            "total_conf": p["total_conf"],
            "home_score": "", "away_score": "", "spread_result": "", "total_result": "",
        }

    finals = {g[6]: g for g in all_games}
    graded = 0
    for eid, row in log.items():
        g = finals.get(eid)
        if not g or row.get("spread_result") or row.get("total_result"):
            continue
        _, _, sh, _, sa, _, _ = g
        row["home_score"], row["away_score"] = sh, sa
        ml = _to_float(row.get("market_home_line"))
        if ml is not None and row.get("spread_lean_side"):
            cover = (sh - sa) + ml            # > 0: home covered
            if cover == 0:
                row["spread_result"] = "push"
            else:
                home_won_ats = cover > 0
                row["spread_result"] = ("win" if home_won_ats ==
                                        (row["spread_lean_side"] == "home") else "loss")
        mt = _to_float(row.get("market_total"))
        if mt is not None and row.get("total_lean"):
            diff = (sh + sa) - mt
            if diff == 0:
                row["total_result"] = "push"
            else:
                row["total_result"] = ("win" if (diff > 0) ==
                                       row["total_lean"].startswith("Over") else "loss")
        graded += 1

    with open(LOG_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in sorted(log.values(), key=lambda r: (r["date"], r["event_id"])):
            w.writerow(row)
    print(f"Prediction log: {len(log)} games logged, {graded} newly graded -> {LOG_CSV}")
    return list(log.values())


def save_backtest(log_rows):
    """Win-loss-push of the model's leans, by confidence tier."""
    def summarize(result_key, conf_key):
        out = {}
        for label in [l for _, l in CONFIDENCE_TIERS] + ["Pass", "All"]:
            rows = [r for r in log_rows if r.get(result_key)
                    and (label == "All" or r.get(conf_key) == label)]
            w = sum(r[result_key] == "win" for r in rows)
            l = sum(r[result_key] == "loss" for r in rows)
            p = sum(r[result_key] == "push" for r in rows)
            out[label] = {"wins": w, "losses": l, "pushes": p,
                          "win_pct": round(w / (w + l), 3) if w + l else None}
        return out

    result = {
        "last_updated": datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "note": ("Graded against the last line logged before kickoff. "
                 "About 52.4% is needed to break even at -110."),
        "spread": summarize("spread_result", "spread_conf"),
        "total":  summarize("total_result", "total_conf"),
    }
    write_json(BACKTEST_JSON, result)
    s, t = result["spread"]["All"], result["total"]["All"]
    print(f"Backtest -> {BACKTEST_JSON}: spreads {s['wins']}-{s['losses']}-{s['pushes']}, "
          f"totals {t['wins']}-{t['losses']}-{t['pushes']}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== College Football Ratings {SEASON_YEAR} (weeks {WEEKS}) ===")

    print("\nFetching ESPN scoreboard...")
    all_games, upcoming, lines, team_meta, unknown_conf_ids = scrape_season()
    print(f"\nCompleted games: {len(all_games)} | upcoming: {len(upcoming)}")

    if MANUAL_GAMES:
        print(f"\nAdding {len(MANUAL_GAMES)} manual game(s)...")
        manual = normalize_manual(MANUAL_GAMES)
        all_games.extend(manual)
        for g in manual:
            for t in (g[1], g[3]):
                team_meta.setdefault(t, {"conference": NON_FBS_LABEL,
                                         "team_id": None, "abbreviation": None})

    if not all_games:
        print("No completed games found -- exiting. (If games have been "
              "played, check espn_raw/ to see what ESPN returned.)")
        raise SystemExit(1)

    print("\nApplying score corrections...")
    all_games = apply_score_corrections(all_games)
    print("\nApplying game exclusions...")
    all_games = apply_exclusions(all_games)
    print("\nDeduplicating games...")
    all_games = deduplicate_games(all_games)

    print("\nSaving scoreboard CSV...")
    save_csv(all_games)

    print("\nLoading prior...")
    prior = load_prior()

    print("\nRunning ratings engine...")
    groups = {t: m.get("conference_id") for t, m in team_meta.items()}
    off, dfn, ovr, league_avg, hfa, converged = calculate_ratings(all_games, prior, groups)

    print("\nSaving ratings...")
    save_ratings(off, dfn, ovr, league_avg, hfa, converged, team_meta, all_games)

    print("\nBuilding picks for upcoming games...")
    picks = make_picks(upcoming, off, dfn, league_avg, hfa)
    save_picks(picks, hfa)

    print("\nUpdating prediction log and backtest...")
    log_rows = update_prediction_log(picks, all_games)
    save_backtest(log_rows)

    if unknown_conf_ids:
        print(f"\nNote: conferenceIds not in FBS_CONFERENCES (labelled "
              f"'{NON_FBS_LABEL}'): {sorted(unknown_conf_ids)}")
    print("\n=== Done ===")
