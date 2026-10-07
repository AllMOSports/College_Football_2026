"""
College Football Ratings 2026
-----------------------------
Same v2 rating engine as football_ratings_2025.py (soft competitiveness
weighting + shrinkage + MOV cap), with the MSHSAA scraper swapped out for
ESPN's college-football scoreboard feed.

Why the JSON feed instead of the schedule page:
  https://www.espn.com/college-football/schedule/_/week/1/year/2026/seasontype/2
is rendered client-side with JavaScript, so requests + BeautifulSoup gets an
empty shell. The page pulls its data from the scoreboard API below; we read
that directly. Same games, same week/year/seasontype parameters.

Classifications (1-6) and districts are replaced by conference, which comes
straight from the ESPN feed -- no classifications.json / schools CSV needed.
"""

import requests
import json
import csv
import os
import pandas as pd
from datetime import datetime
import time

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

SEASON_YEAR   = 2026
SEASON_TYPE   = 2            # 2 = regular season, 3 = postseason (bowls/CFP)
WEEKS         = [1]          # e.g. list(range(1, 16)) for the whole regular season


def _weeks_from_env(default):
    """
    Optional override from the CFB_WEEKS environment variable (used by the
    GitHub Actions workflow). Accepts "1", "1,2,3", or a range like "1-15".
    Weeks that haven't been played yet are harmless: the script only keeps
    games ESPN marks as final.
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
INCLUDE_POSTSEASON = False   # also pull bowls/CFP (seasontype=3, week=1)

# ESPN "groups": 80 = FBS (includes FBS-vs-FCS games), 81 = FCS.
# The ESPN schedule page you linked shows group 80. Add 81 to also rate
# FCS-vs-FCS games, which gives the FCS teams far more data to work with.
GROUPS        = [80]

SCOREBOARD_URL = ("https://site.api.espn.com/apis/site/v2/sports/football/"
                  "college-football/scoreboard"
                  "?year={year}&seasontype={stype}&week={week}"
                  "&groups={group}&limit=500")

MAX_POINTS    = 150          # sanity ceiling (CFB can exceed the HS 100 cap)
OUTPUT_PATH   = f"cfb_ratings_{SEASON_YEAR}.json"
CSV_PATH      = f"cfb_scoreboard_{SEASON_YEAR}.csv"
RAW_CACHE_DIR = "espn_raw"   # raw JSON per week/group is saved here for auditing
ITERATIONS    = 1000
LEARNING_RATE = 0.1

# --- v2 rating engine settings (unchanged from the high school version) ---
COMPETITIVE_THRESHOLD = 40    # "half-weight" point of the smooth decay curve
REGULARIZATION_K      = 3.0   # pseudo-games added to every team's denominator
MOV_CAP               = 28    # max points of "error" a single game can contribute

# ESPN conferenceId -> conference name for FBS. Anything not in this map is
# labelled NON_FBS_LABEL (FCS and below). Verify against ESPN if realignment
# changes an ID; unknown IDs fall through to NON_FBS_LABEL and are reported.
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
# MANUAL GAMES / CORRECTIONS / EXCLUSIONS  (same format as the HS script)
# ---------------------------------------------------------------------------
# Team names must match the names this script prints (ESPN "location" names,
# e.g. "Ohio State", "Miami", "Miami (OH)", "North Dakota State").
# Format: ("YYYY-MM-DD", "Team 1", score1, "Team 2", score2)
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
# HTTP SESSION (connection reuse + one retry, same as HS script)
# ---------------------------------------------------------------------------

def build_session():
    from requests.adapters import HTTPAdapter
    try:
        from urllib3.util.retry import Retry
    except ImportError:
        from requests.packages.urllib3.util.retry import Retry

    session = requests.Session()
    retry = Retry(total=1, connect=1, read=1, backoff_factor=1.5,
                  status_forcelist=[500, 502, 503, 504], raise_on_status=False)
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


def parse_scoreboard(data, team_info):
    """
    Turn one ESPN scoreboard JSON payload into game tuples.

    Returns a list of (event_id, date, home_id, home_score, away_id,
    away_score, neutral_site) for COMPLETED games only. Also fills
    team_info[team_id] = {"name", "full_name", "conference_id"}.
    Counts of skipped games are returned for reporting.
    """
    games = []
    skipped = {"not_final": 0, "bad_score": 0, "malformed": 0}

    for event in data.get("events", []):
        comps = event.get("competitions") or []
        if not comps:
            skipped["malformed"] += 1
            continue
        comp = comps[0]

        status = (comp.get("status") or event.get("status") or {}).get("type", {})
        if not status.get("completed", False):
            skipped["not_final"] += 1
            continue
        # Canceled/postponed/forfeit games can be flagged completed with no
        # real score; ESPN marks these with a non-FINAL status name.
        if status.get("name", "STATUS_FINAL") not in ("STATUS_FINAL",):
            skipped["not_final"] += 1
            continue

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
                "conference_id": str(team.get("conferenceId", "")),
            })
            sides[c.get("homeAway")] = (tid, parse_score(c.get("score")))

        if set(sides) != {"home", "away"}:
            skipped["malformed"] += 1
            continue
        (h_id, h_score), (a_id, a_score) = sides["home"], sides["away"]
        if h_score is None or a_score is None:
            skipped["bad_score"] += 1
            continue

        games.append((
            str(event.get("id")),
            (event.get("date") or comp.get("date") or "")[:10],
            h_id, h_score, a_id, a_score,
            bool(comp.get("neutralSite", False)),
        ))

    return games, skipped


def make_unique_names(team_info):
    """
    ESPN 'location' is the short school name ("Ohio State"). Two schools can
    share a location, so fall back to the full display name on collisions.
    Returns {team_id: display_name}.
    """
    by_name = {}
    for tid, info in team_info.items():
        by_name.setdefault(info["name"], []).append(tid)
    names = {}
    for name, ids in by_name.items():
        for tid in ids:
            names[tid] = name if len(ids) == 1 else team_info[tid]["full_name"]
    # Last resort: if full names also collide, tag with the ESPN team id so
    # two different schools can never be merged into one rating.
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
    raw_games = []
    failed    = []

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

            games, skipped = parse_scoreboard(data, team_info)
            raw_games.extend(games)
            print(f"{len(games)} final games "
                  f"(skipped: {skipped['not_final']} not final, "
                  f"{skipped['bad_score']} bad score, "
                  f"{skipped['malformed']} malformed)")
            time.sleep(0.5)

    print(f"\n  [TIMING] Fetching took {time.perf_counter() - t0:.1f}s.")
    if failed:
        print(f"  *** {len(failed)} pull(s) failed -- these weeks may be "
              f"missing games: {failed}")

    # Dedup by ESPN event id (a game shows up in both group 80 and 81
    # when it's FBS vs FCS).
    seen, unique = set(), []
    for g in raw_games:
        if g[0] in seen:
            continue
        seen.add(g[0])
        unique.append(g)
    if len(unique) != len(raw_games):
        print(f"  Removed {len(raw_games) - len(unique)} duplicate event(s) "
              f"across groups.")

    names = make_unique_names(team_info)
    all_games = [(date, names[h], hs, names[a], as_)
                 for _, date, h, hs, a, as_, _neutral in unique]

    team_to_conf = {}
    unknown_conf_ids = set()
    for tid, info in team_info.items():
        cid = info["conference_id"]
        conf = FBS_CONFERENCES.get(cid)
        if conf is None:
            conf = NON_FBS_LABEL
            if cid:
                unknown_conf_ids.add(cid)
        team_to_conf[names[tid]] = conf

    return all_games, team_to_conf, unknown_conf_ids

# ---------------------------------------------------------------------------
# CORRECTIONS / EXCLUSIONS / DEDUP  (unchanged logic from HS script)
# ---------------------------------------------------------------------------

def apply_score_corrections(all_games, corrections=SCORE_CORRECTIONS):
    lookup = {(d, frozenset([a, b])): {a: sa, b: sb}
              for d, a, sa, b, sb in corrections}
    corrected, fixed = 0, []
    for d, t1, s1, t2, s2 in all_games:
        fix = lookup.get((d, frozenset([t1, t2])))
        if fix:
            n1, n2 = fix.get(t1, s1), fix.get(t2, s2)
            corrected += (n1, n2) != (s1, s2)
            fixed.append((d, t1, n1, t2, n2))
        else:
            fixed.append((d, t1, s1, t2, s2))
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

# ---------------------------------------------------------------------------
# CSV OUTPUT
# ---------------------------------------------------------------------------

def save_csv(all_games):
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Home Team", "Home Score", "Away Team", "Away Score"])
        w.writerows(all_games)
    print(f"Saved {len(all_games)} games to {CSV_PATH}")

# ---------------------------------------------------------------------------
# RATING ENGINE (v2) -- identical to football_ratings_2025.py
# ---------------------------------------------------------------------------

def competitiveness_weight(gap, scale=COMPETITIVE_THRESHOLD):
    return 1.0 / (1.0 + (gap / scale) ** 2)


def run_iterations(games, teams, off_rating, def_rating, league_avg,
                   iterations, phase_label="Fit"):
    for iteration in range(iterations):
        off_error  = {t: 0.0 for t in teams}
        def_error  = {t: 0.0 for t in teams}
        weight_sum = {t: 0.0 for t in teams}

        for t1, t2, actual_s1, actual_s2 in games:
            gap = abs((off_rating[t1] + def_rating[t1]) -
                      (off_rating[t2] + def_rating[t2]))
            w = competitiveness_weight(gap)

            predicted_s1 = off_rating[t1] - def_rating[t2] + league_avg
            predicted_s2 = off_rating[t2] - def_rating[t1] + league_avg

            error_s1 = max(-MOV_CAP, min(MOV_CAP, actual_s1 - predicted_s1))
            error_s2 = max(-MOV_CAP, min(MOV_CAP, actual_s2 - predicted_s2))

            off_error[t1] += w * error_s1
            off_error[t2] += w * error_s2
            def_error[t1] += -w * error_s2
            def_error[t2] += -w * error_s1

            weight_sum[t1] += w
            weight_sum[t2] += w

        for team in teams:
            denom = weight_sum[team] + REGULARIZATION_K
            off_rating[team] += (off_error[team] / denom) * LEARNING_RATE
            def_rating[team] += (def_error[team] / denom) * LEARNING_RATE

        if (iteration + 1) % 100 == 0:
            print(f"  [{phase_label}] Iteration {iteration + 1}/{iterations} complete")


def calculate_ratings(all_games, iterations=ITERATIONS):
    games = [(t1, t2, s1, s2) for _, t1, s1, t2, s2 in all_games]
    teams = list({t for t1, t2, _, _ in games for t in (t1, t2)})
    if not teams:
        return {}, {}, {}, 0

    all_scores = [s for _, _, s1, s2 in games for s in (s1, s2)]
    league_avg = sum(all_scores) / len(all_scores)
    print(f"  League average: {league_avg:.2f} points per game")

    off_rating = {t: 0.0 for t in teams}
    def_rating = {t: 0.0 for t in teams}

    print(f"\n  Running rating fit ({iterations} iterations, "
          f"scale={COMPETITIVE_THRESHOLD}, K={REGULARIZATION_K}, "
          f"MOV cap={MOV_CAP})...")
    print(f"  [TIMING] {len(teams)} teams, {len(games)} games going into the fit.")
    t0 = time.perf_counter()
    run_iterations(games, teams, off_rating, def_rating, league_avg, iterations)
    print(f"  [TIMING] Rating fit took {time.perf_counter() - t0:.1f}s.")

    ovr_rating = {t: round(off_rating[t] + def_rating[t], 2) for t in teams}
    return off_rating, def_rating, ovr_rating, league_avg

# ---------------------------------------------------------------------------
# OUTPUT (conference replaces classification/district)
# ---------------------------------------------------------------------------

def games_played(all_games):
    gp = {}
    for _, t1, _, t2, _ in all_games:
        gp[t1] = gp.get(t1, 0) + 1
        gp[t2] = gp.get(t2, 0) + 1
    return gp


def build_team_entries(off_rating, def_rating, ovr_rating, team_to_conf, gp,
                       conf_filter=None, fbs_only=False):
    pool = list(ovr_rating.keys())
    if conf_filter is not None:
        pool = [t for t in pool if team_to_conf.get(t) == conf_filter]
    if fbs_only:
        pool = [t for t in pool if team_to_conf.get(t) != NON_FBS_LABEL]

    ovr_sorted = sorted(pool, key=lambda t: ovr_rating[t], reverse=True)
    off_rank = {t: i + 1 for i, t in enumerate(
        sorted(pool, key=lambda t: off_rating[t], reverse=True))}
    def_rank = {t: i + 1 for i, t in enumerate(
        sorted(pool, key=lambda t: def_rating[t], reverse=True))}

    return [
        {
            "ovr_rank":     i + 1,
            "school":       t,
            "conference":   team_to_conf.get(t),
            "games":        gp.get(t, 0),
            "ovr_rating":   ovr_rating[t],
            "off_rating":   round(off_rating[t], 2),
            "off_rank":     off_rank[t],
            "def_rating":   round(def_rating[t], 2),
            "def_rank":     def_rank[t],
        }
        for i, t in enumerate(ovr_sorted)
    ]


def write_json(path, entries, league_avg, extra=None):
    output = {
        "last_updated":   datetime.now().strftime("%B %d, %Y at %I:%M %p"),
        "season":         SEASON_YEAR,
        "weeks":          WEEKS,
        "league_average": round(league_avg, 2),
        **(extra or {}),
        "teams":          entries,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)


def save_all_outputs(off_rating, def_rating, ovr_rating, league_avg,
                     team_to_conf, all_games):
    gp = games_played(all_games)

    # Overall (every team that played, FCS included)
    entries = build_team_entries(off_rating, def_rating, ovr_rating,
                                 team_to_conf, gp)
    write_json(OUTPUT_PATH, entries, league_avg)
    print(f"Saved {len(entries)} teams to {OUTPUT_PATH}")

    # FBS-only ranking (what most people will want to look at)
    fbs = build_team_entries(off_rating, def_rating, ovr_rating,
                             team_to_conf, gp, fbs_only=True)
    fbs_path = f"cfb_ratings_{SEASON_YEAR}_fbs.json"
    write_json(fbs_path, fbs, league_avg, {"scope": "FBS"})
    print(f"Saved {len(fbs)} FBS teams to {fbs_path}")
    print("Top 10 FBS:")
    for e in fbs[:10]:
        print(f"  {e['ovr_rank']:>2}. {e['school']:<22} ({e['conference']}) "
              f"| OVR {e['ovr_rating']:+.2f} | OFF {e['off_rating']:+.2f} "
              f"| DEF {e['def_rating']:+.2f}")

    # Per-conference JSON
    for conf in sorted(set(team_to_conf.get(t) for t in ovr_rating)):
        conf_entries = build_team_entries(off_rating, def_rating, ovr_rating,
                                          team_to_conf, gp, conf_filter=conf)
        slug = conf.lower().replace(" / ", "_").replace(" ", "_").replace("-", "")
        write_json(f"cfb_ratings_{SEASON_YEAR}_{slug}.json", conf_entries,
                   league_avg, {"conference": conf})
        print(f"  {conf}: {len(conf_entries)} teams")

    # Rankings CSV (overall)
    df = pd.DataFrame([{
        "School": e["school"], "Conference": e["conference"], "Games": e["games"],
        "OFF Rating": e["off_rating"], "DEF Rating": e["def_rating"],
        "OVR Rating": e["ovr_rating"], "OFF Rank": e["off_rank"],
        "DEF Rank": e["def_rank"], "OVR Rank": e["ovr_rank"],
    } for e in entries])
    path = f"cfb_rankings_{SEASON_YEAR}_all.csv"
    df.to_csv(path, index=False)
    print(f"  Rankings CSV: {path}")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"=== College Football Ratings {SEASON_YEAR} (weeks {WEEKS}) ===")

    print("\nFetching ESPN scoreboard...")
    all_games, team_to_conf, unknown_conf_ids = scrape_season()
    print(f"\nTotal completed games: {len(all_games)}")
    if unknown_conf_ids:
        print(f"  Note: conferenceIds not in FBS_CONFERENCES (labelled "
              f"'{NON_FBS_LABEL}'): {sorted(unknown_conf_ids)}")

    if MANUAL_GAMES:
        print(f"\nAdding {len(MANUAL_GAMES)} manual game(s)...")
        all_games.extend(MANUAL_GAMES)
        for _, t1, _, t2, _ in MANUAL_GAMES:
            team_to_conf.setdefault(t1, NON_FBS_LABEL)
            team_to_conf.setdefault(t2, NON_FBS_LABEL)

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

    print("\nRunning ratings engine...")
    off_rating, def_rating, ovr_rating, league_avg = calculate_ratings(all_games)

    print("\nSaving outputs...")
    save_all_outputs(off_rating, def_rating, ovr_rating, league_avg,
                     team_to_conf, all_games)
    print("\n=== Done ===")
