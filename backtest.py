import os, io, sys, json as _json, pathlib, requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score

HERE = pathlib.Path(__file__).parent
OUT  = HERE / "out"
OUT.mkdir(exist_ok=True)

# ---------------- config / utils ----------------
def load_config():
    import yaml
    return yaml.safe_load((HERE / "config.yaml").read_text())

def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)

def ymd(d: date) -> str:
    return d.isoformat()

# ---------------- loaders ----------------
def fetch_json_day_gcs(template_url: str, day_iso: str):
    """Download one daily JSON (list or {'rows': [...]}) -> DataFrame."""
    url = template_url.replace("{date}", day_iso)
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return None
    try:
        j = r.json()
    except Exception:
        j = _json.loads(r.content.decode("utf-8", errors="ignore"))
    rows = j if isinstance(j, list) else j.get("rows", [])
    if not rows:
        return None
    return pd.DataFrame.from_records(rows)

def apply_field_map(df: pd.DataFrame, field_map: dict) -> pd.DataFrame:
    """Rename user fields -> internal names (only those provided)."""
    if not field_map:
        return df
    ren = {}
    cols = set(df.columns)
    for std, src in field_map.items():
        if src in cols:
            ren[src] = std
    return df.rename(columns=ren)

# ---------------- normalization ----------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make columns lower-case and enforce minimal set:
      - REQUIRED: timestamp, price
      - OPTIONAL: s5, s20, s50, s100, bidv50, askv50, best_bid, best_ask
    Adds ts (UTC), sorts, coerces numeric. Missing optional cols are filled with NaN.
    """
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns:
        raise ValueError("Missing 'timestamp' after field mapping.")
    if "price" not in df.columns:
        raise ValueError("Missing 'price' after field mapping.")

    # ensure opt columns exist (if absent, create as NaN)
    opt = ["s5","s20","s50","s100","bidv50","askv50","best_bid","best_ask"]
    for c in opt:
        if c not in df.columns:
            df[c] = np.nan

    # timestamps
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values("ts").dropna(subset=["ts"])

    # numeric coercion
    num_cols = ["price","best_bid","best_ask","s5","s20","s50","s100","bidv50","askv50"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # require price present
    df = df.dropna(subset=["price"])
    return df

# ---------------- features ----------------
def make_features(df: pd.DataFrame, use_depths):
    """
    Build features from whatever spread% columns are present.
    For 1‑minute cadence, rolling windows are in minutes.
    """
    # map depth -> internal column name if present
    depth_map = {5:"s5", 20:"s20", 50:"s50", 100:"s100"}
    use_cols = [depth_map[d] for d in use_depths if depth_map.get(d) in df.columns]

    # generate features
    for c in use_cols:
        df[f"{c}_ma30"]    = df[c].rolling(30, min_periods=10).mean()
        df[f"{c}_ma300"]   = df[c].rolling(300, min_periods=50).mean()
        df[f"{c}_slope5"]  = df[c].diff(5)
        df[f"{c}_slope30"] = df[c].diff(30)
        df[f"{c}_accel"]   = df[c].diff(5).diff(5)
        df[f"{c}_z30"]     = (df[c] - df[c].rolling(30).mean()) / (df[c].rolling(30).std() + 1e-9)
        df[f"{c}_z300"]    = (df[c] - df[c].rolling(300).mean()) / (df[c].rolling(300).std() + 1e-9)

    # depth pressure only if we have volumes
    if "bidv50" in df.columns and "askv50" in df.columns:
        if df["bidv50"].notna().any() and df["askv50"].notna().any():
            df["depth_pressure"] = (df["bidv50"] - df["askv50"]) / (df["bidv50"] + df["askv50"] + 1e-9)

    # cross-depth shape if both present
    if "s100" in df.columns and "s20" in df.columns:
        df["shape_100_20"] = df["s100"] - df["s20"]

    # drop warmup NaNs from rolling features
    return df.dropna().reset_index(drop=True)

# ---------------- labels ----------------
def add_labels(df: pd.DataFrame, horizon_sec: int, cadence_seconds: int, thresh_q: float):
    """
    Nondirectional 'vol spike':
      y=1 if |log-return over next horizon| > day-level quantile.
    Uses cadence to convert seconds -> steps.
    """
    steps = max(1, int(round(horizon_sec / max(1, cadence_seconds))))
    df["logp"] = np.log(df["price"].replace(0, np.nan)).ffill()
    fwd = df["logp"].shift(-steps) - df["logp"]
    df["fwd_vol"] = fwd.abs()

    day = df["ts"].dt.date
    thresh = df.groupby(day)["fwd_vol"].transform(lambda s: s.quantile(thresh_q))
    df["y"] = (df["fwd_vol"] > thresh).astype(int)

    # remove last 'steps' rows (no lookahead)
    if len(df) > steps:
        df = df.iloc[:-steps].copy()
    else:
        df = df.iloc[0:0].copy()
    return df

# ---------------- train/eval ----------------
def train_eval(df: pd.DataFrame):
    feat_cols = [c for c in df.columns if any(k in c for k in [
        "s5","s20","s50","s100","ma","slope","accel","z3","depth_pressure","shape_"
    ])]
    if not feat_cols:
        raise RuntimeError("No feature columns were created. Check use_depths and available fields.")

    X = df[feat_cols].values
    y = df["y"].values

    if len(df) < 2000:
        print("Warning: small sample after warmup/labeling; results may be unstable.")

    tscv = TimeSeriesSplit(n_splits=5)
    ap_scores = []
    for tr, te in tscv.split(X):
        base = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf  = CalibratedClassifierCV(base, method="isotonic", cv=3)
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:,1]
        ap_scores.append(average_precision_score(y[te], p))

    pr_auc = float(np.mean(ap_scores))
    print("PR-AUC (avg across folds):", round(pr_auc, 3))

    # fit on full to emit indicator
    base = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf  = CalibratedClassifierCV(base, method="isotonic", cv=3)
    clf.fit(X, y)
    df["LSI"] = clf.predict_proba(X)[:,1] * 100.0

    (OUT / "metrics.txt").write_text(
        f"PR_AUC_avg={pr_auc:.4f}\nrows={len(df)}\nfeatures={len(feat_cols)}\n"
    )
    return df, clf, feat_cols, pr_auc

# ---------------- main ----------------
def main():
    cfg = load_config()

    source          = cfg.get("data_source", "json_gcs")
    json_template   = cfg.get("json_url_template", "")
    field_map       = cfg.get("field_map", {})
    start           = datetime.fromisoformat(cfg["start_date"]).date()
    end             = datetime.fromisoformat(cfg["end_date"]).date()
    use_depths      = cfg.get("use_depths", [20])
    horizon_sec     = int(cfg.get("horizon_sec", 300))
    cadence_seconds = int(cfg.get("cadence_seconds", 60))
    write_out       = bool(cfg.get("write_indicator_csv", True))
    ex              = cfg.get("exchange", "ex")
    asset           = cfg.get("asset", "asset")

    print(f"[load] {source} {start}..{end} cadence={cadence_seconds}s")

    dfs = []
    for d in daterange(start, end):
        day_iso = ymd(d)
        raw = fetch_json_day_gcs(json_template, day_iso) if source == "json_gcs" else None
        if raw is None:
            continue
        raw = apply_field_map(raw, field_map)
        try:
            df = normalize_columns(raw)
        except Exception as e:
            print(f"  skip {day_iso}: {e}")
            continue
        dfs.append(df)

    if not dfs:
        print("No data loaded. Check template URL, dates, and field_map.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    df = make_features(df, use_depths)
    df = add_labels(df, horizon_sec=horizon_sec, cadence_seconds=cadence_seconds, thresh_q=float(cfg.get("threshold_quantile", 0.80)))

    if df.empty:
        print("No rows left after warmup/labeling. Extend date range or lower horizon.")
        sys.exit(1)

    df, model, feats, pr = train_eval(df)

    if write_out:
        out_csv = OUT / f"{ex}_{asset}_{start}_{end}_indicator.csv"
        df[["ts","LSI","y"]].to_csv(out_csv, index=False)
        print("Saved:", out_csv)

if __name__ == "__main__":
    main()
