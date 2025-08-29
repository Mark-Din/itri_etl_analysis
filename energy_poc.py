# energy_validation_poc.py
import argparse, re, os, math
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# -----------------------------
# Config (CLI)
# -----------------------------
ap = argparse.ArgumentParser(description="Grading/Label validation from Excel files (no SQL).")
ap.add_argument("--grading", nargs="*", default=[], help="Paths to grading-system Excel files")
ap.add_argument("--label",   nargs="*", default=[], help="Paths to label-system Excel files")
ap.add_argument("--year",    type=int, default=2025, help="Target year to validate (current year)")
ap.add_argument("--yoy-low", type=float, default=-50.0, help="YoY low threshold in % (drop)")
ap.add_argument("--yoy-high",type=float, default=200.0, help="YoY high threshold in % (spike)")
ap.add_argument("--out",     type=str, default="outputs", help="Output directory")
ap.add_argument("--synthetic", action="store_true", help="Generate ~1000 synthetic rows if inputs are empty/missing")
args = ap.parse_args()

OUT_DIR = Path(args.out); OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_YEAR = args.year
YOY_LOW, YOY_HIGH = args.yoy_low, args.yoy_high


# -----------------------------
# Helpers
# -----------------------------
def clean_text(x: str) -> str:
    """Normalize vendor/model text for joins."""
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    x = re.sub(r"[\s\-–—]+", "", x)  # remove spaces/dashes
    return x

def pick_first_exist(df: pd.DataFrame, candidates: list[str]):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def extract_year_from_col(col: str):
    """Try to parse a year from column name like '2019產量', '2020銷售量', '登錄年度' -> None."""
    m = re.search(r"(20\d{2})", col)
    return int(m.group(1)) if m else None

def pct_change(cur, prev):
    if prev is None or prev == 0 or pd.isna(prev):
        return None
    return (cur - prev) * 100.0 / prev


# -----------------------------
# Load Excel → Long-form frames
# -----------------------------
@dataclass
class LongGrading:
    vendor: str
    model: str
    year: int
    prod_qty: float | None
    sales_qty: float | None
    eff_class: str | None  # 效率分級 if present

@dataclass
class LongLabel:
    vendor: str
    model: str
    year: int
    status: str | None      # valid/expired/etc if derivable
    eff_class: str | None   # optional, if present


def read_any_excel(paths: list[str]) -> list[pd.DataFrame]:
    frames = []
    for p in paths:
        try:
            df = pd.read_excel(p)
            if df is not None and len(df.columns) > 0:
                frames.append(df)
        except Exception:
            pass
    return frames


def melt_grading_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Convert wide grading excel(s) into long {vendor, model, year, prod_qty, sales_qty, eff_class}."""
    longs = []
    for raw in frames:
        df = raw.copy()

        # Identify vendor/model columns by common headers
        vendor_col = pick_first_exist(df, ["廠牌名稱","標章公司","標示義務公司","廠商","品牌","公司名稱","Vendor"])
        model_col  = pick_first_exist(df, ["型號","產品型號","機型","Model"])
        year_col   = pick_first_exist(df, ["登錄年度","年度","Year"])  # sometimes exists, otherwise rely on '2019產量' etc.
        eff_col    = pick_first_exist(df, ["效率分級","分級","能效等級","EfficiencyClass"])

        # Clean textual keys
        if vendor_col: df[vendor_col] = df[vendor_col].map(clean_text)
        if model_col:  df[model_col]  = df[model_col].map(clean_text)

        # Find any columns that look like year-specific production/sales
        prod_cols = [c for c in df.columns if re.search(r"20\d{2}", str(c)) and ("產量" in str(c))]
        sale_cols = [c for c in df.columns if re.search(r"20\d{2}", str(c)) and ("銷售" in str(c))]

        # print("prod_cols==============",prod_cols)
        # print("sale_cols==============",sale_cols)
        # If explicit annual columns exist, melt them
        melted_rows = []
        if (vendor_col and model_col) and (prod_cols or sale_cols):
            # Build year-wise rows from prod/sales
            # Use the union of years present across prod/sales cols
            years = set()
            for c in prod_cols + sale_cols:
                y = extract_year_from_col(str(c))
                if y: years.add(y)

            for _, r in df.iterrows():
                ven = r[vendor_col]
                mod = r[model_col]
                eff = r[eff_col] if eff_col in df.columns else None
                for y in sorted(years):
                    prod = None
                    sales = None
                    # match exact column for this year
                    for c in prod_cols:
                        if extract_year_from_col(c) == y:
                            prod = r[c]
                            break
                    for c in sale_cols:
                        if extract_year_from_col(c) == y:
                            sales = r[c]
                            break
                    if pd.isna(ven) or pd.isna(mod):
                        continue
                    melted_rows.append(LongGrading(ven, mod, int(y),
                                                   float(prod) if pd.notna(prod) else None,
                                                   float(sales) if pd.notna(sales) else None,
                                                   str(eff) if pd.notna(eff) else None).__dict__)
        else:
            # Fallback: if only a single year column exists (e.g., "登錄年度") and some qty columns
            if vendor_col and model_col and year_col:
                qty_col = pick_first_exist(df, ["產量","銷售量","年銷售量","年產量"])
                if qty_col and qty_col in df.columns:
                    for _, r in df.iterrows():
                        ven = r[vendor_col]; mod = r[model_col]; yr = r[year_col]
                        eff = r[eff_col] if eff_col in df.columns else None
                        if pd.isna(ven) or pd.isna(mod) or pd.isna(yr): 
                            continue
                        melted_rows.append(LongGrading(clean_text(ven), clean_text(mod), int(yr),
                                                       None, float(r[qty_col]), 
                                                       str(eff) if pd.notna(eff) else None).__dict__)
                        
        if melted_rows:
            longs.append(pd.DataFrame(melted_rows))

    return pd.concat(longs, ignore_index=True) if longs else pd.DataFrame(columns=["vendor","model","year","prod_qty","sales_qty","eff_class"])


def melt_label_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Convert wide label excel(s) into long {vendor, model, year, status, eff_class}."""
    longs = []
    for raw in frames:
        df = raw.copy()
        vendor_col = pick_first_exist(df, ["標章公司","標示義務公司","廠牌名稱","Vendor"])
        model_col  = pick_first_exist(df, ["型號","產品型號","機型","Model"])
        year_col   = pick_first_exist(df, ["登錄年度","年度","Year"])  # often present
        eff_col    = pick_first_exist(df, ["效率分級","分級","能效等級","EfficiencyClass"])
        status_col = pick_first_exist(df, ["狀態","Status"])
        # Fallback: derive year from 起約日期
        start_col  = pick_first_exist(df, ["起約日期","生效日期","StartDate"])

        if vendor_col: df[vendor_col] = df[vendor_col].map(clean_text)
        if model_col:  df[model_col]  = df[model_col].map(clean_text)

        rows = []
        for _, r in df.iterrows():
            ven = r[vendor_col] if vendor_col else None
            mod = r[model_col] if model_col else None
            yr  = None
            if year_col and pd.notna(r.get(year_col)):
                try:
                    yr = int(r[year_col])
                except Exception:
                    pass
            elif start_col and pd.notna(r.get(start_col)):
                # attempt parse YYYY from date-like text
                m = re.search(r"(20\d{2})", str(r[start_col]))
                yr = int(m.group(1)) if m else None

            if ven and mod and yr:
                rows.append(LongLabel(ven, mod, yr,
                                      str(r[status_col]) if status_col and pd.notna(r.get(status_col)) else "valid",
                                      str(r[eff_col]) if eff_col and pd.notna(r.get(eff_col)) else None).__dict__)
        if rows:
            longs.append(pd.DataFrame(rows))

    return pd.concat(longs, ignore_index=True) if longs else pd.DataFrame(columns=["vendor","model","year","status","eff_class"])


# -----------------------------
# Synthetic generator (for demos)
# -----------------------------
VENDOR_NAMES = ["ALFA","BRAVO","CHARLIE","DELTA","ECHO","FOXTROT","GOLF","HOTEL","INDIA","JULIET","KILO","LIMA","MIKE","NOVEMBER","OSCAR"]

def generate_synthetic(n_rows: int = 1000, years=(2019,2020,2021)) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(42)
    vendors = rng.choice(VENDOR_NAMES, size=15, replace=False)
    models_per_vendor = rng.integers(low=10, high=25, size=len(vendors))
    rows_g = []
    rows_l = []

    for ven, mcount in zip(vendors, models_per_vendor):
        for mi in range(mcount):
            model = f"{ven}M{mi:03d}"
            # baseline
            base = rng.integers(200, 2000)
            eff  = rng.choice(["1級","2級","3級"])
            # randomly decide if this model participates in label
            has_label = rng.random() > 0.25

            for y in years:
                # introduce YoY drift
                drift = rng.normal(1.0, 0.2)
                sales = max(0, int(base * drift))
                prod  = int(sales * rng.uniform(0.9, 1.1))
                # random missing grading (to trigger 未申報)
                if rng.random() < 0.05 and y == max(years):
                    # missing in last year only
                    pass
                else:
                    rows_g.append({"vendor": ven, "model": model, "year": y,
                                   "prod_qty": float(prod), "sales_qty": float(sales), "eff_class": eff})

                # label presence (simulate some mismatches + expiries)
                if has_label and rng.random() > 0.1:
                    status = "valid" if rng.random() > 0.1 else "expired"
                    eff_l  = eff if rng.random() > 0.9 else rng.choice(["1級","2級","3級"])  # occasional mismatch
                    rows_l.append({"vendor": ven, "model": model, "year": y, "status": status, "eff_class": eff_l})

    g = pd.DataFrame(rows_g)
    l = pd.DataFrame(rows_l)
    # target about n_rows by sampling if too large
    if len(g) > n_rows:
        g = g.sample(n_rows, random_state=7).reset_index(drop=True)
    return g, l


# -----------------------------
# Build long-form datasets
# -----------------------------
grading_frames = read_any_excel(args.grading) if args.grading else []
label_frames   = read_any_excel(args.label) if args.label else []

# print('grading_frames: ======', grading_frames)
# print('label_frames: ======', label_frames)

grading_long = melt_grading_frames(grading_frames) if grading_frames else pd.DataFrame()
label_long   = melt_label_frames(label_frames) if label_frames else pd.DataFrame()

# print('grading_long==========', grading_long)
if args.synthetic or grading_long.empty:
    print("Using synthetic data (~1000 rows)…")
    grading_long, label_long = generate_synthetic(n_rows=1000, years=(2019,2020,2021))

# Safety dtypes
for c in ["prod_qty","sales_qty"]:
    if c in grading_long.columns:
        grading_long[c] = pd.to_numeric(grading_long[c], errors="coerce")

# Normalize keys
grading_long["vendor"] = grading_long["vendor"].map(clean_text)
grading_long["model"]  = grading_long["model"].map(clean_text)
label_long["vendor"]   = label_long["vendor"].map(clean_text)
label_long["model"]    = label_long["model"].map(clean_text)

# -----------------------------
# (1) Grading: missing report & abnormal
# -----------------------------
# Expected = models that appeared in TARGET_YEAR-1 or TARGET_YEAR-2
years_present = grading_long["year"].dropna().astype(int)
min_year, max_year = (int(years_present.min()) if not years_present.empty else TARGET_YEAR-2,
                      int(years_present.max()) if not years_present.empty else TARGET_YEAR)

exp_years = [TARGET_YEAR-2, TARGET_YEAR-1]
expected = (grading_long[grading_long["year"].isin(exp_years)]
            .dropna(subset=["vendor","model"])
            .loc[:, ["vendor","model"]].drop_duplicates())

cur_grading = grading_long[grading_long["year"] == TARGET_YEAR][["vendor","model"]].drop_duplicates()
missing_grade = (expected.merge(cur_grading, on=["vendor","model"], how="left", indicator=True)
                 .query("_merge=='left_only'")
                 .assign(year=TARGET_YEAR,
                         rule_id="MISSING_REPORT_GRADING",
                         detail="No grading report in current year")
                 [["rule_id","vendor","model","year","detail"]])

# YoY abnormal based on sales_qty (fallback to prod_qty if sales missing)
agg = (grading_long
       .assign(qty=lambda d: np.where(d["sales_qty"].notna(), d["sales_qty"], d["prod_qty"]))
       .groupby(["vendor","model","year"], as_index=False)["qty"].sum())
cur  = agg[agg["year"] == TARGET_YEAR].copy()
prev = agg[agg["year"] == TARGET_YEAR-1].rename(columns={"qty":"qty_1y"})[["vendor","model","qty_1y"]]
prev2= agg[agg["year"] == TARGET_YEAR-2].rename(columns={"qty":"qty_2y"})[["vendor","model","qty_2y"]]

yoy = cur.merge(prev, on=["vendor","model"], how="left").merge(prev2, on=["vendor","model"], how="left")
yoy["pct_diff_yoy"]  = yoy.apply(lambda r: pct_change(r["qty"], r["qty_1y"]), axis=1)
yoy["pct_diff_yo2y"] = yoy.apply(lambda r: pct_change(r["qty"], r["qty_2y"]), axis=1)

abn = yoy[
    (yoy["qty_1y"].notna() & ((yoy["pct_diff_yoy"] > YOY_HIGH) | (yoy["pct_diff_yoy"] < YOY_LOW))) |
    (yoy["qty_1y"].isna() & yoy["qty_2y"].notna() & ((yoy["pct_diff_yo2y"] > YOY_HIGH) | (yoy["pct_diff_yo2y"] < YOY_LOW)))
].copy()
abn["rule_id"] = "YOY_QTY_ABNORMAL"
abn["detail"]  = "YoY/Yo2Y change beyond thresholds"
abn = abn[["rule_id","vendor","model","year","detail","qty","qty_1y","qty_2y","pct_diff_yoy","pct_diff_yo2y"]]

# Optional: internal consistency (prod vs sales gap)
gap = (grading_long.assign(prod=grading_long["prod_qty"], sales=grading_long["sales_qty"])
       .dropna(subset=["vendor","model","year"]))
gap = gap.groupby(["vendor","model","year"], as_index=False)[["prod","sales"]].sum()
gap["gap_pct"] = (gap["sales"] - gap["prod"]) / gap["prod"].replace(0,np.nan) * 100.0
incons_g = gap[gap["gap_pct"].abs() > 50]  # >50% gap as example
incons_g = incons_g.assign(rule_id="INTERNAL_INCONSISTENT_GRADING",
                           detail="Sales vs Production gap > 50%")[["rule_id","vendor","model","year","detail","prod","sales","gap_pct"]]

grading_anomalies = pd.concat([missing_grade, abn, incons_g], ignore_index=True)


# -----------------------------
# (2) Label: missing & invalid/expired
# -----------------------------
# Define "should have label": for demo we assume every model reported in grading TARGET_YEAR should have a label
should_label = cur_grading.copy()
lbl_cur = label_long[label_long["year"] == TARGET_YEAR][["vendor","model","status"]].drop_duplicates()

missing_label = (should_label.merge(lbl_cur, on=["vendor","model"], how="left")
                 .query("status.isna()")
                 .assign(year=TARGET_YEAR, rule_id="MISSING_REPORT_LABEL", detail="No label record in current year")
                 [["rule_id","vendor","model","year","detail"]])

invalid_label = (lbl_cur[(lbl_cur["status"].str.lower() != "valid")]
                 .assign(year=TARGET_YEAR, rule_id="INVALID_LABEL_STATUS", detail="Label status not valid")
                 [["rule_id","vendor","model","year","detail"]])

label_anomalies = pd.concat([missing_label, invalid_label], ignore_index=True)


# -----------------------------
# (3) Cross-system consistency
# -----------------------------
# Compare efficiency class if present
g_eff = (grading_long[grading_long["year"] == TARGET_YEAR]
         [["vendor","model","eff_class"]].dropna().drop_duplicates().rename(columns={"eff_class":"g_class"}))
l_eff = (label_long[label_long["year"] == TARGET_YEAR]
         [["vendor","model","eff_class"]].dropna().drop_duplicates().rename(columns={"eff_class":"l_class"}))

cls_cmp = g_eff.merge(l_eff, on=["vendor","model"], how="inner")
cls_mismatch = cls_cmp[cls_cmp["g_class"] != cls_cmp["l_class"]]
cls_mismatch = cls_mismatch.assign(rule_id="GRADING_LABEL_INCONSISTENT",
                                   year=TARGET_YEAR, detail="Efficiency class mismatch (grading vs label)")[
    ["rule_id","vendor","model","year","detail","g_class","l_class"]
]

# Presence mismatch (one exists, the other missing)
g_presence = cur_grading.assign(in_grading=True)
l_presence = lbl_cur.assign(in_label=True)[["vendor","model","in_label"]]
presence = g_presence.merge(l_presence, on=["vendor","model"], how="outer")
presence_incons = presence[(presence["in_grading"].isna()) | (presence["in_label"].isna())].copy()
presence_incons = presence_incons.assign(rule_id="GRADING_LABEL_PRESENCE_MISMATCH",
                                         year=TARGET_YEAR,
                                         detail="Model present in one system but not the other")[
    ["rule_id","vendor","model","year","detail"]
]

cross_anomalies = pd.concat([cls_mismatch, presence_incons], ignore_index=True)


# -----------------------------
# UNION all anomalies & export
# -----------------------------
for df in [grading_anomalies, label_anomalies, cross_anomalies]:
    if "vendor" in df.columns: df["vendor"] = df["vendor"].astype(str)
    if "model"  in df.columns: df["model"]  = df["model"].astype(str)

anomalies_all = pd.concat([grading_anomalies, label_anomalies, cross_anomalies], ignore_index=True)
anomalies_all = anomalies_all.sort_values(["vendor","rule_id","model","year"], kind="stable")

# Write summary CSV
summary_path = OUT_DIR / f"anomalies_summary_{TARGET_YEAR}.csv"
anomalies_all.to_csv(summary_path, index=False)
print(f"[OK] Wrote summary: {summary_path}")

# Per-vendor Excel
vendors = anomalies_all["vendor"].dropna().unique().tolist()
per_vendor_dir = OUT_DIR / f"vendor_reports_anomalies_{TARGET_YEAR}"
if not anomalies_all.empty:
    per_vendor_dir.mkdir(parents=True, exist_ok=True)
    for ven in vendors:
        sub = anomalies_all[anomalies_all["vendor"] == ven].copy()
        if sub.empty: continue
        path = per_vendor_dir / f"{ven}_anomalies_{TARGET_YEAR}.xlsx"
        with pd.ExcelWriter(path, engine="openpyxl") as xw:
            sub.to_excel(xw, index=False, sheet_name="anomalies")
        print(" -", path.name)


# -----------------------------
# Charts (Matplotlib, one plot each)
# -----------------------------
# Market share (by sales) for TARGET_YEAR

grading_long.to_csv('grading_long.csv', index=False)
print('grading_long===========', grading_long)
agg_year = (grading_long[grading_long["year"] == TARGET_YEAR]
            .assign(qty=lambda d: np.where(d["sales_qty"].notna(), d["sales_qty"], d["prod_qty"]))
            .groupby("vendor", as_index=False)["qty"].sum()
            .sort_values("qty", ascending=False))
print('agg_year===========', agg_year)

if not agg_year.empty:
    plt.figure()
    plt.bar(agg_year["vendor"], agg_year["qty"])
    plt.title(f"Grading Market Size (Qty) — {TARGET_YEAR}")
    plt.xlabel("Vendor"); plt.ylabel("Qty"); plt.xticks(rotation=90); plt.grid(True, axis="y")
    plt.tight_layout()
    fig1 = OUT_DIR / f"market_size_{TARGET_YEAR}.png"
    plt.savefig(fig1, dpi=150)
    plt.close()
    print(f"[OK] Chart: {fig1}")

# YoY trend per top-N vendors
topN = agg_year.head(6)["vendor"].tolist() if not agg_year.empty else []
if topN:
    series = (grading_long.assign(qty=lambda d: np.where(d["sales_qty"].notna(), d["sales_qty"], d["prod_qty"]))
              .groupby(["vendor","year"], as_index=False)["qty"].sum())
    for ven in topN:
        sub = series[series["vendor"] == ven].sort_values("year")
        if sub.empty: continue
        plt.figure()
        plt.plot(sub["year"], sub["qty"], marker="o")
        plt.title(f"YoY Grading Qty — {ven}")
        plt.xlabel("Year"); plt.ylabel("Qty"); plt.grid(True)
        plt.tight_layout()
        path = OUT_DIR / f"yoy_trend_{ven}.png"
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"[OK] Chart: {path}")


# -----------------------------
# Simple forecast (per vendor total)
# -----------------------------
def simple_lr_forecast(vendor_df: pd.DataFrame, horizon=(TARGET_YEAR+1, TARGET_YEAR+2)):
    if vendor_df.shape[0] < 2:
        return {y: None for y in horizon}
    X = vendor_df[["year"]].values
    y = vendor_df["qty"].values
    if len(np.unique(X)) < 2:
        return {y: None for y in horizon}
    mdl = LinearRegression().fit(X, y)
    pred = {}
    for fy in horizon:
        pred[fy] = float(mdl.predict(np.array([[fy]], dtype=float))[0])
    return pred

vendor_totals = (grading_long.assign(qty=lambda d: np.where(d["sales_qty"].notna(), d["sales_qty"], d["prod_qty"]))
                 .groupby(["vendor","year"], as_index=False)["qty"].sum())

fh = (TARGET_YEAR+1, TARGET_YEAR+2)
rows = []
for ven, sub in vendor_totals.groupby("vendor"):
    sub = sub.sort_values("year")
    preds = simple_lr_forecast(sub, fh)
    rows.append({"vendor": ven, **{f"pred_{y}": preds.get(y) for y in fh}})
forecast_df = pd.DataFrame(rows).sort_values("vendor")
pred_path = OUT_DIR / f"forecast_{TARGET_YEAR+1}_{TARGET_YEAR+2}.csv"
forecast_df.to_csv(pred_path, index=False)
print(f"[OK] Forecast table: {pred_path}")

print("\nDone.")
