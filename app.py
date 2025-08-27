import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Hybrid Analytics – July 2025 Forecast & Recommendations",
    layout="wide"
)

# =========================
# Helpers
# =========================
def to_number(x):
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan

def find_col(df, candidates):
    """Case/space/underscore-insensitive + substring fallback."""
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    exact = {c.lower().strip(): c for c in cols}
    unders = {c.lower().replace(" ", "_").strip(): c for c in cols}
    for cand in candidates:
        k = cand.lower().strip()
        if k in exact: return exact[k]
        k2 = cand.lower().replace(" ", "_").strip()
        if k2 in unders: return unders[k2]
    wants = [c.lower().replace(" ", "_").strip() for c in candidates]
    for c in cols:
        cc = c.lower().replace(" ", "_").strip()
        if any(w in cc for w in wants):
            return c
    return None

def kpi(label, value):
    st.metric(label, value)

def bar_plot(labels, values, title, ylabel):
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

# =========================
# Sidebar – uploads & options
# =========================
st.sidebar.header("Upload files")

# For KPIs & chart
cal_file    = st.sidebar.file_uploader("calibrated_projection_2025-07.csv", type=["csv"])

# Your master workbook (contains Movement, Upsell, Cross-sell, +2 more)
master_xlsx = st.sidebar.file_uploader("master_plan_expected_2025-07-01.xlsx", type=["xlsx"])

# Optional product dimension for category count
prod_dim    = st.sidebar.file_uploader("dim_product.csv (optional for categories)", type=["csv"])

top_n      = st.sidebar.number_input("Rows to show (tables)", 10, 5000, 500, 10)
show_debug = st.sidebar.checkbox("Show debug info", value=False)

st.title("Hybrid Analytics – July 2025 Forecast & Recommendations")
st.caption("Summary KPIs & Profit Comparison from calibrated output and the master workbook, plus raw previews of Movement/Upsell/Cross-sell sheets from the same master workbook.")

# =========================
# Load master workbook
# =========================
mv_df = up_df = cs_df = None
mv_sheet = up_sheet = cs_sheet = None
other_sheet_names = []

if master_xlsx:
    xls = pd.ExcelFile(master_xlsx, engine="openpyxl")
    names = xls.sheet_names

    # Let you pick which sheets map to Movement/Upsell/Cross-sell (Auto by default)
    st.sidebar.markdown("**Pick sheets (or leave Auto):**")
    mv_pick = st.sidebar.selectbox("Movement sheet", options=["Auto"] + names, index=0)
    up_pick = st.sidebar.selectbox("Upsell sheet",   options=["Auto"] + names, index=0)
    cs_pick = st.sidebar.selectbox("Cross-sell sheet", options=["Auto"] + names, index=0)

    def autodetect(names_list, keywords):
        # Choose the first sheet whose name contains any keyword (case-insensitive)
        lowers = [s.lower() for s in names_list]
        for i, s in enumerate(lowers):
            if any(k in s for k in keywords):
                return names_list[i]
        return None

    # Resolve sheets (prefer manual choice if provided)
    mv_sheet = mv_pick if mv_pick != "Auto" else (autodetect(names, ["move", "movement"]) or (names[0] if names else None))
    up_sheet = up_pick if up_pick != "Auto" else (autodetect(names, ["upsell", "up-sell"]) or (names[1] if len(names) > 1 else None))
    cs_sheet = cs_pick if cs_pick != "Auto" else (autodetect(names, ["cross", "cross-sell", "crosssell"]) or (names[2] if len(names) > 2 else None))

    # Read the three chosen sheets (if they exist)
    if mv_sheet in names: mv_df = xls.parse(mv_sheet)
    if up_sheet in names: up_df = xls.parse(up_sheet)
    if cs_sheet in names: cs_df = xls.parse(cs_sheet)

    # Any remaining sheets (the workbook has ~5 total)
    chosen = {mv_sheet, up_sheet, cs_sheet}
    other_sheet_names = [s for s in names if s not in chosen]

    if show_debug:
        st.sidebar.write({"movement_sheet": mv_sheet, "upsell_sheet": up_sheet, "crosssell_sheet": cs_sheet})
        if mv_df is not None: st.sidebar.write("Movement cols:", list(mv_df.columns)[:20])
        if up_df is not None: st.sidebar.write("Upsell cols:", list(up_df.columns)[:20])
        if cs_df is not None: st.sidebar.write("Cross-sell cols:", list(cs_df.columns)[:20])

# =========================
# KPIs from calibrated_projection CSV
# =========================
baseline_profit = expected_inc_profit = calibrated_inc_profit = np.nan
uplift_expected_pct = uplift_calibrated_pct = np.nan

if cal_file:
    cal = pd.read_csv(cal_file)
    if {"metric","value"}.issubset(cal.columns):
        def pick(metric_sub):
            row = cal.loc[cal["metric"].str.contains(metric_sub, case=False, na=False)]
            return to_number(row["value"].iloc[0]) if not row.empty else np.nan
        baseline_profit        = pick("Baseline")
        expected_inc_profit    = pick("TOTAL expected profit")
        calibrated_inc_profit  = pick("Calibrated July — TOTAL profit")
        uplift_expected_pct    = pick("Model July uplift")
        uplift_calibrated_pct  = pick("Calibrated July uplift")

def safe_add(a, b):
    return a + b if (pd.notna(a) and pd.notna(b)) else np.nan

campaign_expected_profit   = safe_add(baseline_profit, expected_inc_profit)
campaign_calibrated_profit = safe_add(baseline_profit, calibrated_inc_profit)

# =========================
# Unique counts from the master workbook (Movement/Upsell/Cross-sell)
# =========================
prod_col = cust_col = None
total_products = total_customers = np.nan
all_prod = set()

if any(df is not None for df in [mv_df, up_df, cs_df]):
    # product id
    for df in [mv_df, up_df, cs_df]:
        if df is not None and prod_col is None:
            prod_col = find_col(df, ["Product_Id","ProductID","SKU","Item"])
    if prod_col:
        for df in [mv_df, up_df, cs_df]:
            if df is not None and prod_col in df.columns:
                all_prod.update(pd.Series(df[prod_col]).dropna().unique().tolist())
        total_products = len(all_prod)

    # customer id (from upsell/cross-sell)
    for df in [up_df, cs_df]:
        if df is not None and cust_col is None:
            cust_col = find_col(df, ["Customer_Id","CustomerID","Cust_Id","Customer"])
    if cust_col:
        all_cust = set()
        for df in [up_df, cs_df]:
            if df is not None and cust_col in df.columns:
                all_cust.update(pd.Series(df[cust_col]).dropna().unique().tolist())
        total_customers = len(all_cust)

# unique categories (optional)
total_categories = np.nan
if prod_dim is not None:
    dimp = pd.read_csv(prod_dim)
    cat_col = find_col(dimp, ["Category","category","Product_Category","Category_Id"])
    pid_col = find_col(dimp, ["Product_Id","ProductID","SKU","Item"])
    if pid_col and cat_col:
        total_categories = dimp.loc[dimp[pid_col].isin(list(all_prod)), cat_col].nunique() if all_prod else dimp[cat_col].nunique()

# Total recommendations = rows in upsell + cross-sell sheets from the master
total_recos = (len(up_df) if up_df is not None else 0) + (len(cs_df) if cs_df is not None else 0)

# =========================
# Render – single page
# =========================
st.subheader("Summary KPIs")

# Row 1
c1, c2, c3, c4 = st.columns(4)
with c1: kpi("Unique Products",    f"{int(total_products):,}"     if not np.isnan(total_products)  else "—")
with c2: kpi("Targeted Customers", f"{int(total_customers):,}"    if not np.isnan(total_customers) else "—")
with c3: kpi("Unique Categories",  f"{int(total_categories):,}"   if not np.isnan(total_categories) else "—")
with c4: kpi("Total Recommendations", f"{total_recos:,}")

# Row 2
c5, c6, c7 = st.columns(3)
with c5: kpi("Baseline Profit (₹)",             f"{baseline_profit:,.0f}"            if pd.notna(baseline_profit)            else "—")
with c6: kpi("Campaign Profit – Expected (₹)",  f"{campaign_expected_profit:,.0f}"    if pd.notna(campaign_expected_profit)    else "—")
with c7: kpi("Campaign Profit – Calibrated (₹)",f"{campaign_calibrated_profit:,.0f}"  if pd.notna(campaign_calibrated_profit)  else "—")

# Row 3
c8, c9 = st.columns(2)
with c8:
    if pd.notna(uplift_expected_pct) and pd.notna(baseline_profit) and baseline_profit != 0:
        kpi("Expected Uplift %", f"{uplift_expected_pct:.2f}%")
    elif pd.notna(baseline_profit) and baseline_profit != 0 and pd.notna(expected_inc_profit):
        kpi("Expected Uplift %", f"{(expected_inc_profit/baseline_profit*100):.2f}%")
    else:
        kpi("Expected Uplift %", "—")
with c9:
    if pd.notna(uplift_calibrated_pct) and pd.notna(baseline_profit) and baseline_profit != 0:
        kpi("Calibrated Uplift %", f"{uplift_calibrated_pct:.2f}%")
    elif pd.notna(baseline_profit) and baseline_profit != 0 and pd.notna(calibrated_inc_profit):
        kpi("Calibrated Uplift %", f"{(calibrated_inc_profit/baseline_profit*100):.2f}%")
    else:
        kpi("Calibrated Uplift %", "—")

# Profit comparison chart
st.subheader("Profit Comparison")
if pd.notna(baseline_profit) and pd.notna(campaign_expected_profit) and pd.notna(campaign_calibrated_profit):
    bar_plot(
        ["Baseline","Campaign (Expected)","Campaign (Calibrated)"],
        [baseline_profit, campaign_expected_profit, campaign_calibrated_profit],
        "Profit – Baseline vs Campaign",
        "Profit (₹)"
    )
else:
    st.info("Upload the calibrated_projection CSV to enable the profit chart.")

# =========================
# Show the three master sheets (raw tables, exactly as in Excel)
# =========================
st.subheader("Movement (from Master Workbook)")
if mv_df is not None:
    st.dataframe(mv_df.head(top_n))
else:
    st.info("Select/Auto-detect a Movement sheet in the sidebar and upload the master workbook.")

st.subheader("Upsell (from Master Workbook)")
if up_df is not None:
    st.dataframe(up_df.head(top_n))
else:
    st.info("Select/Auto-detect an Upsell sheet in the sidebar and upload the master workbook.")

st.subheader("Cross-sell (from Master Workbook)")
if cs_df is not None:
    st.dataframe(cs_df.head(top_n))
else:
    st.info("Select/Auto-detect a Cross-sell sheet in the sidebar and upload the master workbook.")

# Optionally preview any remaining sheets from the same master file
if other_sheet_names:
    st.subheader("Other Sheets in Master Workbook")
    pick_others = st.multiselect("Select other sheets to preview",
                                 other_sheet_names,
                                 default=other_sheet_names[:min(2, len(other_sheet_names))])
    for sn in pick_others:
        df_other = pd.read_excel(master_xlsx, sheet_name=sn, engine="openpyxl")
        with st.expander(f"Sheet: {sn}", expanded=True):
            st.dataframe(df_other.head(top_n))
