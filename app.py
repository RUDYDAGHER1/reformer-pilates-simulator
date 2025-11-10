import streamlit as st
import numpy as np
import numpy_financial as nf
import pandas as pd

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Reformer Pilates Investment & Customer Simulator",
    layout="wide"
)

st.title("Reformer Pilates Investment & Recurrent Customer Simulator")

st.markdown(
    "This tool simulates **investment returns** (IRR, NPV, payback), "
    "**capital structure & WACC**, and estimates **recurring clients** needed. "
    "Outputs are shown mainly as tables for clarity."
)

# ---------- HELPER: PAYBACK ----------
def compute_payback(cash_flows):
    """
    cash_flows: array-like, Year 0..N
    returns: payback in years (float) or None if never paid back
    """
    cumulative = np.cumsum(cash_flows)
    for i, val in enumerate(cumulative):
        if val >= 0:
            if i == 0:
                return 0.0
            prev_val = cumulative[i - 1]
            return (i - 1) + (0 - prev_val) / (val - prev_val)
    return None


# ---------- TABS ----------
tab1, tab2, tab3 = st.tabs(
    ["📊 Investment & OPEX Tables", "👥 Recurrent Customers", "💰 Capex Breakdown"]
)

# ==========================
# TAB 1 – INVESTMENT & OPEX
# ==========================
with tab1:
    st.caption("📄 To export this tab to PDF: use your browser's **Print → Save as PDF**.")
    st.header("Investment, Assumptions & OPEX Breakdown")

    # --- KEY ASSUMPTIONS (STATIC, FROM MODEL) ---
    with st.expander("Key Assumptions (Base Case)", expanded=True):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Revenue Growth Assumptions (per year)**")
            st.markdown("- Group Classes: **3.0 %**")
            st.markdown("- Private Sessions: **3.0 %**")
            st.markdown("- Retail (F&B): **3.0 %**")

            st.markdown("**OPEX Escalation Assumptions**")
            st.markdown("- Payroll YoY: **3.0 %**")
            st.markdown("- Rent YoY: **0%, 5%, 0%, 5%** (Years 1–4)")
            st.markdown("- Other Opex Inflation: **2.5 %**")
            st.markdown("- Contingency: **3.0 % of costs**")

        with col_b:
            st.markdown("**Financial Assumptions**")
            st.markdown("- Corporate Tax: **9.0 %**")
            st.markdown("- Target WACC in deck: **10.5 %**")
            st.markdown("- Sales Simulation Factor: **1.0** (slider below)")
            st.markdown("- OPEX Simulation Factor: **1.0** (slider below)")
            st.info(
                "In this app, WACC is recalculated dynamically from your **equity vs loan** "
                "choices and tax rate."
            )

    # ========== INPUTS ==========
    col_left, col_right = st.columns(2)

    # ---- LEFT: OPERATIONS / TAX / SALES ----
    with col_left:
        st.subheader("Operational Assumptions")

        tax_rate = st.number_input(
            "Corporate Tax (%)",
            min_value=0.0,
            max_value=50.0,
            value=9.0,
            step=0.5
        ) / 100.0

        # Net Sales per Year (total, from financial model)
        base_revenue = np.array(
            [2_755_720, 2_838_392, 2_923_543, 3_011_250, 3_101_587],
            dtype=float
        )

        st.caption("Net Sales (Year 1–5, AED) – base case from financial model")

        sales_factor = st.slider(
            "Sales level vs Base Case",
            min_value=0.7,
            max_value=1.3,
            value=1.0,
            step=0.05,
            help="0.7 = -30% sales, 1.3 = +30% sales"
        )

        opex_factor = st.slider(
            "Operating Expenses vs Base Case (multiplier)",
            min_value=0.8,
            max_value=1.2,
            value=1.0,
            step=0.05,
            help="0.8 = -20% OPEX, 1.2 = +20% OPEX (applied to all OPEX lines)"
        )

    # ---- RIGHT: CAPITAL STRUCTURE / WACC / EQUITY SPLIT ----
    with col_right:
        st.subheader("Capital Structure & WACC")

        # Total project cost (Capex + Pre-Op), as per sheet
        total_capex = st.number_input(
            "Total Initial Investment / Project Cost (AED)",
            value=1_712_085.0,
            step=50_000.0
        )

        # Lever-style slider for equity vs loan
        st.caption("Use the lever to split the project cost between **Equity** and **Loan**.")
        max_equity = max(int(total_capex), 0)
        # Default = 100% equity, 0% loan
        default_equity = max_equity

        equity_amount_int = st.slider(
            "Equity (Cash Invested, AED)",
            min_value=0,
            max_value=max_equity,
            value=default_equity,
            step=1000,
            help="Default is 100% equity, 0% loan. Move left for more leverage. Step = 1,000 AED."
        )
        equity_amount = float(equity_amount_int)
        debt_amount = float(total_capex - equity_amount)

        st.markdown(
            f"- **Equity amount:** `{equity_amount:,.0f}` AED  \n"
            f"- **Debt / Loan amount:** `{debt_amount:,.0f}` AED"
        )

        total_capital = max(total_capex, 1.0)
        equity_share = equity_amount / total_capital if total_capital > 0 else 0.0
        debt_share = debt_amount / total_capital if total_capital > 0 else 0.0

        st.markdown(
            f"- **Equity share of capital (for WACC):** `{equity_share * 100:,.1f}%`  \n"
            f"- **Debt share of capital (for WACC):** `{debt_share * 100:,.1f}%`"
        )

        cost_of_equity = st.number_input(
            "Cost of Equity (required return, %)",
            value=10.5,          # <- default 10.5%
            step=0.5
        ) / 100.0

        cost_of_debt = st.number_input(
            "Cost of Debt / Interest Rate (%)",
            value=8.0,
            step=0.5
        ) / 100.0

        loan_term_years = st.number_input(
            "Loan Term (years)",
            min_value=1,
            max_value=10,
            value=5,
            help="Used only if Debt Amount > 0."
        )

        # Calculate WACC
        wacc = equity_share * cost_of_equity + debt_share * cost_of_debt * (1 - tax_rate)
        st.metric("Calculated WACC (dynamic)", f"{wacc * 100:,.2f} %")

        st.subheader("Equity Ownership Structure")

        investor_equity_share = st.slider(
            "Investor Ownership (%)",
            min_value=0,
            max_value=100,
            value=50,
            step=5
        ) / 100.0
        sweat_equity_share = 1 - investor_equity_share

        sweat_equity_mode = st.checkbox(
            "Sweat-Equity Partnership (Investor funds 100% of equity, sweat partner invests 0 cash)",
            value=True,
            help="If checked, the investor provides all equity cash, but ownership is shared as per the slider."
        )

        if sweat_equity_mode:
            investor_equity_investment = equity_amount      # Investor funds all equity
            sweat_equity_investment = 0.0                   # Sweat partner invests 0
        else:
            investor_equity_investment = equity_amount * investor_equity_share
            sweat_equity_investment = equity_amount * sweat_equity_share

        st.markdown(
            f"- **Investor ownership:** `{investor_equity_share * 100:,.1f}%` "
            f"with cash investment `{investor_equity_investment:,.0f}` AED  \n"
            f"- **Sweat-equity ownership:** `{sweat_equity_share * 100:,.1f}%` "
            f"with cash investment `{sweat_equity_investment:,.0f}` AED"
        )

    # ========== OPEX BREAKDOWN ==========
    st.subheader("OPEX Breakdown by Line Item (Simulated)")

    years = np.arange(1, 6)

    # Base OPEX per category from your model
    opex_base = {
        "Instructors Cost": np.array([522_400, 538_072, 554_214, 570_841, 587_966], dtype=float),
        "Admin Salaries & Staff": np.array([300_000, 309_000, 318_270, 327_818, 337_653], dtype=float),
        "Rent": np.array([530_000, 530_000, 556_500, 556_500, 584_325], dtype=float),
        "Utilities & Maintenance": np.array([73_200, 75_030, 76_906, 78_828, 80_799], dtype=float),
        "Marketing & PR": np.array([60_000, 61_500, 63_038, 64_613, 66_229], dtype=float),
        "Consumables & Laundry": np.array([30_000, 30_750, 31_519, 32_307, 33_114], dtype=float),
        "Software & Licenses": np.array([20_000, 20_500, 21_013, 21_538, 22_076], dtype=float),
        "Insurance & Misc.": np.array([30_000, 30_750, 31_519, 32_307, 33_114], dtype=float),
        "Contingency": np.array([46_968, 47_868, 49_589, 50_543, 52_358], dtype=float),
    }

    st.markdown(
        f"**OPEX multiplier applied:** `{opex_factor:,.2f}x` relative to base model "
        "(values below are **already adjusted**)."
    )

    opex_rows = []
    adj_total = np.zeros(5)

    for name, arr in opex_base.items():
        adj = arr * opex_factor
        row = {"OPEX Item": f"**{name}**"}
        for i, y in enumerate(years):
            row[f"Year {y} (AED)"] = adj[i]
            adj_total[i] += adj[i]
        opex_rows.append(row)

    # Totals row
    total_row = {"OPEX Item": "**TOTAL OPEX (all items)**"}
    for i, y in enumerate(years):
        total_row[f"Year {y} (AED)"] = adj_total[i]
    df_opex = pd.DataFrame(opex_rows + [total_row])

    # Format numeric columns only
    num_cols_opex = [c for c in df_opex.columns if c != "OPEX Item"]
    style_map_opex = {c: "{:,.0f}" for c in num_cols_opex}

    st.dataframe(
        df_opex.style.format(style_map_opex),
        use_container_width=True
    )

    # ========== CASH FLOW & RETURNS (NO EXIT) ==========
    st.subheader("Cash Flow & Returns (Year 0–5)")

    # Adjusted Net Sales
    revenue = base_revenue * sales_factor

    # Total adjusted OPEX
    opex_total = adj_total

    # Operating profit and unlevered FCFF
    ebit = revenue - opex_total
    fcff_unlevered = ebit * (1 - tax_rate)

    # Debt schedule
    n_years = len(years)
    interest = np.zeros(n_years)
    principal = np.zeros(n_years)

    if debt_amount > 0:
        n = min(int(loan_term_years), n_years)
        r = cost_of_debt

        if r > 0:
            annuity_factor = r * (1 + r) ** n / ((1 + r) ** n - 1)
            annual_payment = debt_amount * annuity_factor
        else:
            annual_payment = debt_amount / n

        outstanding = debt_amount
        for t in range(n):
            interest[t] = outstanding * r
            principal[t] = annual_payment - interest[t]
            outstanding -= principal[t]
            outstanding = max(outstanding, 0.0)

    # After financing
    ebt = ebit - interest
    tax = np.where(ebt > 0, ebt * tax_rate, 0.0)
    net_income = ebt - tax
    fcfe = net_income - principal
    profit_margin = np.where(revenue > 0, net_income / revenue, 0.0)

    # Project (unlevered) cash flows
    project_cf = np.concatenate(([-total_capex], fcff_unlevered))
    project_irr = nf.irr(project_cf)
    project_npv = sum(cf / ((1 + wacc) ** t) for t, cf in enumerate(project_cf))
    project_payback = compute_payback(project_cf)

    # Equity (levered) cash flows
    equity_cf = np.concatenate(([-equity_amount], fcfe))
    if equity_amount > 0:
        try:
            equity_irr = nf.irr(equity_cf)
        except Exception:
            equity_irr = np.nan
    else:
        equity_irr = np.nan
    equity_payback = compute_payback(equity_cf)

    # Investor cash flows (share of equity CF)
    investor_cf = equity_cf * investor_equity_share
    investor_cf[0] = -investor_equity_investment
    if investor_equity_investment > 0:
        try:
            investor_irr = nf.irr(investor_cf)
        except Exception:
            investor_irr = np.nan
    else:
        investor_irr = np.nan

    years0 = np.arange(0, len(project_cf))
    cum_project_cf = np.cumsum(project_cf)
    cum_equity_cf = np.cumsum(equity_cf)

    # KPIs
    kcol1, kcol2, kcol3 = st.columns(3)
    kcol1.metric("Project IRR (unlevered)", f"{project_irr * 100:,.1f} %")
    kcol2.metric("Project NPV (AED)", f"{project_npv:,.0f}")
    kcol3.metric(
        "Project Payback (years)",
        f"{project_payback:,.2f}" if project_payback is not None else "n/a"
    )

    kcol4, kcol5, kcol6 = st.columns(3)
    kcol4.metric(
        "Equity IRR (levered)",
        f"{equity_irr * 100:,.1f} %" if not np.isnan(equity_irr) else "n/a"
    )
    kcol5.metric(
        "Equity Payback (years)",
        f"{equity_payback:,.2f}" if equity_payback is not None else "n/a"
    )
    if not np.isnan(investor_irr):
        label = "Investor IRR (sweat-equity deal)" if sweat_equity_mode else "Investor IRR"
        kcol6.metric(label, f"{investor_irr * 100:,.1f} %")
    else:
        kcol6.metric("Investor IRR", "n/a")

    if sweat_equity_mode:
        st.info(
            "Sweat-equity partner invests **no cash** and gets their ownership via work. "
            "Their financial IRR is technically infinite / not applicable."
        )

    # Cash flow table Year 0–5
    cash_rows = []

    # Year 0
    cash_rows.append({
        "Year": 0,
        "Net Sales (AED)": 0.0,
        "Total OPEX (AED)": 0.0,
        "EBIT (AED)": 0.0,
        "Interest (AED)": 0.0,
        "Tax (AED)": 0.0,
        "Net Income (AED)": 0.0,
        "Profit Margin (%)": 0.0,
        "Project FCFF (AED)": -total_capex,
        "Equity FCFE (AED)": -equity_amount,
        "Cumulative Project CF (AED)": cum_project_cf[0],
        "Cumulative Equity CF (AED)": cum_equity_cf[0],
    })

    # Years 1–5
    for i, y in enumerate(years):
        cash_rows.append({
            "Year": int(y),
            "Net Sales (AED)": revenue[i],
            "Total OPEX (AED)": opex_total[i],
            "EBIT (AED)": ebit[i],
            "Interest (AED)": interest[i],
            "Tax (AED)": tax[i],
            "Net Income (AED)": net_income[i],
            "Profit Margin (%)": profit_margin[i] * 100,
            "Project FCFF (AED)": fcff_unlevered[i],
            "Equity FCFE (AED)": fcfe[i],
            "Cumulative Project CF (AED)": cum_project_cf[i+1],
            "Cumulative Equity CF (AED)": cum_equity_cf[i+1],
        })

    df_cash = pd.DataFrame(cash_rows)

    cash_style = {
        "Net Sales (AED)": "{:,.0f}",
        "Total OPEX (AED)": "{:,.0f}",
        "EBIT (AED)": "{:,.0f}",
        "Interest (AED)": "{:,.0f}",
        "Tax (AED)": "{:,.0f}",
        "Net Income (AED)": "{:,.0f}",
        "Profit Margin (%)": "{:,.1f}",
        "Project FCFF (AED)": "{:,.0f}",
        "Equity FCFE (AED)": "{:,.0f}",
        "Cumulative Project CF (AED)": "{:,.0f}",
        "Cumulative Equity CF (AED)": "{:,.0f}",
    }

    st.dataframe(
        df_cash.style.format(cash_style),
        use_container_width=True
    )

# ==============================
# TAB 2 – RECURRENT CUSTOMERS
# ==============================
with tab2:
    st.caption("📄 To export this tab to PDF: use your browser's **Print → Save as PDF**.")
    st.header("Recurrent Customer Requirement")

    st.markdown(
        "Estimate how many **recurring clients** you need "
        "if each attends X classes/week at Y AED per class."
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        target_basis = st.selectbox("Target Revenue Basis", ["Monthly", "Yearly"], index=1)
        default_annual_target = 2_396_000.0  # e.g. group class revenue level

        if target_basis == "Yearly":
            target_revenue_year = st.number_input(
                "Target Yearly Revenue (AED)",
                value=float(default_annual_target),
                step=50_000.0
            )
        else:
            default_monthly_target = default_annual_target / 12.0
            target_revenue_month = st.number_input(
                "Target Monthly Revenue (AED)",
                value=float(round(default_monthly_target, -2)),
                step=5_000.0
            )
            target_revenue_year = target_revenue_month * 12.0

    with col_b:
        classes_per_week = st.slider("Classes per Week per Customer", 1, 7, 2)
        weeks_per_year = st.number_input("Active Weeks per Year", 1, 52, 48)

    with col_c:
        price_per_class = st.number_input(
            "Price per Class (AED)",
            value=150.0,
            step=10.0
        )

    st.subheader("Results")

    if classes_per_week > 0 and weeks_per_year > 0 and price_per_class > 0:
        revenue_per_customer_year = classes_per_week * weeks_per_year * price_per_class
        customers_needed = target_revenue_year / revenue_per_customer_year

        st.markdown(
            f"- **Revenue per customer per year:** `{revenue_per_customer_year:,.0f}` AED  \n"
            f"- **Customers required to hit target:** `{customers_needed:,.1f}` clients  \n"
            f"- Approx. **new recurring clients per month (if built over 1 year):** "
            f"`{customers_needed / 12:,.1f}`"
        )

        st.subheader("Sensitivity Table: Clients Needed vs Class Frequency")
        freq_values = np.arange(1, 7)
        clients_for_freq = [
            target_revenue_year / (f * weeks_per_year * price_per_class)
            for f in freq_values
        ]
        df_freq = pd.DataFrame({
            "Classes per Week per Customer": freq_values,
            "Clients Needed": np.round(clients_for_freq, 1),
        })
        st.dataframe(df_freq, use_container_width=True)

    else:
        st.warning("Please enter positive values for price, weeks, and classes per week.")

# ==============================
# TAB 3 – CAPEX BREAKDOWN
# ==============================
with tab3:
    st.caption("📄 To export this tab to PDF: use your browser's **Print → Save as PDF**.")
    st.header("Capex Breakdown & Optimization (Static from Model)")

    st.markdown(
        "This tab mirrors your **Capex sheet**: categories, quantities, unit prices, "
        "estimated costs, descriptions, and optimization strategies. "
        "Values are static for now (read-only)."
    )

    capex_rows = [
        # --- Overview & Location (no direct cost) ---
        {
            "Category": "Project Overview",
            "Qty": "",
            "Unit Price (AED)": 0,
            "Estimated Cost (AED)": 0,
            "Description": "Boutique Reformer Pilates Studio - 136 m² shell & core",
            "Optimization Strategy": "Focus on premium small-group training with luxury design",
        },
        {
            "Category": "Location",
            "Qty": "",
            "Unit Price (AED)": 0,
            "Estimated Cost (AED)": 0,
            "Description": "Jumeirah Village Circle - Binghatti Royale",
            "Optimization Strategy": "Negotiate rent to ≤ 450K/year and secure high visibility",
        },
        {
            "Category": "Rent Value (Annual)",
            "Qty": "",
            "Unit Price (AED)": 530_000,
            "Estimated Cost (AED)": 530_000,
            "Description": "",
            "Optimization Strategy": "Rent-free period to reduce cash outflow during fit-out",
        },
        {
            "Category": "Lease Term",
            "Qty": "",
            "Unit Price (AED)": 0,
            "Estimated Cost (AED)": 0,
            "Description": "3 years lease, 3 months fit-out grace",
            "Optimization Strategy": "",
        },
        {
            "Category": "Fit-out & MEP",
            "Qty": "",
            "Unit Price (AED)": 550_000,
            "Estimated Cost (AED)": 550_000,
            "Description": "Flooring, partitions, HVAC, ceiling, lighting, plumbing",
            "Optimization Strategy": "Efficient MEP and minimalistic finishes reduce cost",
        },
        {
            "Category": "Joinery & Interior",
            "Qty": "",
            "Unit Price (AED)": 180_000,
            "Estimated Cost (AED)": 180_000,
            "Description": "Reception desk, lockers, mirrors, shelving",
            "Optimization Strategy": "Modular cabinetry from local joiner",
        },

        # --- Equipment ---
        {
            "Category": "Equipment – Reformer machine",
            "Qty": "9",
            "Unit Price (AED)": 9_950,
            "Estimated Cost (AED)": 89_550,
            "Description": "Align Pilates R8 – pre-order 40–60 days",
            "Optimization Strategy": "Local suppliers or refurbished equipment",
        },
        {
            "Category": "Equipment – Ladder barrel",
            "Qty": "1",
            "Unit Price (AED)": 7_395,
            "Estimated Cost (AED)": 7_395,
            "Description": "",
            "Optimization Strategy": "",
        },
        {
            "Category": "Equipment – Trapeze table",
            "Qty": "1",
            "Unit Price (AED)": 30_000,
            "Estimated Cost (AED)": 30_000,
            "Description": "",
            "Optimization Strategy": "",
        },
        {
            "Category": "Equipment – Props, mats, weights",
            "Qty": "LS",
            "Unit Price (AED)": 4_440,
            "Estimated Cost (AED)": 4_440,
            "Description": "",
            "Optimization Strategy": "",
        },
        {
            "Category": "Equipment – Café setup",
            "Qty": "1",
            "Unit Price (AED)": 50_000,
            "Estimated Cost (AED)": 50_000,
            "Description": "",
            "Optimization Strategy": "",
        },

        # --- Other Capex ---
        {
            "Category": "Sound & Lighting",
            "Qty": "LS",
            "Unit Price (AED)": 40_000,
            "Estimated Cost (AED)": 40_000,
            "Description": "Acoustic + dimmable lighting system",
            "Optimization Strategy": "Wellness-focused atmosphere",
        },
        {
            "Category": "Software & Tech",
            "Qty": "LS",
            "Unit Price (AED)": 30_000,
            "Estimated Cost (AED)": 30_000,
            "Description": "Booking, POS, CCTV, and accounting software",
            "Optimization Strategy": "Bundle tech tools to minimize monthly fees",
        },
        {
            "Category": "Branding & Launch",
            "Qty": "LS",
            "Unit Price (AED)": 50_000,
            "Estimated Cost (AED)": 50_000,
            "Description": "Logo, website, photoshoot, influencer launch",
            "Optimization Strategy": "Organic PR instead of paid campaigns",
        },
        {
            "Category": "Licenses & Permits",
            "Qty": "LS",
            "Unit Price (AED)": 25_000,
            "Estimated Cost (AED)": 25_000,
            "Description": "Trade license, DCD/DM approvals, Ejari",
            "Optimization Strategy": "Plan license and approvals early to avoid delay",
        },

        # --- Pre-opening & working capital (3 months) ---
        {
            "Category": "Instructors (3 months reserve – 5 instructors)",
            "Qty": "768",
            "Unit Price (AED)": 150,
            "Estimated Cost (AED)": 115_200,
            "Description": "3 months operations reserve (salaries, utilities, marketing buffer)",
            "Optimization Strategy": "Maintain cash for early-stage operations",
        },
        {
            "Category": "Utilities Buffer (3 months)",
            "Qty": "3",
            "Unit Price (AED)": 1_500,
            "Estimated Cost (AED)": 4_500,
            "Description": "",
            "Optimization Strategy": "",
        },
        {
            "Category": "Marketing Buffer (3 months)",
            "Qty": "3",
            "Unit Price (AED)": 2_000,
            "Estimated Cost (AED)": 6_000,
            "Description": "",
            "Optimization Strategy": "",
        },
    ]

    df_capex = pd.DataFrame(capex_rows)

    total_investment = df_capex["Estimated Cost (AED)"].sum()

    st.dataframe(
        df_capex.style.format({
            "Unit Price (AED)": "{:,.0f}",
            "Estimated Cost (AED)": "{:,.0f}",
        }),
        use_container_width=True
    )

    st.markdown(
        f"### Total Investment (Capex + Pre-opening) : **AED {total_investment:,.0f}**"
    )
