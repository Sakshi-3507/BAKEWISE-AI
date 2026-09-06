import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from streamlit_option_menu import option_menu

# ==================================================================
# PAGE SETUP + BRAND STYLING
# ==================================================================
st.set_page_config(page_title="BakeWise AI", page_icon="🍰", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: linear-gradient(180deg, #FCF3E7 0%, #FBF1E1 55%, #FAEEDC 100%); }

h1, h2, h3, .bw-serif {
    font-family: 'Fraunces', serif !important;
    color: #3B2A20;
}

/* Hero */
.bw-hero-wrap {
    background: linear-gradient(120deg, #FFF6E9 0%, #FCEBD9 55%, #F7DCC3 100%);
    border-radius: 20px;
    padding: 34px 38px;
    box-shadow: 0 8px 28px rgba(168, 80, 58, 0.10);
    border: 1px solid #F0DCC0;
    position: relative;
    overflow: hidden;
}
.bw-hero-wrap::after {
    content: "🍰";
    position: absolute;
    right: 28px;
    top: 50%;
    transform: translateY(-50%) rotate(-8deg);
    font-size: 5.5rem;
    opacity: 0.16;
}
.bw-hero-title {
    font-family: 'Fraunces', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: #3B2A20;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.bw-hero-sub {
    font-size: 1.05rem;
    color: #6B5A4E;
    max-width: 560px;
    margin-top: 0.4rem;
}

/* Stat strip -> soft cards */
.bw-stat-strip {
    display: flex;
    gap: 16px;
    margin: 26px 0 8px 0;
}
.bw-stat {
    flex: 1;
    background: #FFFDF8;
    border: 1px solid #EFE3D0;
    border-radius: 14px;
    padding: 18px 20px;
    box-shadow: 0 3px 12px rgba(59, 42, 32, 0.05);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.bw-stat:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 20px rgba(168, 80, 58, 0.12);
}
.bw-stat-num {
    font-family: 'Fraunces', serif;
    font-size: 2rem;
    font-weight: 600;
    color: #A8503A;
}
.bw-stat-label {
    font-size: 0.85rem;
    color: #6B5A4E;
}

/* Feature list -> gentle cards */
.bw-feature {
    display: flex;
    align-items: flex-start;
    gap: 14px;
    padding: 16px 18px;
    margin-bottom: 10px;
    background: #FFFDF8;
    border: 1px solid #EFE3D0;
    border-radius: 12px;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.bw-feature:hover {
    transform: translateX(3px);
    box-shadow: 0 6px 16px rgba(59, 42, 32, 0.07);
}
.bw-feature-icon { font-size: 1.5rem; }
.bw-feature-title { font-weight: 600; color: #3B2A20; margin-bottom: 2px; }
.bw-feature-desc { font-size: 0.9rem; color: #6B5A4E; }

/* Result panel */
.bw-panel {
    background: #FFFDF8;
    border: 1px solid #EFE3D0;
    border-left: 5px solid #A8503A;
    border-radius: 12px;
    padding: 22px 26px;
    margin-bottom: 14px;
    box-shadow: 0 6px 18px rgba(59, 42, 32, 0.06);
}
.bw-panel-profit { border-left-color: #6E8B63; }
.bw-panel-waste { border-left-color: #9C3D46; }
.bw-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    background: #C99A3B22;
    color: #A8763B;
}

/* Flavour chips */
.bw-chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.bw-chip {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 500;
    background: #F7E3C6;
    color: #7A4E22;
    border: 1px solid #EAD1A3;
}
.bw-chip-egg { background: #E9F1E4; color: #4C6E43; border-color: #D2E4C9; }
.bw-chip-eggless { background: #FDE9E9; color: #9C3D46; border-color: #F5CFCF; }

.bw-footer {
    text-align: center;
    color: #9C8A78;
    font-size: 0.8rem;
    padding: 18px 0 6px 0;
    border-top: 1px solid #E4D5BE;
    margin-top: 24px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #F6E8CF 0%, #F1E1C6 100%);
}
div[data-testid="stMetric"] {
    background: #FFFDF8;
    border: 1px solid #EFE3D0;
    border-radius: 12px;
    padding: 10px 14px;
    box-shadow: 0 3px 10px rgba(59, 42, 32, 0.05);
}
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
    border-radius: 12px !important;
    overflow: hidden;
    border: 1px solid #EFE3D0;
    box-shadow: 0 4px 14px rgba(59, 42, 32, 0.06);
}
div[data-testid="stExpander"] {
    background: #FFFDF8;
    border: 1px solid #EFE3D0 !important;
    border-radius: 12px !important;
    box-shadow: 0 3px 10px rgba(59, 42, 32, 0.05);
}
div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* ---- Aesthetic animations ---- */
@keyframes bwFadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes bwFadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes bwFloat {
    0%, 100% { transform: translateY(-50%) rotate(-8deg); }
    50%      { transform: translateY(-58%) rotate(-3deg); }
}

.bw-hero-wrap {
    animation: bwFadeInUp 0.7s ease both;
}
.bw-hero-wrap::after {
    animation: bwFloat 3.5s ease-in-out infinite;
}
.bw-stat-strip .bw-stat {
    animation: bwFadeInUp 0.6s ease both;
}
.bw-stat-strip .bw-stat:nth-child(1) { animation-delay: 0.05s; }
.bw-stat-strip .bw-stat:nth-child(2) { animation-delay: 0.12s; }
.bw-stat-strip .bw-stat:nth-child(3) { animation-delay: 0.19s; }
.bw-stat-strip .bw-stat:nth-child(4) { animation-delay: 0.26s; }

.bw-feature { animation: bwFadeInUp 0.5s ease both; }
.bw-panel { animation: bwFadeInUp 0.5s ease both; }

div[data-testid="stMetric"] {
    animation: bwFadeInUp 0.45s ease both;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 18px rgba(168, 80, 58, 0.12);
}
div[data-testid="stDataFrame"], div[data-testid="stTable"], div[data-testid="stExpander"] {
    animation: bwFadeIn 0.5s ease both;
}
[data-testid="stMarkdownContainer"] > p, .main .block-container h2, .main .block-container h3 {
    animation: bwFadeIn 0.5s ease both;
}
button[kind="primary"] {
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
}
button[kind="primary"]:hover {
    transform: translateY(-2px) scale(1.01);
    box-shadow: 0 8px 18px rgba(168, 80, 58, 0.25) !important;
}
</style>
""", unsafe_allow_html=True)


def show_footer():
    st.markdown(
        '<div class="bw-footer">🍰 BakeWise AI — Demand Forecasting · Cost Intelligence · Customer Insight</div>',
        unsafe_allow_html=True
    )


# ==================================================================
# DATA + RECIPES
# ==================================================================
PRODUCTS = ["Cake", "Cupcake", "Jar Cake", "Chocolate", "Brownie", "Croissant", "Cookies", "Muffin"]

# approx finished weight of ONE unit — so "1 unit" has a clear, real-world size
PRODUCT_QUANTITY = {
    "Cake": "1 kg",
    "Cupcake": "50 g",
    "Jar Cake": "200 g",
    "Chocolate": "30 g",
    "Brownie": "120 g",
    "Croissant": "150 g",
    "Cookies": "50 g",
    "Muffin": "50 g",
}

# fixed menu selling price per unit (as priced on the counter)
SELLING_PRICES = {
    "Cake": 800,
    "Cupcake": 65,
    "Jar Cake": 100,
    "Chocolate": 60,
    "Brownie": 150,
    "Croissant": 200,
    "Cookies": 70,
    "Muffin": 40,
}

# flavours on offer for each product
FLAVOURS = {
    "Cake": ["Chocolate Truffle", "Red Velvet", "Butterscotch", "Black Forest", "Vanilla", "Pineapple"],
    "Cupcake": ["Chocolate", "Vanilla", "Red Velvet", "Butterscotch"],
    "Jar Cake": ["Choco Overload", "Biscoff", "Mango", "Butterscotch"],
    "Chocolate": ["Dark Chocolate", "Milk Chocolate", "White Chocolate", "Dry Fruit Chocolate"],
    "Brownie": ["Classic Fudge", "Walnut", "Nutella", "Choco Chip"],
    "Croissant": ["Plain Butter", "Chocolate Filled", "Almond"],
    "Cookies": ["Choco Chip", "Double Chocolate", "Oatmeal", "Butter"],
    "Muffin": ["Blueberry", "Chocolate Chip", "Banana Walnut", "Vanilla"],
}

EGG_SUBSTITUTE_PRICE_PER_EGG = 3.0  # ₹ per egg-equivalent, using curd/flax substitute

# grams(kg)/pieces of each ingredient needed to make ONE unit of the product
RECIPES = {
    "Cake":       {"Flour": 0.30,  "Butter": 0.15,  "Sugar": 0.20,  "Eggs": 3.0,  "Cocoa": 0.05},
    "Cupcake":    {"Flour": 0.017, "Butter": 0.010, "Sugar": 0.013, "Eggs": 0.17, "Cocoa": 0.003},
    "Jar Cake":   {"Flour": 0.064, "Butter": 0.040, "Sugar": 0.048, "Eggs": 0.4,  "Cocoa": 0.024},
    "Chocolate":  {"Flour": 0.00,  "Butter": 0.005, "Sugar": 0.01,  "Eggs": 0.0,  "Cocoa": 0.015},
    "Brownie":    {"Flour": 0.03,  "Butter": 0.02,  "Sugar": 0.03,  "Eggs": 0.2,  "Cocoa": 0.03},
    "Croissant":  {"Flour": 0.078, "Butter": 0.046, "Sugar": 0.013, "Eggs": 0.26, "Cocoa": 0.00},
    "Cookies":    {"Flour": 0.017, "Butter": 0.0125,"Sugar": 0.0125,"Eggs": 0.083,"Cocoa": 0.004},
    "Muffin":     {"Flour": 0.019, "Butter": 0.011, "Sugar": 0.015, "Eggs": 0.11, "Cocoa": 0.00},
}

# egg vs eggless is offered wherever the recipe actually uses eggs
EGG_ELIGIBLE = [p for p in PRODUCTS if RECIPES[p]["Eggs"] > 0]


@st.cache_data
def load_data():
    sales = pd.read_csv("bakery_daily_sales.csv", parse_dates=["Date"])
    prices = pd.read_csv("ingredient_prices.csv", parse_dates=["Date"])
    customers = pd.read_csv("customer_preferences.csv")

    sales["Rainfall"] = sales["Rainfall"].astype(int)
    sales["Is_Weekend"] = sales["Is_Weekend"].astype(int)
    sales["Is_Festival"] = sales["Is_Festival"].astype(int)

    day_encoder = LabelEncoder()
    product_encoder = LabelEncoder()
    sales["Day_Code"] = day_encoder.fit_transform(sales["Day_of_Week"])
    sales["Product_Code"] = product_encoder.fit_transform(sales["Product"])

    # Butter in the raw CSV came in at an unrealistically high ₹/kg — rescale
    # it so today's rate sits around a realistic ₹150/kg, keeping the same
    # day-to-day up/down fluctuation shape.
    butter_mask = prices["Ingredient"] == "Butter"
    if butter_mask.any():
        prices_sorted = prices.sort_values("Date")
        current_butter = prices_sorted.loc[prices_sorted["Ingredient"] == "Butter", "Price_Per_Unit"].iloc[-1]
        if current_butter and current_butter > 0:
            scale = 150.0 / current_butter
            prices.loc[butter_mask, "Price_Per_Unit"] = prices.loc[butter_mask, "Price_Per_Unit"] * scale

    return sales, prices, customers, day_encoder, product_encoder


@st.cache_resource
def train_model(sales):
    features = ["Day_Code", "Is_Weekend", "Temperature_C", "Rainfall", "Is_Festival", "Product_Code"]
    X = sales[features]
    y = sales["Units_Sold"]
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(xtrain, ytrain)
    preds = model.predict(xtest)

    mae = mean_absolute_error(ytest, preds)
    r2 = r2_score(ytest, preds)
    return model, features, mae, r2


sales, prices, customers, day_encoder, product_encoder = load_data()
model, feature_cols, mae, r2 = train_model(sales)

latest_prices = prices.sort_values("Date").groupby("Ingredient").last()["Price_Per_Unit"].to_dict()


def unit_cost(product, eggless=False):
    """Cost per unit. If eggless=True and the recipe uses eggs, eggs are
    swapped for a curd/flax substitute at EGG_SUBSTITUTE_PRICE_PER_EGG."""
    recipe = RECIPES[product]
    total = 0.0
    for ing, qty in recipe.items():
        if ing == "Eggs" and eggless:
            total += qty * EGG_SUBSTITUTE_PRICE_PER_EGG
        else:
            total += qty * latest_prices[ing]
    return total


with st.sidebar:
    st.markdown('<div class="bw-serif" style="font-size:1.4rem; margin-bottom:6px;">🍰 BakeWise AI</div>', unsafe_allow_html=True)
    selected = option_menu(
        None,
        options=["Overview", "Sales Insights", "Ingredient Costs", "Customer Preferences", "Tomorrow's Bake Plan"],
        icons=["house", "graph-up", "cash-coin", "people", "stars"],
        default_index=0,
        styles={
            "container": {"background-color": "#F3E6D0"},
            "icon": {"color": "#A8503A"},
            "nav-link": {"color": "#3B2A20", "font-size": "14px"},
            "nav-link-selected": {"background-color": "#A8503A", "color": "white"},
        }
    )



# ==================================================================
# OVERVIEW
# ==================================================================
if selected == "Overview":
    st.markdown(
        '<div class="bw-hero-wrap">'
        '<div class="bw-hero-title">🍰 BakeWise AI</div>'
        '<div class="bw-hero-sub">Bake exactly what tomorrow needs. BakeWise reads weather, weekday '
        'patterns, festivals, and ingredient prices to tell you how much to bake, what it will cost, '
        'and who\'s actually going to buy it.</div>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(f"""
    <div class="bw-stat-strip">
        <div class="bw-stat"><div class="bw-stat-num">{sales['Units_Wasted'].sum() / sales['Units_Sold'].sum() * 100:.1f}%</div>
            <div class="bw-stat-label">Historical waste rate</div></div>
        <div class="bw-stat"><div class="bw-stat-num">{r2*100:.0f}%</div>
            <div class="bw-stat-label">Forecast R² accuracy</div></div>
        <div class="bw-stat"><div class="bw-stat-num">{len(PRODUCTS)}</div>
            <div class="bw-stat-label">Products tracked</div></div>
        <div class="bw-stat"><div class="bw-stat-num">{customers.shape[0]}</div>
            <div class="bw-stat-label">Customers profiled</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.markdown("#### How it fits together")

    features = [
        ("📈", "Demand Forecasting", "A Random Forest model learns from day-of-week, weather, and festival patterns to predict tomorrow's sales, per product."),
        ("💰", "Cost & Profit Intelligence", "Flour, butter, sugar, eggs, and cocoa prices shift daily — the plan recalculates true cost per unit before suggesting a bake quantity."),
        ("👥", "Customer Preference Signal", "500 profiled customers reveal which products Loyal, Regular, and New customers actually prefer — so popular items aren't underbaked."),
        ("🧾", "One Combined Bake Plan", "All three signals merge into a single daily plan: how much to bake, what it costs, what it earns, and who it's for."),
    ]
    for icon, title, desc in features:
        st.markdown(f"""
        <div class="bw-feature">
            <div class="bw-feature-icon">{icon}</div>
            <div>
                <div class="bw-feature-title">{title}</div>
                <div class="bw-feature-desc">{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    st.write("")
    show_footer()

# ==================================================================
# SALES INSIGHTS
# ==================================================================
if selected == "Sales Insights":
    st.markdown('<h2 class="bw-serif">Sales Insights</h2>', unsafe_allow_html=True)
    st.caption("How demand actually moves — by product, weekday, weather, and festivals.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Units Sold", f"{sales['Units_Sold'].sum():,}")
    c2.metric("Total Units Wasted", f"{sales['Units_Wasted'].sum():,}")
    c3.metric("Avg Daily Temp", f"{sales['Temperature_C'].mean():.1f}°C")
    c4.metric("Festival Days Logged", int(sales['Is_Festival'].sum() / len(PRODUCTS)))

    st.write("")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Units Sold by Product**")
        product_totals = sales.groupby("Product")["Units_Sold"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.barh(product_totals.index, product_totals.values, color="#A8503A")
        ax.invert_yaxis()
        ax.set_xlabel("Total Units Sold (2 years)")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig); plt.close(fig)
    with right:
        st.markdown("**Weekday vs Weekend**")
        wk = sales.groupby("Is_Weekend")["Units_Sold"].mean()
        fig, ax = plt.subplots(figsize=(4, 3.5))
        ax.bar(["Weekday", "Weekend"], wk.values, color=["#C99A3B", "#A8503A"], width=0.5)
        ax.set_ylabel("Avg units/day")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig); plt.close(fig)

    st.write("")
    st.markdown("**Monthly Demand Trend — by Product**")
    st.caption("Each line is one product's total units sold per month, so you can see who's rising and who's dipping.")
    monthly = sales.copy()
    monthly["Month"] = monthly["Date"].dt.to_period("M").astype(str)
    monthly_by_product = monthly.groupby(["Month", "Product"])["Units_Sold"].sum().unstack(fill_value=0)
    monthly_by_product = monthly_by_product.sort_index()

    product_colors = {
        "Cake": "#A8503A", "Cupcake": "#C99A3B", "Jar Cake": "#6E8B63",
        "Chocolate": "#3B2A20", "Brownie": "#9C3D46", "Croissant": "#D9A441",
        "Cookies": "#7A5C3E", "Muffin": "#5C7A6B",
    }
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x_positions = range(len(monthly_by_product))
    for product in PRODUCTS:
        if product in monthly_by_product.columns:
            ax.plot(x_positions, monthly_by_product[product].values,
                    label=product, color=product_colors.get(product, "#A8503A"), linewidth=1.8)
    ax.set_xticks(list(x_positions)[::3])
    ax.set_xticklabels(monthly_by_product.index[::3], rotation=45, ha="right")
    ax.set_ylabel("Units Sold")
    ax.set_title("Monthly units sold, per product", fontsize=11, color="#3B2A20", loc="left")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.28),
        ncol=4, frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.4
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.write("")
    st.markdown("**Waste by Product**")
    waste_pct = (sales.groupby("Product")["Units_Wasted"].sum() / sales.groupby("Product")["Units_Sold"].sum() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(waste_pct.index, waste_pct.values, color="#9C3D46")
    ax.set_ylabel("Waste %")
    plt.xticks(rotation=20, ha="right")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    st.pyplot(fig); plt.close(fig)

    st.write("")
    show_footer()

# ==================================================================
# INGREDIENT COSTS
# ==================================================================
if selected == "Ingredient Costs":
    st.markdown('<h2 class="bw-serif">Ingredient Costs</h2>', unsafe_allow_html=True)
    st.caption("Daily fluctuating prices — the hidden variable that quietly eats into margins.")

    c1, c2, c3, c4, c5 = st.columns(5)
    unit_labels = {"Flour": "₹/kg", "Butter": "₹/kg", "Sugar": "₹/kg", "Eggs": "₹/egg", "Cocoa": "₹/kg"}
    for col, ing in zip([c1, c2, c3, c4, c5], ["Flour", "Butter", "Sugar", "Eggs", "Cocoa"]):
        col.metric(f"{ing} ({unit_labels[ing]})", f"₹{latest_prices[ing]:.2f}")

    st.write("")
    st.markdown("**Price Trend (last 90 days)**")
    recent = prices[prices["Date"] >= prices["Date"].max() - pd.Timedelta(days=90)]
    fig, ax = plt.subplots(figsize=(11, 4.6))
    colors_map = {"Flour": "#C99A3B", "Butter": "#A8503A", "Sugar": "#6E8B63", "Eggs": "#9C3D46", "Cocoa": "#3B2A20"}
    for ing in ["Flour", "Butter", "Sugar", "Eggs", "Cocoa"]:
        sub = recent[recent["Ingredient"] == ing]
        ax.plot(sub["Date"], sub["Price_Per_Unit"], label=ing, color=colors_map[ing], linewidth=2.0)
    ax.set_ylabel("Price (₹ per unit)")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=5, frameon=False, fontsize=9, handlelength=1.6, columnspacing=1.4
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.write("")
    st.markdown("**Cost Per Unit by Product** (at today's ingredient prices)")
    cost_data = pd.DataFrame({"Product": PRODUCTS, "Cost": [unit_cost(p) for p in PRODUCTS]}).sort_values("Cost", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.barh(cost_data["Product"], cost_data["Cost"], color="#C99A3B")
    ax.invert_yaxis()
    ax.set_xlabel("Cost per unit (₹)")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    st.pyplot(fig); plt.close(fig)

    st.write("")
    st.markdown("**📋 Product Reference — quantity, cost, price, and margin per unit**")
    st.caption("So '1 unit' has a clear real-world size, not just a number. Menu prices are fixed; cost moves with today's ingredient rates.")
    ref_rows = []
    for product in PRODUCTS:
        price = SELLING_PRICES[product]
        cost_egg = unit_cost(product, eggless=False)
        has_egg = product in EGG_ELIGIBLE
        cost_eggless = unit_cost(product, eggless=True) if has_egg else cost_egg
        margin_egg = (price - cost_egg) / price * 100
        margin_eggless = (price - cost_eggless) / price * 100
        ref_rows.append({
            "Product": product,
            "Quantity (1 unit)": PRODUCT_QUANTITY[product],
            "Egg Option": "Egg / Eggless" if has_egg else "—",
            "Cost/Unit (Egg)": round(cost_egg, 1),
            "Cost/Unit (Eggless)": round(cost_eggless, 1) if has_egg else None,
            "Selling Price": price,
            "Margin % (Egg)": round(margin_egg, 1),
            "Margin % (Eggless)": round(margin_eggless, 1) if has_egg else None,
        })
    ref_df = pd.DataFrame(ref_rows)
    st.dataframe(
        ref_df, use_container_width=True, hide_index=True,
        column_config={
            "Cost/Unit (Egg)": st.column_config.NumberColumn("Cost/Unit (Egg)", format="₹%.1f"),
            "Cost/Unit (Eggless)": st.column_config.NumberColumn("Cost/Unit (Eggless)", format="₹%.1f"),
            "Selling Price": st.column_config.NumberColumn("Selling Price", format="₹%d"),
            "Margin % (Egg)": st.column_config.NumberColumn("Margin % (Egg)", format="%.1f%%"),
            "Margin % (Eggless)": st.column_config.NumberColumn("Margin % (Eggless)", format="%.1f%%"),
        }
    )

    st.write("")
    show_footer()

# ==================================================================
# CUSTOMER PREFERENCES
# ==================================================================
if selected == "Customer Preferences":
    st.markdown('<h2 class="bw-serif">Customer Preferences</h2>', unsafe_allow_html=True)
    st.caption("Who's buying, and what they actually want — so popular items don't run out.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Loyal Customers", (customers["Customer_Type"] == "Loyal").sum())
    c2.metric("Regular Customers", (customers["Customer_Type"] == "Regular").sum())
    c3.metric("New Customers", (customers["Customer_Type"] == "New").sum())

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown("**Preferred Product by Customer Type**")
        pref = customers.groupby(["Preferred_Product", "Customer_Type"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        pref.plot(kind="barh", stacked=True, ax=ax, color=["#C99A3B", "#A8503A", "#6E8B63"])
        ax.set_xlabel("Number of customers")
        ax.legend(title="", frameon=False)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig); plt.close(fig)
    with right:
        st.markdown("**Preference by Age Group**")
        age_pref = customers.groupby(["Age_Group", "Preferred_Product"]).size().unstack(fill_value=0)
        fig, ax = plt.subplots(figsize=(6, 4))
        age_pref.plot(kind="bar", stacked=True, ax=ax, colormap="copper")
        plt.xticks(rotation=20, ha="right")
        ax.legend(fontsize=7, frameon=False, bbox_to_anchor=(1, 1))
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig); plt.close(fig)

    st.write("")
    st.markdown("**At-Risk Customers** — no order in 60+ days")
    at_risk = customers[customers["Days_Since_Last_Order"] > 60]
    st.dataframe(at_risk[["Customer_ID", "Customer_Type", "Preferred_Product", "Days_Since_Last_Order"]], use_container_width=True)

    st.write("")
    show_footer()

# ==================================================================
# TOMORROW'S BAKE PLAN
# ==================================================================
if selected == "Tomorrow's Bake Plan":
    st.markdown('<h2 class="bw-serif">Tomorrow\'s Bake Plan</h2>', unsafe_allow_html=True)
    st.caption(f"Powered by a Random Forest model — MAE {mae:.1f} units, R² {r2:.2f}")

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        plan_date = st.date_input("Date", datetime.today() + timedelta(days=1), min_value=datetime.today() + timedelta(days=1))
        day_name = plan_date.strftime("%A")
        is_weekend = 1 if day_name in ["Saturday", "Sunday"] else 0
        st.caption(f"Day: {day_name}")
    with c2:
        temperature = st.slider("Expected Temperature (°C)", 10, 45, 28)
        rainfall = st.checkbox("Rain expected")
    with c3:
        festival = st.checkbox("Festival / Special day")

    st.write("")
    st.markdown("**🧭 Product Types** — pick what you want a plan for tomorrow")
    selected_products = st.multiselect(
        "Product types to include",
        PRODUCTS,
        default=PRODUCTS,
        label_visibility="collapsed",
    )

    flavour_pref = {}
    egg_pref = {}

    if selected_products:
        with st.expander("🎨 Flavours & Egg/Eggless — customize each product", expanded=True):
            for product in selected_products:
                st.markdown(f"**{product}** · {PRODUCT_QUANTITY[product]}")
                fcol, ecol = st.columns([2, 1])
                if product not in EGG_ELIGIBLE:
                    ecol = None
                with fcol:
                    flavour_pref[product] = st.multiselect(
                        f"Flavours — {product}", FLAVOURS[product],
                        default=FLAVOURS[product][:2],
                        key=f"flavour_{product}", label_visibility="collapsed",
                    )
                if ecol is not None:
                    with ecol:
                        egg_pref[product] = st.radio(
                            "Egg type", ["Egg", "Eggless"],
                            key=f"egg_{product}", label_visibility="collapsed",
                        )
                st.write("")
    else:
        st.warning("Kam se kam ek product type select karo plan banane ke liye.")

    st.write("")
    if st.button("🧁 Generate Bake Plan", type="primary", disabled=not selected_products):
        day_code = day_encoder.transform([day_name])[0]

        plan_rows = []
        for product in selected_products:
            product_code = product_encoder.transform([product])[0]
            row = pd.DataFrame([[day_code, is_weekend, temperature, int(rainfall), int(festival), product_code]], columns=feature_cols)
            predicted_units = max(0, int(round(model.predict(row)[0])))

            is_eggless = product in EGG_ELIGIBLE and egg_pref.get(product) == "Eggless"
            cost = unit_cost(product, eggless=is_eggless)
            price = SELLING_PRICES[product]
            total_cost = cost * predicted_units
            total_revenue = price * predicted_units
            profit = total_revenue - total_cost

            top_type = customers[customers["Preferred_Product"] == product]["Customer_Type"].value_counts()
            top_type_str = top_type.index[0] if len(top_type) else "N/A"

            egg_label = ("Eggless" if is_eggless else "Egg") if product in EGG_ELIGIBLE else "—"
            chosen_flavours = flavour_pref.get(product) or []
            flavour_str = ", ".join(chosen_flavours) if chosen_flavours else "Not specified"

            plan_rows.append({
                "Product": product,
                "Unit Size": PRODUCT_QUANTITY[product],
                "Egg Type": egg_label,
                "Bake Quantity": predicted_units,
                "Cost/Unit": round(cost, 1),
                "Price/Unit": price,
                "Total Cost": round(total_cost, 1),
                "Total Revenue": round(total_revenue, 1),
                "Profit": round(profit, 1),
                "Top Fanbase": top_type_str,
                "Selected Flavours": flavour_str,
            })

        plan_df = pd.DataFrame(plan_rows).sort_values("Profit", ascending=False)
        total_profit = plan_df["Profit"].sum()
        total_units = plan_df["Bake Quantity"].sum()

        st.balloons()

        st.markdown(f"""
        <div class="bw-panel bw-panel-profit">
            <span class="bw-badge">Plan for {plan_date.strftime('%d %b %Y')} ({day_name})</span>
            <h3 class="bw-serif" style="margin:10px 0 4px 0;">Bake {total_units} units today, expect ₹{total_profit:,.0f} profit</h3>
            <div style="color:#6B5A4E; font-size:0.9rem;">Based on {temperature}°C weather, {"rain" if rainfall else "clear skies"}{", and a festival day" if festival else ""}.</div>
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            plan_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Cost/Unit": st.column_config.NumberColumn("Cost/Unit", format="₹%.1f"),
                "Price/Unit": st.column_config.NumberColumn("Price/Unit", format="₹%d"),
                "Total Cost": st.column_config.NumberColumn("Total Cost", format="₹%.1f"),
                "Total Revenue": st.column_config.NumberColumn("Total Revenue", format="₹%.1f"),
                "Profit": st.column_config.NumberColumn("Profit", format="₹%.1f"),
            }
        )

        st.write("")
        left, right = st.columns(2)
        with left:
            st.markdown("**Recommended Quantities**")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(plan_df["Product"], plan_df["Bake Quantity"], color="#A8503A")
            ax.invert_yaxis()
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            st.pyplot(fig); plt.close(fig)
        with right:
            st.markdown("**Profit Contribution**")
            fig, ax = plt.subplots(figsize=(6, 4))
            colors_bar = ["#6E8B63" if p >= 0 else "#9C3D46" for p in plan_df["Profit"]]
            ax.barh(plan_df["Product"], plan_df["Profit"], color=colors_bar)
            ax.invert_yaxis()
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            st.pyplot(fig); plt.close(fig)

        st.write("")
        csv_data = plan_df.to_csv(index=False)
        st.download_button("⬇️ Download Bake Plan (.csv)", csv_data, file_name=f"bake_plan_{plan_date}.csv", mime="text/csv")

    st.write("")
    show_footer()