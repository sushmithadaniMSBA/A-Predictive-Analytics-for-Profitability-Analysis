A Predictive Analytics for Profitability Analytics

This capstone project develops a hybrid analytics framework to generate actionable product recommendations for multi shelf-life inventory. The framework integrates product movement forecasting, upsell modeling, and cross-sell association mining to optimize inventory decisions and drive profitability.

Project Overview
Movement Forecasting: A two-stage Random Forest classifier predicts product movement categories (To-Zero, Drop, Steady, Rise).
Upsell Modeling: Customer–product level model estimates incremental purchase quantities using features such as recency, frequency, and rolling averages.
Cross-Sell Associations: FP-Growth algorithm identifies statistically significant product basket associations for cross-sell recommendations.
Hybrid Recommendation Layer: Combines movement, upsell, and cross-sell insights into a single decision table csv  with business-ready actions (e.g., replenish, phase-out, upsell, cross-sell).
The results are deployed through a Streamlit-based app that allows users to:
Upload feature-engineered monthly sales data.
View movement, upsell, and cross-sell recommendations.
Compare baseline vs. uplifted revenue scenarios.
Download final recommendation tables for business use.

How to Run
Install dependencies: streamlit, pandas, scikit-learn, mlxtend, joblib, matplotlib, seaborn.

Run the app:
streamlit run app.py
Upload the monthly Master data Excel/CSV file to view recommendations and profitability analysis.

Demo Video
Watch the demo
 
This video shows how to use the Streamlit app for product recommendation and profitability analysis.

Done by
Sushmitha Dani
MS Business Analytics
Capstone Project 2 – A Predictive Analytics for Profitability Analysis
REVA University, 2025
