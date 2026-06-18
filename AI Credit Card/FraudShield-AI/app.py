import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime

st.set_page_config(page_title="FraudShield AI", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:#0b0f1a;color:#e2e8f0;}
[data-testid="stSidebar"]{background:#111827!important;border-right:1px solid #1e2d3d;}
.hero{background:linear-gradient(135deg,#0f172a 0%,#1e293b 50%,#0f172a 100%);border:1px solid #1e2d3d;border-radius:16px;padding:32px 36px;margin-bottom:28px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:220px;height:220px;background:radial-gradient(circle,rgba(56,189,248,0.12) 0%,transparent 70%);border-radius:50%;}
.hero-title{font-size:32px;font-weight:700;color:#f1f5f9;letter-spacing:-0.5px;}
.hero-title span{color:#38bdf8;}
.hero-sub{font-size:15px;color:#94a3b8;margin-top:8px;}
.section-header{font-size:12px;font-weight:600;letter-spacing:1.5px;text-transform:uppercase;color:#38bdf8;margin:20px 0 10px;padding-bottom:8px;border-bottom:1px solid #1e2d3d;}
.result-fraud{background:rgba(239,68,68,0.08);border:1px solid rgba(239,68,68,0.35);border-radius:14px;padding:24px 28px;}
.result-legit{background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.35);border-radius:14px;padding:24px 28px;}
.result-title-fraud{font-size:22px;font-weight:700;color:#f87171;}
.result-title-legit{font-size:22px;font-weight:700;color:#4ade80;}
.badge-fraud{display:inline-block;background:rgba(239,68,68,0.15);border:1px solid rgba(239,68,68,0.4);color:#f87171;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;}
.badge-legit{display:inline-block;background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.4);color:#4ade80;padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;}
.warn-box{background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);border-radius:8px;padding:10px 14px;font-size:12px;color:#fbbf24;margin-bottom:12px;}
.stButton>button{background:linear-gradient(135deg,#0ea5e9,#38bdf8);color:#0f172a;font-weight:700;font-size:15px;border:none;border-radius:10px;padding:14px 28px;width:100%;transition:all 0.2s;}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 8px 24px rgba(56,189,248,0.35);}
.stTabs [data-baseweb="tab-list"]{background:#111827;border-radius:10px;border:1px solid #1e2d3d;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{background:transparent;border-radius:7px;color:#94a3b8;font-weight:500;font-size:13px;}
.stTabs [aria-selected="true"]{background:#1e293b!important;color:#f1f5f9!important;}
[data-testid="metric-container"]{background:#111827;border:1px solid #1e2d3d;border-radius:10px;padding:14px 18px;}
[data-testid="metric-container"] label{color:#64748b!important;}
[data-testid="metric-container"] [data-testid="stMetricValue"]{color:#f1f5f9!important;font-family:'JetBrains Mono',monospace;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model    = joblib.load('fraud_model.pkl')
    features = joblib.load('feature_columns.pkl')
    cats     = joblib.load('categories.pkl')
    encoders = joblib.load('label_encoders.pkl')
    metrics  = joblib.load('model_metrics.pkl')
    return model, features, cats, encoders, metrics

@st.cache_data
def load_data():
    import os
    for name in ['powerbi_data.csv', 'powerbi_data_csv.csv', 'powerbi_data_sample.csv', 'powerbi_data_csv']:
        if os.path.exists(name):
            return pd.read_csv(name)
    return pd.DataFrame()

model, FEATURES, CATEGORIES, encoders, metrics = load_assets()
df_all = load_data()

if 'history' not in st.session_state:
    st.session_state.history = []

# ── Sidebar ───────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="font-size:20px;font-weight:700;color:#38bdf8;margin-bottom:4px;">🛡️ FraudShield AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#64748b;font-size:12px;">Random Forest · 1.29M transactions</p>', unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;color:#38bdf8;margin-bottom:10px;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2d3d;border-radius:10px;padding:16px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
            <span style="color:#64748b;font-size:13px;">AUC-ROC</span>
            <span style="color:#38bdf8;font-weight:700;font-family:monospace;">{metrics['auc']*100:.2f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
            <span style="color:#64748b;font-size:13px;">Accuracy</span>
            <span style="color:#4ade80;font-weight:700;font-family:monospace;">{metrics['accuracy']*100:.2f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:10px;">
            <span style="color:#64748b;font-size:13px;">Recall</span>
            <span style="color:#fbbf24;font-weight:700;font-family:monospace;">{metrics['recall']*100:.2f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#64748b;font-size:13px;">F1 Score</span>
            <span style="color:#a78bfa;font-weight:700;font-family:monospace;">{metrics['f1']*100:.2f}%</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown('<div style="font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;color:#38bdf8;margin-bottom:10px;">Dataset</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#111827;border:1px solid #1e2d3d;border-radius:10px;padding:16px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#64748b;font-size:13px;">Total Transactions</span>
            <span style="color:#f1f5f9;font-weight:600;font-family:monospace;">{metrics['total_transactions']:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#64748b;font-size:13px;">Fraud Cases</span>
            <span style="color:#f87171;font-weight:600;font-family:monospace;">{metrics['fraud_count']:,}</span>
        </div>
        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
            <span style="color:#64748b;font-size:13px;">Fraud Rate</span>
            <span style="color:#fbbf24;font-weight:600;font-family:monospace;">{metrics['fraud_rate']*100:.2f}%</span>
        </div>
        <div style="display:flex;justify-content:space-between;">
            <span style="color:#64748b;font-size:13px;">Features</span>
            <span style="color:#f1f5f9;font-weight:600;font-family:monospace;">{len(FEATURES)}</span>
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()
    n_checks  = len(st.session_state.history)
    n_flagged = sum(1 for h in st.session_state.history if h['pred'] == 1)
    st.markdown('<div style="font-size:11px;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;color:#38bdf8;margin-bottom:10px;">Session</div>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    sc1.metric("Checked", n_checks)
    sc2.metric("Flagged",  n_flagged)

# ── Hero ──────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">🛡️ <span>FraudShield</span> AI Detection System</div>
    <div class="hero-sub">Enterprise-grade credit card fraud detection · 1.29M transactions · 99.51% AUC · Random Forest</div>
</div>""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔍  Fraud Checker", "📊  Analytics", "🧠  Model Insights", "📈  EDA", "📋  History"
])

# ════════════════════════════════════════════════
# TAB 1 — FRAUD CHECKER
# ════════════════════════════════════════════════
with tab1:
    col_form, col_result = st.columns([1.1, 0.9], gap="large")

    with col_form:
        st.markdown('<div class="section-header">Transaction Details</div>', unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            amt      = st.number_input("💰 Amount (₹)", min_value=1.0, max_value=50000.0, value=150.0, step=1.0)
            category = st.selectbox("🏪 Category", CATEGORIES)
            gender   = st.selectbox("👤 Gender", ["M", "F"])
            age      = st.slider("🎂 Age", 18, 100, 38)
        with c2:
            city_pop = st.number_input("🏙️ City Population", min_value=100, max_value=5_000_000, value=80_000, step=1000)
            zip_code = st.number_input("📮 ZIP Code", min_value=10000, max_value=99999, value=30301)
            hour     = st.slider("🕐 Hour (0–23)", 0, 23, 14)
            day_of_week = st.selectbox("📅 Day of Week",
                options=[0,1,2,3,4,5,6],
                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x], index=2)

        c3, c4 = st.columns(2)
        with c3:
            day   = st.slider("Day of Month", 1, 31, 15)
            month = st.slider("Month", 1, 12, 6)
        with c4:
            merchant_name = st.text_input("🏬 Merchant Name", value="fraud_Rutherford-Mertz")
            job_title     = st.text_input("💼 Job Title",     value="Mechanical engineer")
            city_name     = st.text_input("🏙️ City",          value="Atlanta")
            state_code    = st.text_input("🗺️ State (2-letter)", value="GA")

        with st.expander("📍 Location Coordinates (optional)"):
            ca, cb = st.columns(2)
            with ca:
                lat      = st.number_input("Cardholder Lat",  value=33.9,   format="%.4f")
                long_    = st.number_input("Cardholder Long", value=-84.5,  format="%.4f")
            with cb:
                merch_lat  = st.number_input("Merchant Lat",  value=33.92,  format="%.4f")
                merch_long = st.number_input("Merchant Long", value=-84.52, format="%.4f")

        check_btn = st.button("🔍 Analyze Transaction", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">Analysis Result</div>', unsafe_allow_html=True)

        if check_btn:
            def safe_encode(encoder, value):
                classes = list(encoder.classes_)
                if value in classes:
                    return int(encoder.transform([value])[0]), False
                return 0, True   # unknown → 0, flag warning

            warnings_list = []
            merchant_enc, w = safe_encode(encoders['merchant'], merchant_name)
            if w: warnings_list.append(f"Merchant '{merchant_name}' not in training data")
            category_enc, _ = safe_encode(encoders['category'], category)
            gender_enc,   _ = safe_encode(encoders['gender'],   gender)
            city_enc,     w = safe_encode(encoders['city'],      city_name)
            if w: warnings_list.append(f"City '{city_name}' not in training data")
            state_enc,    w = safe_encode(encoders['state'],     state_code)
            if w: warnings_list.append(f"State '{state_code}' not in training data")
            job_enc,      w = safe_encode(encoders['job'],       job_title)
            if w: warnings_list.append(f"Job '{job_title}' not in training data")

            if warnings_list:
                warn_html = "<br>".join([f"⚠️ {w} — using default encoding" for w in warnings_list])
                st.markdown(f'<div class="warn-box">{warn_html}</div>', unsafe_allow_html=True)

            distance = np.sqrt((lat - merch_lat)**2 + (long_ - merch_long)**2)

            input_dict = {
                'merchant_enc': merchant_enc, 'category_enc': category_enc,
                'amt': amt, 'gender_enc': gender_enc, 'city_enc': city_enc,
                'state_enc': state_enc, 'zip': zip_code, 'lat': lat, 'long': long_,
                'city_pop': city_pop, 'job_enc': job_enc,
                'merch_lat': merch_lat, 'merch_long': merch_long,
                'hour': hour, 'day': day, 'month': month,
                'day_of_week': day_of_week, 'age': age, 'distance': distance
            }

            input_df   = pd.DataFrame([input_dict])[FEATURES]
            prediction = model.predict(input_df)[0]
            proba      = model.predict_proba(input_df)[0]
            fraud_prob = proba[1]
            legit_prob = proba[0]

            st.session_state.history.append({
                'time': datetime.now().strftime("%H:%M:%S"),
                'amt': amt, 'category': category, 'hour': hour,
                'age': age, 'pred': int(prediction), 'prob': fraud_prob
            })

            risk_label, risk_color = (
                ("LOW RISK",    "#4ade80") if fraud_prob < 0.25 else
                ("MEDIUM RISK", "#fbbf24") if fraud_prob < 0.60 else
                ("HIGH RISK",   "#f87171")
            )

            if prediction == 1:
                st.markdown(f"""
                <div class="result-fraud">
                    <div class="result-title-fraud">⚠️ FRAUD DETECTED</div>
                    <div style="font-size:13px;color:#94a3b8;margin-top:4px;">Transaction flagged for immediate review</div>
                    <div style="margin-top:14px;display:flex;gap:12px;align-items:center;">
                        <span class="badge-fraud">FRAUDULENT</span>
                        <span style="font-size:13px;color:#64748b;">Confidence: {fraud_prob*100:.1f}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-legit">
                    <div class="result-title-legit">✅ LEGITIMATE</div>
                    <div style="font-size:13px;color:#94a3b8;margin-top:4px;">Transaction appears safe to process</div>
                    <div style="margin-top:14px;display:flex;gap:12px;align-items:center;">
                        <span class="badge-legit">APPROVED</span>
                        <span style="font-size:13px;color:#64748b;">Fraud probability: {fraud_prob*100:.1f}%</span>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number", value=round(fraud_prob*100, 1),
                title={'text': "Fraud Risk Score", 'font': {'color': '#94a3b8', 'size': 13}},
                number={'suffix': '%', 'font': {'color': '#f1f5f9', 'size': 30, 'family': 'JetBrains Mono'}},
                gauge={
                    'axis': {'range': [0,100], 'tickcolor': '#334155', 'tickfont': {'color':'#64748b','size':9}},
                    'bar': {'color': risk_color, 'thickness': 0.25},
                    'bgcolor': '#111827', 'bordercolor': '#1e2d3d',
                    'steps': [
                        {'range':[0,25],   'color':'rgba(34,197,94,0.10)'},
                        {'range':[25,60],  'color':'rgba(251,191,36,0.10)'},
                        {'range':[60,100], 'color':'rgba(239,68,68,0.10)'},
                    ],
                }
            ))
            fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                height=200, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(fig_g, use_container_width=True)

            # Prob bars
            fig_b = go.Figure(go.Bar(
                x=[legit_prob*100, fraud_prob*100], y=['Legitimate','Fraudulent'],
                orientation='h', marker_color=['#4ade80','#f87171'], marker_line_width=0,
                text=[f"{legit_prob*100:.1f}%", f"{fraud_prob*100:.1f}%"],
                textposition='inside', textfont={'color':'#0f172a','size':12,'family':'JetBrains Mono'}
            ))
            fig_b.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                height=100, margin=dict(l=0,r=0,t=0,b=0),
                                xaxis=dict(showgrid=False,showticklabels=False,range=[0,100]),
                                yaxis=dict(tickfont={'color':'#94a3b8','size':12}),
                                showlegend=False, bargap=0.35)
            st.plotly_chart(fig_b, use_container_width=True)

            st.markdown(f'<div style="text-align:center;font-size:13px;font-weight:700;color:{risk_color};letter-spacing:1.5px;padding:6px;background:rgba(255,255,255,0.03);border-radius:8px;border:1px solid #1e2d3d;">{risk_label}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Amount",   f"₹{amt:,.0f}")
            m2.metric("Hour",     f"{hour:02d}:00")
            m3.metric("Distance", f"{distance:.2f}°")

            st.markdown('<div class="section-header">Risk Signals</div>', unsafe_allow_html=True)
            factors = []
            if hour >= 22 or hour <= 4:
                factors.append(("🌙 Late-night transaction (peak fraud hour)", "warning"))
            if amt > 500:
                factors.append((f"💸 High amount (₹{amt:,.0f})", "warning"))
            if category in ['misc_net','shopping_net','travel','grocery_net']:
                factors.append((f"🛒 Higher-risk category: {category}", "warning"))
            if distance > 5:
                factors.append((f"📍 Large cardholder-merchant distance ({distance:.1f}°)", "warning"))
            if age < 25 or age > 80:
                factors.append(("👤 Unusual age profile", "info"))
            if not factors:
                factors.append(("✅ No significant risk signals detected", "ok"))
            for text, level in factors:
                color = "#fbbf24" if level=="warning" else ("#94a3b8" if level=="info" else "#4ade80")
                st.markdown(f'<div style="font-size:13px;color:{color};padding:6px 10px;background:rgba(255,255,255,0.02);border-left:3px solid {color};border-radius:0 6px 6px 0;margin-bottom:5px;">{text}</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="background:#111827;border:1px dashed #1e2d3d;border-radius:14px;padding:48px;text-align:center;">
                <div style="font-size:48px;margin-bottom:12px;">🛡️</div>
                <div style="font-size:15px;font-weight:600;color:#475569;">Enter transaction details</div>
                <div style="font-size:13px;color:#334155;margin-top:6px;">and click Analyze to get a fraud risk score</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 2 — ANALYTICS
# ════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    total  = len(df_all)
    frauds = df_all['is_fraud'].sum()
    avg_fa = df_all[df_all['is_fraud']==1]['amt'].mean()
    avg_la = df_all[df_all['is_fraud']==0]['amt'].mean()

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Total Transactions", f"{total:,}")
    mc2.metric("Fraud Cases",        f"{frauds:,}", delta=f"{frauds/total*100:.2f}%", delta_color="inverse")
    mc3.metric("Avg Fraud Amount",   f"₹{avg_fa:,.0f}")
    mc4.metric("Avg Legit Amount",   f"₹{avg_la:,.0f}")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    ca, cb = st.columns(2)

    with ca:
        st.markdown('<div class="section-header">Fraud by Category</div>', unsafe_allow_html=True)
        cat_f = df_all[df_all['is_fraud']==1].groupby('category').size().sort_values(ascending=True)
        fig1 = go.Figure(go.Bar(x=cat_f.values, y=cat_f.index, orientation='h',
            marker=dict(color=cat_f.values, colorscale=[[0,'#1e3a5f'],[1,'#f87171']], line_width=0),
            text=cat_f.values, textposition='outside', textfont={'color':'#64748b','size':10}))
        fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=340, margin=dict(l=0,r=40,t=0,b=0),
            xaxis=dict(showgrid=False,showticklabels=False),
            yaxis=dict(tickfont={'color':'#94a3b8','size':11}))
        st.plotly_chart(fig1, use_container_width=True)

    with cb:
        st.markdown('<div class="section-header">Fraud by Hour</div>', unsafe_allow_html=True)
        hour_f = df_all[df_all['is_fraud']==1].groupby('hour').size().reset_index(name='count')
        fig2 = go.Figure(go.Scatter(x=hour_f['hour'], y=hour_f['count'],
            mode='lines+markers', line=dict(color='#38bdf8',width=2.5),
            marker=dict(color='#0ea5e9',size=6,line=dict(color='#0f172a',width=2)),
            fill='tozeroy', fillcolor='rgba(56,189,248,0.08)'))
        fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=340, margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(tickfont={'color':'#64748b'},gridcolor='#1e2d3d',title=dict(text='Hour',font={'color':'#64748b'})),
            yaxis=dict(tickfont={'color':'#64748b'},gridcolor='#1e2d3d'))
        st.plotly_chart(fig2, use_container_width=True)

    cc, cd = st.columns(2)
    with cc:
        st.markdown('<div class="section-header">Fraud Amount Distribution</div>', unsafe_allow_html=True)
        fig3 = go.Figure(go.Histogram(x=df_all[df_all['is_fraud']==1]['amt'], nbinsx=50,
            marker_color='rgba(248,113,113,0.6)', marker_line_color='rgba(248,113,113,0.8)', marker_line_width=0.5))
        fig3.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=260, margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(tickfont={'color':'#64748b'},gridcolor='#1e2d3d'),
            yaxis=dict(tickfont={'color':'#64748b'},gridcolor='#1e2d3d'), bargap=0.05)
        st.plotly_chart(fig3, use_container_width=True)

    with cd:
        st.markdown('<div class="section-header">Fraud by Day of Week</div>', unsafe_allow_html=True)
        dow_map = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
        dow_f   = df_all[df_all['is_fraud']==1].groupby('day_of_week').size().reset_index(name='count')
        dow_f['day_name'] = dow_f['day_of_week'].map(dow_map)
        fig4 = go.Figure(go.Bar(x=dow_f['day_name'], y=dow_f['count'],
            marker_color='rgba(56,189,248,0.7)', marker_line_width=0))
        fig4.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=260, margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(tickfont={'color':'#64748b'}),
            yaxis=dict(tickfont={'color':'#64748b'},gridcolor='#1e2d3d'))
        st.plotly_chart(fig4, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 3 — MODEL INSIGHTS
# ════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Model Performance Metrics</div>', unsafe_allow_html=True)
    pm1,pm2,pm3,pm4,pm5 = st.columns(5)
    pm1.metric("AUC-ROC",   f"{metrics['auc']*100:.2f}%")
    pm2.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%")
    pm3.metric("Recall",    f"{metrics['recall']*100:.2f}%")
    pm4.metric("Precision", f"{metrics['precision']*100:.2f}%")
    pm5.metric("F1 Score",  f"{metrics['f1']*100:.2f}%")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    ia, ib = st.columns(2)

    with ia:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = metrics['confusion_matrix']
        fig_cm = go.Figure(go.Heatmap(
            z=[[cm[0][0],cm[0][1]],[cm[1][0],cm[1][1]]],
            x=['Pred: Legit','Pred: Fraud'], y=['Actual: Legit','Actual: Fraud'],
            colorscale=[[0,'#111827'],[0.5,'#1e3a5f'],[1,'#38bdf8']],
            text=[[f"{cm[0][0]:,}",f"{cm[0][1]:,}"],[f"{cm[1][0]:,}",f"{cm[1][1]:,}"]],
            texttemplate="%{text}", textfont={'size':16,'color':'white'}, showscale=False
        ))
        fig_cm.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=280, margin=dict(l=0,r=0,t=0,b=0),
            xaxis=dict(tickfont={'color':'#94a3b8'}),
            yaxis=dict(tickfont={'color':'#94a3b8'}))
        st.plotly_chart(fig_cm, use_container_width=True)

    with ib:
        st.markdown('<div class="section-header">Why These Metrics?</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2d3d;border-radius:10px;padding:16px;font-size:13px;color:#94a3b8;line-height:1.8;">
            <b style="color:#38bdf8;">AUC-ROC (99.51%)</b> — Primary metric for fraud detection. Measures ability to separate fraud from legitimate across all thresholds.<br><br>
            <b style="color:#4ade80;">Recall (93.60%)</b> — Most critical: out of all real fraud cases, we catch 93.6%. Missing fraud is more costly than a false alarm.<br><br>
            <b style="color:#fbbf24;">Accuracy (98.43%)</b> — High due to class imbalance (99.4% legit). Recall and AUC are stronger indicators.<br><br>
            <b style="color:#a78bfa;">Class Imbalance</b> — Only 0.58% transactions are fraud. RandomUnderSampler applied during training.
        </div>""", unsafe_allow_html=True)

    # ROC + PR curves
    st.markdown('<div class="section-header">ROC & Precision-Recall Curves</div>', unsafe_allow_html=True)
    try:
        st.image('roc_pr_curves.png', use_container_width=True)
    except:
        st.info("roc_pr_curves.png not found.")

    # SHAP
    st.markdown('<div class="section-header">SHAP Feature Importance</div>', unsafe_allow_html=True)
    try:
        st.image('shap_plot.png', use_container_width=True,
                 caption="SHAP values — top features driving fraud prediction")
    except:
        st.info("shap_plot.png not found.")

    # RF Feature importance
    st.markdown('<div class="section-header">Random Forest Feature Importance</div>', unsafe_allow_html=True)
    fi_df = pd.DataFrame({'feature': FEATURES, 'importance': model.feature_importances_}).sort_values('importance', ascending=True)
    fig_fi = go.Figure(go.Bar(
        x=fi_df['importance'], y=fi_df['feature'], orientation='h',
        marker=dict(color=fi_df['importance'], colorscale=[[0,'#1e3a5f'],[1,'#38bdf8']], line_width=0),
        text=(fi_df['importance']*100).round(1).astype(str)+'%',
        textposition='outside', textfont={'color':'#64748b','size':10}
    ))
    fig_fi.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=420, margin=dict(l=0,r=60,t=0,b=0),
        xaxis=dict(showgrid=False,showticklabels=False),
        yaxis=dict(tickfont={'color':'#94a3b8','size':11}))
    st.plotly_chart(fig_fi, use_container_width=True)

# ════════════════════════════════════════════════
# TAB 4 — EDA
# ════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Exploratory Data Analysis — 1.29M Transactions</div>', unsafe_allow_html=True)
    try:
        st.image('eda_plots.png', use_container_width=True,
                 caption="EDA: Full fraud pattern analysis from 1.29M credit card transactions")
    except:
        st.info("eda_plots.png not found.")

    st.markdown('<div class="section-header">Key EDA Findings</div>', unsafe_allow_html=True)
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2d3d;border-radius:10px;padding:16px;">
            <div style="font-size:11px;color:#38bdf8;font-weight:600;letter-spacing:1px;margin-bottom:8px;">⏰ TIME PATTERN</div>
            <div style="font-size:13px;color:#94a3b8;line-height:1.7;">
                Fraud spikes sharply between <b style="color:#f1f5f9;">10 PM – 2 AM</b>. 
                Daytime hours (9AM–5PM) show the lowest fraud rates. 
                Late-night is the highest-risk window.
            </div>
        </div>""", unsafe_allow_html=True)
    with e2:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2d3d;border-radius:10px;padding:16px;">
            <div style="font-size:11px;color:#38bdf8;font-weight:600;letter-spacing:1px;margin-bottom:8px;">🏪 CATEGORY PATTERN</div>
            <div style="font-size:13px;color:#94a3b8;line-height:1.7;">
                <b style="color:#f1f5f9;">grocery_pos</b> and <b style="color:#f1f5f9;">shopping_net</b> 
                have the highest fraud counts. Online categories (net) carry higher risk than in-person (pos).
            </div>
        </div>""", unsafe_allow_html=True)
    with e3:
        st.markdown("""
        <div style="background:#111827;border:1px solid #1e2d3d;border-radius:10px;padding:16px;">
            <div style="font-size:11px;color:#38bdf8;font-weight:600;letter-spacing:1px;margin-bottom:8px;">💰 AMOUNT PATTERN</div>
            <div style="font-size:13px;color:#94a3b8;line-height:1.7;">
                Fraud occurs across all amounts but clusters around 
                <b style="color:#f1f5f9;">small amounts (under ₹100)</b> and 
                <b style="color:#f1f5f9;">₹300–₹1,200 range</b>.
            </div>
        </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════
# TAB 5 — HISTORY
# ════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Transaction Check History</div>', unsafe_allow_html=True)
    if not st.session_state.history:
        st.markdown("""
        <div style="background:#111827;border:1px dashed #1e2d3d;border-radius:14px;padding:48px;text-align:center;">
            <div style="font-size:40px;margin-bottom:12px;">📋</div>
            <div style="font-size:14px;color:#475569;">No checks yet.</div>
        </div>""", unsafe_allow_html=True)
    else:
        h = st.session_state.history
        h_fraud = sum(1 for x in h if x['pred']==1)
        hs1,hs2,hs3 = st.columns(3)
        hs1.metric("Checks",    len(h))
        hs2.metric("Flagged",   h_fraud)
        hs3.metric("Avg Risk",  f"{np.mean([x['prob'] for x in h])*100:.1f}%")
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        df_h = pd.DataFrame(h)
        df_h['Status'] = df_h['pred'].map({1:'🔴 Fraud',0:'🟢 Legit'})
        df_h['Risk %'] = (df_h['prob']*100).round(1).astype(str)+'%'
        df_h['Amount'] = df_h['amt'].apply(lambda x: f"₹{x:,.0f}")
        st.dataframe(df_h[['time','Amount','category','hour','age','Status','Risk %']].rename(
            columns={'time':'Time','category':'Category','hour':'Hour','age':'Age'}),
            use_container_width=True, hide_index=True)
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
        if len(h) > 1:
            st.markdown('<div class="section-header">Risk Trend</div>', unsafe_allow_html=True)
            fig_t = go.Figure(go.Scatter(
                y=[x['prob']*100 for x in h], x=list(range(1,len(h)+1)),
                mode='lines+markers', line=dict(color='#38bdf8',width=2),
                marker=dict(color=['#f87171' if x['pred']==1 else '#4ade80' for x in h],
                            size=9, line=dict(color='#0f172a',width=2)),
                fill='tozeroy', fillcolor='rgba(56,189,248,0.06)'
            ))
            fig_t.add_hline(y=50, line_dash="dash", line_color="#334155")
            fig_t.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=200, margin=dict(l=0,r=0,t=10,b=0),
                xaxis=dict(tickfont={'color':'#64748b'},gridcolor='#1e2d3d'),
                yaxis=dict(tickfont={'color':'#64748b'},gridcolor='#1e2d3d',range=[0,105]))
            st.plotly_chart(fig_t, use_container_width=True)
