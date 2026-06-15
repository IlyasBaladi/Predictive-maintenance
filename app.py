import streamlit as st
import numpy as np
import joblib
import os

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predictive Maintenance — Hasnaa Moutawakil",
    page_icon="🔧",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0D1B2A; color: #E8F4F8; }

    /* Header bar */
    .header-box {
        background: linear-gradient(135deg, #1B4F72 0%, #00A8CC 100%);
        border-radius: 12px;
        padding: 20px 30px;
        margin-bottom: 25px;
    }
    .header-box h1 { color: white; margin: 0; font-size: 2rem; }
    .header-box p  { color: #D0EAF4; margin: 4px 0 0 0; font-size: 0.95rem; }

    /* Metric cards */
    .metric-card {
        background: #1B2E3C;
        border: 1px solid #00A8CC44;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        margin: 6px 0;
    }
    .metric-card .val  { font-size: 2.2rem; font-weight: 700; color: #00A8CC; }
    .metric-card .lbl  { font-size: 0.8rem; color: #7EA8C0; margin-top: 2px; }

    /* Result panels */
    .result-ok {
        background: #0D2E1A;
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-warn {
        background: #2E1A0D;
        border: 2px solid #F59E0B;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-danger {
        background: #2E0D0D;
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .result-ok h2, .result-warn h2, .result-danger h2 {
        margin: 0; font-size: 1.5rem;
    }

    /* Section titles */
    .section-title {
        color: #00A8CC;
        font-size: 1.1rem;
        font-weight: 700;
        border-left: 4px solid #00A8CC;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0A1520; }
    [data-testid="stSidebar"] .stMarkdown { color: #A8C0D6; }

    /* Sliders & inputs labels */
    label { color: #A8D8EA !important; font-size: 0.9rem !important; }

    /* Button */
    .stButton > button {
        background: linear-gradient(135deg, #1B4F72, #00A8CC);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 30px;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* Divider */
    hr { border-color: #1B4F7244; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
    <h1>🔧 Predictive Maintenance Dashboard</h1>
    <p>AI4I 2020 Dataset · Gradient Boosting · Hasnaa Moutawakil · EHTP</p>
</div>
""", unsafe_allow_html=True)


# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load both saved Gradient Boosting models."""
    models = {}
    for key, fname in [("reg", "gb_regressor.pkl"), ("clf", "gb_classifier.pkl")]:
        if os.path.exists(fname):
            models[key] = joblib.load(fname)
        else:
            models[key] = None
    return models

models = load_models()

if models["reg"] is None or models["clf"] is None:
    st.error(
        "⚠️ Modèles non trouvés. Assure-toi d'avoir exécuté `save_models.py` "
        "dans ton notebook Colab et d'avoir placé `gb_regressor.pkl` et "
        "`gb_classifier.pkl` dans le même dossier que `app.py`."
    )
    st.stop()


# ── Sidebar — Dataset stats ───────────────────────────────────
with st.sidebar:
    st.markdown("### 📊 Infos du Dataset")
    st.markdown("---")

    stats = {
        "Enregistrements": "10 000",
        "Variables": "14",
        "Taux de pannes": "3,4 %",
        "Meilleur modèle": "Gradient Boosting",
        "R² (Régression)": "0.84",
        "F1-Score (Classif.)": "0.76",
    }
    for k, v in stats.items():
        st.markdown(f"**{k}** : `{v}`")

    st.markdown("---")
    st.markdown("### 📌 Guide des valeurs")
    st.markdown("""
- **Type L** : Qualité basse (51%)
- **Type M** : Qualité moyenne (43%)
- **Type H** : Qualité haute (6%)
- **Température air** : 295 – 305 K
- **Vitesse** : 1168 – 2886 rpm
- **Couple** : 3.8 – 76.6 Nm
- **Usure outil** : 0 – 253 min
    """)


# ── Input form ────────────────────────────────────────────────
st.markdown('<div class="section-title">⚙️ Paramètres de la Machine</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**🌡️ Températures**")
    air_temp = st.slider(
        "Température air [K]",
        min_value=295.0, max_value=305.0, value=300.0, step=0.1,
        help="Température ambiante autour de la machine"
    )
    proc_temp = st.slider(
        "Température process [K]",
        min_value=305.0, max_value=314.0, value=310.0, step=0.1,
        help="Température interne du processus"
    )

with col2:
    st.markdown("**⚙️ Mécanique**")
    rot_speed = st.slider(
        "Vitesse rotation [rpm]",
        min_value=1168, max_value=2886, value=1538, step=1,
        help="Vitesse de rotation de la machine"
    )
    tool_wear = st.slider(
        "Usure de l'outil [min]",
        min_value=0, max_value=253, value=108, step=1,
        help="Durée d'utilisation de l'outil en minutes"
    )

with col3:
    st.markdown("**🏷️ Type de produit**")
    product_type = st.radio(
        "Type de qualité",
        options=["L (Basse)", "M (Moyenne)", "H (Haute)"],
        index=1,
        help="Catégorie de qualité du produit fabriqué"
    )
    type_code = product_type[0]  # L, M ou H

    # Température delta info
    delta_t = proc_temp - air_temp
    color = "#10B981" if delta_t <= 8.6 else "#F59E0B" if delta_t <= 10 else "#EF4444"
    st.markdown(f"""
    <div style="margin-top:16px; background:#1B2E3C; border-radius:8px; padding:12px; border:1px solid #1B4F72;">
        <span style="color:#A8C0D6; font-size:0.85rem;">Écart de température (ΔT)</span><br>
        <span style="font-size:1.6rem; font-weight:700; color:{color};">{delta_t:.1f} K</span><br>
        <span style="color:#7EA8C0; font-size:0.8rem;">Optimal : ΔT ≤ 8.6 K</span>
    </div>
    """, unsafe_allow_html=True)


# ── Prepare features ──────────────────────────────────────────
def build_features(air_temp, proc_temp, rot_speed, tool_wear, type_code):
    """
    Reproduit exactement le preprocessing du notebook :
    - Colonnes numériques : Air temperature K, Process temperature K,
                            Rotational speed rpm, Tool wear min
    - One-Hot encodé : Type_H, Type_L, Type_M
    Ordre : [Air temp K, Process temp K, Rot speed rpm, Tool wear min,
             Type_H, Type_L, Type_M]
    """
    type_h = 1 if type_code == "H" else 0
    type_l = 1 if type_code == "L" else 0
    type_m = 1 if type_code == "M" else 0
    return np.array([[air_temp, proc_temp, rot_speed, tool_wear,
                      type_h, type_l, type_m]])


# ── Predict button ────────────────────────────────────────────
st.markdown("---")
predict_btn = st.button("🚀 Lancer la Prédiction")

if predict_btn:
    X_input = build_features(air_temp, proc_temp, rot_speed, tool_wear, type_code)

    # ── Regression: Torque prediction ──────────────────────────
    torque_pred = models["reg"].predict(X_input)[0]

    # ── Classification: Failure prediction ─────────────────────
    failure_pred = models["clf"].predict(X_input)[0]
    failure_proba = models["clf"].predict_proba(X_input)[0][1]  # prob de panne

    st.markdown('<div class="section-title">📈 Résultats des Prédictions</div>', unsafe_allow_html=True)

    res_col1, res_col2 = st.columns(2)

    # Torque result
    with res_col1:
        torque_color = "#00A8CC"
        st.markdown(f"""
        <div style="background:#1B2E3C; border:2px solid #00A8CC; border-radius:12px; padding:24px; text-align:center;">
            <div style="color:#7EA8C0; font-size:0.9rem; margin-bottom:6px;">⚙️ RÉGRESSION — Couple prédit</div>
            <div style="font-size:3rem; font-weight:800; color:{torque_color};">{torque_pred:.1f}</div>
            <div style="color:#A8C0D6; font-size:1rem;">Nm (Newton-mètre)</div>
            <hr style="border-color:#1B4F7244; margin:14px 0;">
            <div style="color:#7EA8C0; font-size:0.82rem;">
                Plage normale : 3.8 – 76.6 Nm<br>
                Moyenne dataset : 40.0 Nm<br>
                <b style="color:#00A8CC;">R² du modèle : 0.84</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Failure result
    with res_col2:
        if failure_pred == 0 and failure_proba < 0.3:
            box_class = "result-ok"
            icon = "✅"
            status = "Machine en bon état"
            msg = "Aucune panne détectée. La machine fonctionne normalement."
            prob_color = "#10B981"
        elif failure_pred == 0 and failure_proba < 0.6:
            box_class = "result-warn"
            icon = "⚠️"
            status = "Surveillance recommandée"
            msg = "Risque modéré. Planifier une inspection prochaine."
            prob_color = "#F59E0B"
        else:
            box_class = "result-danger"
            icon = "🚨"
            status = "RISQUE DE PANNE DÉTECTÉ"
            msg = "Intervention de maintenance urgente recommandée !"
            prob_color = "#EF4444"

        st.markdown(f"""
        <div class="{box_class}">
            <div style="font-size:2.5rem;">{icon}</div>
            <h2 style="color:white; margin:8px 0;">{status}</h2>
            <p style="color:#CBD5E1; font-size:0.9rem; margin:4px 0;">{msg}</p>
            <hr style="border-color:#ffffff22; margin:14px 0;">
            <div style="color:#7EA8C0; font-size:0.85rem;">Probabilité de panne</div>
            <div style="font-size:2.2rem; font-weight:800; color:{prob_color};">{failure_proba*100:.1f}%</div>
            <div style="color:#7EA8C0; font-size:0.82rem; margin-top:6px;">
                <b style="color:{prob_color};">F1-Score du modèle : 0.76</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Summary metrics row ──────────────────────────────────
    st.markdown('<div class="section-title">📋 Récapitulatif des Entrées</div>', unsafe_allow_html=True)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    inputs_display = [
        (m1, f"{air_temp:.1f} K",   "Temp. Air"),
        (m2, f"{proc_temp:.1f} K",  "Temp. Process"),
        (m3, f"{rot_speed} rpm",    "Vitesse"),
        (m4, f"{tool_wear} min",    "Usure outil"),
        (m5, type_code,             "Type produit"),
        (m6, f"{delta_t:.1f} K",   "ΔT"),
    ]
    for col, val, lbl in inputs_display:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Failure probability bar ──────────────────────────────
    st.markdown('<div class="section-title">📊 Niveau de risque</div>', unsafe_allow_html=True)
    st.progress(min(failure_proba, 1.0))

    pct = failure_proba * 100
    if pct < 30:
        risk_text = f"🟢 Risque faible ({pct:.1f}%) — Fonctionnement normal"
    elif pct < 60:
        risk_text = f"🟡 Risque modéré ({pct:.1f}%) — Surveillance conseillée"
    else:
        risk_text = f"🔴 Risque élevé ({pct:.1f}%) — Maintenance urgente !"
    st.markdown(f"<p style='color:#A8C0D6; text-align:center;'>{risk_text}</p>", unsafe_allow_html=True)

else:
    # Placeholder before first prediction
    st.markdown("""
    <div style="background:#1B2E3C; border:1px dashed #1B4F72; border-radius:12px;
                padding:40px; text-align:center; color:#4A7FA5;">
        <div style="font-size:3rem;">🔧</div>
        <p style="font-size:1.1rem; margin-top:10px;">
            Ajuste les paramètres de la machine ci-dessus puis clique sur
            <b style="color:#00A8CC;">🚀 Lancer la Prédiction</b>
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style="text-align:center; color:#2A4A5E; font-size:0.8rem;">
    Hasnaa Moutawakil · Projet Final ML & Data Mining · EHTP ·
    AI4I 2020 Predictive Maintenance Dataset · Gradient Boosting
</p>
""", unsafe_allow_html=True)
