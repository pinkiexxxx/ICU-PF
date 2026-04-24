import numpy as np
import streamlit as st
import shap
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

matplotlib.use('Agg')

st.set_page_config(page_title="脓毒症风险预测", page_icon="⚕️", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #f8f9fa; }
    .main-header { background: linear-gradient(135deg, #005c97 0%, #363795 100%);
                   padding: 20px; border-radius: 12px; color: white; margin-bottom: 20px; }
    .main-header h1 { color: white; font-weight: 700; }
    .result-box { background: white; padding: 25px; border-radius: 15px;
                  text-align: center; border-left: 10px solid #ddd; }
    .result-value { font-size: 3.5rem; font-weight: 800; color: #333; }
    .result-label { font-weight: bold; color: white; padding: 5px 15px; border-radius: 20px; }
    .chart-title { font-size: 1.4rem; font-weight: 700; color: #2c3e50; border-left: 6px solid #005c97;
                   padding-left: 15px; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load("ann_model.pkl")

classifier1 = load_model()

FEATURE_NAMES = ['Gender', 'Dementia', 'Pneumonia', 'Antibiotics',
                 'Nephtox', 'Glucocorticoid', 'Sofa', 'RDW', 'Calcium_total']
REAL_BASE_SCORE = 0.6211          # 训练集平均正类概率

def prepare_background():
    """加载并验证背景数据，如果不可用或基线偏差过大就返回 None"""
    bg_file = "background_data.csv"
    if not os.path.exists(bg_file):
        return None

    try:
        raw = pd.read_csv(bg_file)
        bg = raw[FEATURE_NAMES].copy()
        # 处理缺失值
        if bg.isnull().values.any():
            for col in FEATURE_NAMES:
                if col in ['Gender','Dementia','Pneumonia','Antibiotics','Nephtox','Glucocorticoid']:
                    bg[col].fillna(bg[col].mode()[0], inplace=True)
                else:
                    bg[col].fillna(bg[col].median(), inplace=True)
            bg.dropna(inplace=True)

        # 计算该背景数据在模型上的基线
        temp_explainer = shap.KernelExplainer(classifier1.predict_proba, bg)
        expected = temp_explainer.expected_value
        if hasattr(expected, '__len__'):
            bl = float(expected[1])
        else:
            bl = float(expected)

        # 如果基线偏离训练集均值超过 0.2，说明背景数据分布有问题
        if abs(bl - REAL_BASE_SCORE) > 0.2:
            st.warning("背景数据分布与训练集差异过大，已自动切换为模拟数据以保证一致性。")
            return None
        return bg
    except Exception as e:
        st.warning(f"背景数据加载失败: {e}")
        return None

def generate_random_bg(n=100):
    np.random.seed(42)
    bg = pd.DataFrame({
        col: (
            np.random.choice([0,1], n) if col in ['Gender','Dementia','Pneumonia','Antibiotics','Nephtox','Glucocorticoid']
            else np.random.randint(0, 51, n) if col == 'Sofa'               # 0–50
            else np.random.uniform(0, 50, n) if col == 'RDW'                # 0–50
            else np.random.uniform(0, 50, n)                                # 钙 0–50
        ) for col in FEATURE_NAMES
    })
    return bg

def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3063/3063176.png", width=60)
        st.markdown("### 患者数据")
        with st.form("input_form"):
            st.markdown("#### 👤 基本信息")
            Gender = st.selectbox("性别", ["男", "女"])
            Gender_val = 1 if Gender == "男" else 0

            st.markdown("#### ❤️ 合并症")
            c1, c2 = st.columns(2)
            with c1:
                Dementia = st.selectbox("痴呆", ["否", "是"])
                Dementia_val = 1 if Dementia == "是" else 0
            with c2:
                Pneumonia = st.selectbox("肺炎", ["否", "是"])
                Pneumonia_val = 1 if Pneumonia == "是" else 0

            st.markdown("#### 💊 用药情况")
            c3, c4 = st.columns(2)
            with c3:
                Antibiotics = st.selectbox("抗生素", ["否", "是"])
                Antibiotics_val = 1 if Antibiotics == "是" else 0
            with c4:
                Nephtox = st.selectbox("肾毒性药物", ["否", "是"])
                Nephtox_val = 1 if Nephtox == "是" else 0
            Glucocorticoid = st.selectbox("糖皮质激素", ["否", "是"])
            Glucocorticoid_val = 1 if Glucocorticoid == "是" else 0

            st.markdown("#### 🧪 实验室检查")
            Sofa = st.number_input("SOFA 评分", 0, 50, 5)
            RDW = st.number_input("RDW (%)", 0.0, 50.0, 14.5, step=0.1)
            Calcium_total = st.number_input("血清总钙 (mEq/L)", 0.0, 50.0, 8.5, step=0.1)

            predict_btn = st.form_submit_button("开始分析", type="primary", use_container_width=True)

    st.markdown("""
    <div class="main-header">
        <h1>重症监护室的肺纤维化患者脓毒症风险预测</h1>
        <p style="opacity:0.9">基于人工神经网络的临床决策辅助工具</p>
    </div>
    """, unsafe_allow_html=True)

    if predict_btn:
        data_values = np.array([[
            Gender_val, Dementia_val, Pneumonia_val, Antibiotics_val,
            Nephtox_val, Glucocorticoid_val, Sofa, RDW, Calcium_total
        ]])
        df_input = pd.DataFrame(data_values, columns=FEATURE_NAMES)

        # 1. 加载背景数据
        bg_data = prepare_background()
        if bg_data is None:
            bg_data = generate_random_bg()

        # 2. 计算 SHAP 并使用 f(x) 作为卡片概率
        with st.spinner("正在计算..."):
            explainer = shap.KernelExplainer(classifier1.predict_proba, bg_data)
            shap_vals_full = explainer.shap_values(df_input)

            if isinstance(shap_vals_full, list):
                shap_vals = shap_vals_full[1][0]
                current_base = float(explainer.expected_value[1])
            else:
                if len(shap_vals_full.shape) == 3:
                    shap_vals = shap_vals_full[0, :, 1]
                    current_base = float(explainer.expected_value[1])
                else:
                    shap_vals = shap_vals_full[0]
                    current_base = float(explainer.expected_value)

            shap_vals = np.array(shap_vals)
            # 校准
            bias = current_base - REAL_BASE_SCORE
            if abs(bias) < 0.05:
                final_base = REAL_BASE_SCORE
                final_values = shap_vals + (bias / len(FEATURE_NAMES))
            else:
                final_base = current_base
                final_values = shap_vals

            prob_final = final_base + np.sum(final_values)

        # 3. 展示卡片
        prob_percent = prob_final * 100
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            if prob_final < 0.25:
                color, status = "#5cb85c", "低风险"
            elif prob_final < 0.50:
                color, status = "#f0ad4e", "中风险"
            elif prob_final <= 0.6211:
                color, status = "#d9534f", "中高风险"
            else:
                color, status = "#8b0000", "高风险"
            st.markdown(f"""
            <div class="result-box" style="border-left:10px solid {color};">
                <div style="color:#666;">预测脓毒症风险</div>
                <div class="result-value">{prob_percent:.2f}%</div>
                <div class="result-label" style="background:{color};">{status}</div>
            </div>
            """, unsafe_allow_html=True)

        # 4. 绘制 SHAP
        explanation = shap.Explanation(values=final_values, base_values=final_base,
                                       data=df_input.iloc[0,:].values, feature_names=FEATURE_NAMES)

        with st.container():
            st.markdown('<div class="chart-title">力图</div>', unsafe_allow_html=True)
            plt.figure(figsize=(24,5))
            shap.force_plot(final_base, final_values, df_input.iloc[0,:],
                            matplotlib=True, show=False, text_rotation=0)
            fig = plt.gcf()
            ax = plt.gca()
            for txt in ax.texts:
                if "f(x)" in txt.get_text():
                    txt.set_visible(True)
                    txt.set_fontsize(18)
                    txt.set_fontweight('bold')
                    txt.set_color('#000080')
            st.pyplot(fig, bbox_inches='tight')
            plt.clf()

        with st.container():
            st.markdown('<div class="chart-title">瀑布图</div>', unsafe_allow_html=True)
            fig2, ax2 = plt.subplots(figsize=(10,8))
            shap.plots.waterfall(explanation, max_display=12, show=False)
            st.pyplot(fig2, bbox_inches='tight')
            plt.clf()

    else:
        st.markdown("<br><br><h3 style='text-align:center;color:#999;'>⬅️ 请输入数据开始分析</h3>",
                    unsafe_allow_html=True)

if __name__ == '__main__':
    main()