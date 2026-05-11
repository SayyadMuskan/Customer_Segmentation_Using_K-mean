# Customer_Segmentation_Using_K-mean
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import time

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="AI Customer Analytics",
    page_icon="🚀",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.hero {
    background: linear-gradient(135deg,#111827,#1e293b);
    padding: 60px;
    border-radius: 30px;
    box-shadow: 0px 0px 25px rgba(0,255,255,0.2);
    text-align: center;
    margin-bottom: 30px;
}

.hero h1 {
    font-size: 60px;
    color: white;
    font-weight: bold;
}

.hero p {
    font-size: 22px;
    color: #cbd5e1;
}

.metric-card {
    background: linear-gradient(135deg,#111827,#1f2937);
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    box-shadow: 0px 0px 15px rgba(0,255,255,0.15);
    transition: 0.3s;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0px 0px 25px rgba(0,255,255,0.35);
}

.metric-number {
    font-size: 40px;
    color: #22d3ee;
    font-weight: bold;
}

.metric-title {
    font-size: 18px;
    color: white;
}

.feature-box {
    background: linear-gradient(135deg,#1e293b,#0f172a);
    padding: 30px;
    border-radius: 20px;
    box-shadow: 0px 0px 20px rgba(255,255,255,0.08);
    margin-top: 20px;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customer.csv")

try:
    df = load_data()

except:
    st.error("Mall_Customer.csv file not found")
    st.stop()

st.sidebar.title("🚀 AI Dashboard")

menu = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📁 Dataset",
        "📈 Visualizations",
        "🤖 Clustering",
        "💡 Business Insights"
    ]
)

if menu == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <h1>🚀 AI Customer Segmentation Dashboard</h1>
        <p>
            Advanced Machine Learning Analytics Platform
            for Customer Intelligence and Business Insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.image(
        "https://images.unsplash.com/photo-1551288049-bebda4e38f71",
        use_container_width=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">{len(df)}</div>
            <div class="metric-title">Customers</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">
                {round(df['Annual Income (k$)'].mean(),1)}K
            </div>
            <div class="metric-title">Average Income</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">
                {round(df['Spending Score (1-100)'].mean(),1)}
            </div>
            <div class="metric-title">Spending Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-number">
                {round(df['Age'].mean(),1)}
            </div>
            <div class="metric-title">Average Age</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div class="feature-box">
        <h2>📊 About The Project</h2>

        <p>
        This project uses Machine Learning and Data Science techniques
        to analyze customer purchasing behavior.
        Customers are segmented using K-Means Clustering
        based on income, age, and spending score.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col5, col6 = st.columns(2)

    with col5:

        st.markdown("""
        <div class="feature-box">
            <h2>🤖 AI Features</h2>

            <ul>
                <li>K-Means Clustering</li>
                <li>PCA Visualization</li>
                <li>Interactive Dashboard</li>
                <li>Customer Analytics</li>
                <li>Real-Time Insights</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col6:

        st.markdown("""
        <div class="feature-box">
            <h2>💡 Business Benefits</h2>

            <ul>
                <li>Customer Segmentation</li>
                <li>Targeted Marketing</li>
                <li>Business Intelligence</li>
                <li>Premium Customer Detection</li>
                <li>Marketing Optimization</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    components.html(
        """
        <div style="
        padding:20px;
        background:#111827;
        border-radius:20px;
        ">

        <marquee behavior="scroll"
        direction="left"
        scrollamount="10"
        style="color:#22d3ee;font-size:24px;">

        🚀 AI Clustering Active |
        📊 Analytics Loaded |
        🤖 Machine Learning Running |
        💰 Premium Customers Detected |
        📈 Business Intelligence Ready

        </marquee>

        </div>
        """,
        height=100
    )

elif menu == "📁 Dataset":

    st.title("📁 Customer Dataset")

    st.dataframe(df, use_container_width=True)

elif menu == "📈 Visualizations":

    st.title("📈 Advanced Visualizations")

    fig1 = px.histogram(
        df,
        x='Age',
        nbins=20,
        template='plotly_dark',
        title='Age Distribution'
    )

    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.scatter(
        df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',
        color='Gender',
        size='Age',
        hover_data=['CustomerID'],
        template='plotly_dark',
        title='Income vs Spending'
    )

    st.plotly_chart(fig2, use_container_width=True)

elif menu == "🤖 Clustering":

    st.title("🤖 AI Customer Segmentation")

    features = st.multiselect(
        "Select Features",
        [
            'Age',
            'Annual Income (k$)',
            'Spending Score (1-100)'
        ],
        default=[
            'Annual Income (k$)',
            'Spending Score (1-100)'
        ]
    )

    if len(features) >= 2:

        X = df[features]

        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)

        k = st.slider(
            "Select Number of Clusters",
            2,
            10,
            5
        )

        with st.spinner("🤖 Training AI Model..."):

            time.sleep(2)

            kmeans = KMeans(
                n_clusters=k,
                random_state=42,
                n_init=10
            )

            clusters = kmeans.fit_predict(X_scaled)

        df['Cluster'] = clusters

        score = silhouette_score(
            X_scaled,
            clusters
        )

        st.success(
            f"Silhouette Score : {round(score,3)}"
        )

        fig5 = px.scatter(
            df,
            x=features[0],
            y=features[1],
            color=df['Cluster'].astype(str),
            size='Age',
            hover_data=['CustomerID'],
            template='plotly_dark',
            title='AI Customer Segmentation'
        )

        st.plotly_chart(fig5, use_container_width=True)

elif menu == "💡 Business Insights":

    st.title("💡 AI Business Insights")

    st.success("""
    🟢 High Income + High Spending Customers
    → Premium customers
    → Offer luxury memberships
    """)

    st.warning("""
    🟡 High Income + Low Spending Customers
    → Potential customers
    → Use personalized marketing
    """)

    st.info("""
    🔵 Low Income + High Spending Customers
    → Young impulsive buyers
    → Provide seasonal discounts
    """)

    st.error("""
    🔴 Low Income + Low Spending Customers
    → Budget customers
    → Promote affordable products
    """)
