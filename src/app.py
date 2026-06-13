import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import altair as alt
import os
import json
import re
import tempfile
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
from transformers import pipeline as hf_pipeline
from pyvis.network import Network

st.set_page_config(page_title="Social Bot Detector Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# ─── NLP MODEL (cached, loaded once per session) ────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading NLP model (first time only)...")
def load_nlp_classifier():
    return hf_pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-3",
        device=-1           # CPU; change to 0 for GPU
    )

@st.cache_data
def get_user_posts(json_path: str, username: str) -> list[str]:
    """Return a list of clean post texts for a given username."""
    if not os.path.exists(json_path):
        return []
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_users = json.load(f)
    for user in raw_users:
        if user['username'] == username:
            texts = []
            for post in user.get('posts', []):
                content = post.get('content', '').strip()
                if content:
                    texts.append(content)
            return texts
    return []

@st.cache_data
def get_hashtags_map(json_path: str) -> dict[str, set]:
    """Return {username: set_of_hashtags} extracted from raw posts."""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_users = json.load(f)
    result = {}
    for user in raw_users:
        tags = set()
        for post in user.get('posts', []):
            content = post.get('content', '')
            tags.update(t.lower() for t in re.findall(r'#(\w+)', content))
        result[user['username']] = tags
    return result

@st.cache_data
def build_network_graph_html(
    usernames: list,
    hybrid_labels: list,
    detection_sources: list,
    outlier_reasons: list,
    feature_matrix: np.ndarray,    # shape (N, F) — unscaled features
    hashtag_map: dict,
    dist_threshold: float = 0.10,
    min_shared_tags: int = 3,
) -> str:
    """Build a pyvis network and return the generated HTML string."""
    net = Network(
        height="620px",
        width="100%",
        bgcolor="#000000",
        font_color="#ffffff",
        directed=False,
    )
    net.barnes_hut(
        gravity=-12000,
        central_gravity=0.3,
        spring_length=120,
        spring_strength=0.05,
        damping=0.09,
        overlap=0,
    )

    # Add nodes
    for i, (uname, label, src, reason) in enumerate(
        zip(usernames, hybrid_labels, detection_sources, outlier_reasons)
    ):
        is_bot = label == -1
        color  = "#ff4b4b" if is_bot else "#00cc96"
        size   = 18 if is_bot else 10
        title  = (
            f"<b>{uname}</b><br>"
            f"Status: {'🚨 BOT' if is_bot else '✅ Normal'}<br>"
            f"Source: {src or 'N/A'}<br>"
            f"Reason: {reason or 'N/A'}"
        )
        net.add_node(
            uname,
            label=uname,
            color=color,
            size=size,
            title=title,
            borderWidth=2 if is_bot else 1,
        )

    N = len(usernames)

    # Normalise feature matrix for distance computation
    feat_std = feature_matrix.std(axis=0)
    feat_std[feat_std == 0] = 1.0
    X_norm = (feature_matrix - feature_matrix.mean(axis=0)) / feat_std

    # Add edges — limit to avoid O(N^2) explosion on 500 nodes
    edge_count = 0
    MAX_EDGES = 1500
    for i in range(N):
        if edge_count >= MAX_EDGES:
            break
        for j in range(i + 1, N):
            if edge_count >= MAX_EDGES:
                break
            a, b = usernames[i], usernames[j]

            # Criterion 1: very close in feature space
            dist = float(np.linalg.norm(X_norm[i] - X_norm[j]))
            if dist < dist_threshold:
                net.add_edge(a, b, color="#334455", width=1,
                             title=f"Behavioral distance: {dist:.4f}")
                edge_count += 1
                continue

            # Criterion 2: many shared hashtags
            shared = hashtag_map.get(a, set()) & hashtag_map.get(b, set())
            if len(shared) > min_shared_tags:
                net.add_edge(a, b, color="#9933cc", width=2,
                             title=f"Shared hashtags: {', '.join(list(shared)[:6])}")
                edge_count += 1

    # Generate HTML to a temp file then read back
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode='w', encoding='utf-8') as tmp:
        net.save_graph(tmp.name)
        tmp_path = tmp.name
    with open(tmp_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    os.unlink(tmp_path)
    return html_content

# ─── DATA LOADING ────────────────────────────────────────────────────────────
@st.cache_data
def get_processed_raw_data(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_users = json.load(f)

    records = []
    for user in raw_users:
        followers  = user['followers_count']
        following  = user['following_count']
        follow_ratio = following / (followers + 1)
        posts = user['posts']
        n = len(posts)
        avg_time_diff_minutes = 1440.0
        engagement_rate = 0.0
        night_post_ratio = 0.0
        lexical_diversity = 1.0

        if n > 0:
            timestamps, engagements, night_count, all_words = [], 0, 0, []
            for post in posts:
                content = post.get('content', '')
                dt = pd.to_datetime(post['created_at']).tz_localize(None)
                timestamps.append(dt)
                engagements += post['favourites_count'] + post['reblogs_count'] + post['replies_count']
                if 1 <= dt.hour <= 5:
                    night_count += 1
                all_words.extend(re.findall(r'\w+', content.lower()))
            if n > 1:
                timestamps.sort()
                diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 60.0 for i in range(1, n)]
                avg_time_diff_minutes = float(np.mean(diffs))
            engagement_rate = engagements / n
            night_post_ratio = night_count / n
            if all_words:
                lexical_diversity = len(set(all_words)) / len(all_words)

        records.append({
            'user_id': user['id'],
            'username': user['username'],
            'is_bot_declared': 1 if user['bot'] else 0,
            'follow_ratio': follow_ratio,
            'avg_time_diff_minutes': avg_time_diff_minutes,
            'engagement_rate': engagement_rate,
            'night_post_ratio': night_post_ratio,
            'lexical_diversity': lexical_diversity,
        })
    return pd.DataFrame(records)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'mastodon_dataset.json')
df_raw = get_processed_raw_data(JSON_PATH)

if df_raw is None:
    st.error("⚠️ Không tìm thấy file `data/mastodon_dataset.json`. Vui lòng chạy crawler trước!")
    st.stop()

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2721/2721272.png", width=80)
    st.title("🛡️ Bot Detector Controls")
    st.markdown("Cấu hình trực tiếp các thuật toán Machine Learning.")
    st.markdown("---")
    st.header("⚙️ K-Means")
    n_clusters = st.slider("Số lượng cụm (n_clusters)", 2, 10, 3)
    st.header("⚙️ DBSCAN")
    eps = st.slider("Bán kính lân cận (eps)", 0.1, 2.5, 0.8, 0.1)
    min_samples = st.slider("Mẫu tối thiểu (min_samples)", 1, 10, 2)
    st.header("⚙️ Isolation Forest")
    contamination = st.slider("Tỷ lệ dị biệt (contamination)", 0.01, 0.30, 0.10, 0.01)

# ─── HYBRID DETECTION ENGINE ─────────────────────────────────────────────────
df = df_raw.copy()
FEATURES = ['follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity']

df['hybrid_cluster']  = 0
df['detection_source'] = ''
df['outlier_reason']   = ''

# ── Layer 1: Rule-Based Heuristics (raw feature values, no scaling needed) ──
rule_mask = pd.Series(False, index=df.index)

m1 = df['is_bot_declared'] == 1
df.loc[m1, 'outlier_reason'] += 'API Declared Bot | '
rule_mask |= m1

m2 = df['username'].str.contains('bot', case=False, na=False)
df.loc[m2, 'outlier_reason'] += 'Bot in Username | '
rule_mask |= m2

m3 = df['lexical_diversity'] < 0.1
df.loc[m3, 'outlier_reason'] += 'Extreme Repetitive Content | '
rule_mask |= m3

m4 = df['avg_time_diff_minutes'] < 0.5
df.loc[m4, 'outlier_reason'] += 'Inhuman Posting Speed | '
rule_mask |= m4

df.loc[rule_mask, 'hybrid_cluster']   = -1
df.loc[rule_mask, 'detection_source'] = 'rule'

# ── Layer 2: ML Models on clean subset only ───────────────────────────────────
clean_mask = ~rule_mask
df_clean   = df[clean_mask].copy()
X_clean    = df_clean[FEATURES].values

kmeans_sil = kmeans_db = dbscan_sil = dbscan_db = iso_sil = iso_db = "N/A"
dbscan_count = iso_count = 0

if len(df_clean) > 1:
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42, n_init=10)
    km_labels = kmeans.fit_predict(X_scaled)
    df_clean['kmeans_cluster'] = km_labels
    if len(set(km_labels)) > 1:
        kmeans_sil = f"{silhouette_score(X_scaled, km_labels):.4f}"
        kmeans_db  = f"{davies_bouldin_score(X_scaled, km_labels):.4f}"

    # DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = dbscan.fit_predict(X_scaled)
    df_clean['dbscan_label'] = db_labels
    db_out = db_labels == -1
    dbscan_count = int(db_out.sum())
    df_clean.loc[db_out, 'hybrid_cluster']   = -1
    df_clean.loc[db_out, 'detection_source'] = 'dbscan'
    df_clean.loc[db_out, 'outlier_reason']  += 'Density Outlier (DBSCAN) | '
    unique_db = set(db_labels)
    if len(unique_db - {-1}) >= 1 and len(unique_db) > 1:
        dbscan_sil = f"{silhouette_score(X_scaled, db_labels):.4f}"
        dbscan_db  = f"{davies_bouldin_score(X_scaled, db_labels):.4f}"

    # Isolation Forest — run on all clean users; flag only those NOT already caught by DBSCAN
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_labels = iso.fit_predict(X_scaled)
    df_clean['iso_label'] = iso_labels
    iso_out = iso_labels == -1
    iso_count = int(iso_out.sum())
    iso_only  = iso_out & ~db_out
    consensus  = iso_out & db_out
    df_clean.loc[iso_only,  'hybrid_cluster']   = -1
    df_clean.loc[iso_only,  'detection_source'] = 'iso'
    df_clean.loc[iso_only,  'outlier_reason']  += 'Statistical Outlier (Isolation Forest) | '
    df_clean.loc[consensus, 'detection_source'] = 'consensus'
    if len(set(iso_labels)) > 1:
        iso_sil = f"{silhouette_score(X_scaled, iso_labels):.4f}"
        iso_db  = f"{davies_bouldin_score(X_scaled, iso_labels):.4f}"

    # PCA for visualisation
    pca = PCA(n_components=2)
    pca_coords = pca.fit_transform(X_scaled)
    df_clean['pca_1'] = pca_coords[:, 0]
    df_clean['pca_2'] = pca_coords[:, 1]

    # Write results back to main df
    for col in ['kmeans_cluster', 'dbscan_label', 'iso_label', 'hybrid_cluster', 'detection_source', 'outlier_reason', 'pca_1', 'pca_2']:
        if col in df_clean.columns:
            df.loc[clean_mask, col] = df_clean[col].values

# Clean up trailing separators
df['outlier_reason'] = df['outlier_reason'].str.strip(' |').str.strip()
df['is_bot_label']   = df['hybrid_cluster'].apply(lambda x: 'Bot/Anomaly' if x == -1 else 'Normal User')

# Fill pca for rule-flagged (they were excluded from PCA, place at extreme)
if 'pca_1' not in df.columns:
    df['pca_1'] = 0.0
    df['pca_2'] = 0.0
df['pca_1'] = df['pca_1'].fillna(df['pca_1'].max() + 2 if df['pca_1'].notna().any() else 5.0)
df['pca_2'] = df['pca_2'].fillna(df['pca_2'].max() + 2 if df['pca_2'].notna().any() else 5.0)

# ─── METRICS SUMMARY ─────────────────────────────────────────────────────────
total_users   = len(df)
rule_flagged  = int(rule_mask.sum())
ml_flagged    = int((df['hybrid_cluster'] == -1).sum()) - rule_flagged
total_bots    = int((df['hybrid_cluster'] == -1).sum())
bot_ratio     = (total_bots / total_users) * 100

# ─── MAIN UI ─────────────────────────────────────────────────────────────────
st.title("🛡️ Real-Time Social Bot Detection Dashboard")
st.markdown("Hệ thống **Hybrid Detection**: Kết hợp Rule-Based heuristics và Machine Learning để phát hiện bot với độ chính xác cao nhất.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Dashboard & Analytics",
    "🗂️ Database & Blacklist",
    "🔍 Account Lookup (XAI)",
    "🕸️ Botnet Network Graph",
])

# ═══ TAB 1 ═══════════════════════════════════════════════════════════════════
with tab1:
    st.header("Thống kê Tổng quan")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tổng Tài Khoản", total_users)
    c2.metric("Rule-Based Flagged", rule_flagged)
    c3.metric("ML Flagged (thêm)", ml_flagged)
    c4.metric("Tổng Bot Phát Hiện", total_bots, delta=f"{bot_ratio:.1f}%", delta_color="inverse")
    c5.metric("K-Means Clusters", n_clusters)

    st.markdown("---")
    st.subheader("2D Behavior Projection (PCA)")
    scatter = alt.Chart(df).mark_circle(size=55, opacity=0.75).encode(
        x=alt.X('pca_1', title='PCA 1 (Behavior Trend)'),
        y=alt.Y('pca_2', title='PCA 2 (Activity Pattern)'),
        color=alt.Color('is_bot_label',
                        scale=alt.Scale(domain=['Normal User', 'Bot/Anomaly'],
                                        range=['#00cc96', '#ff4b4b']),
                        title='Label'),
        shape=alt.Shape('detection_source',
                        scale=alt.Scale(domain=['', 'rule', 'dbscan', 'iso', 'consensus'],
                                        range=['circle', 'triangle-up', 'square', 'diamond', 'cross']),
                        title='Detection Source'),
        tooltip=['username', 'follow_ratio', 'avg_time_diff_minutes',
                 'engagement_rate', 'night_post_ratio', 'lexical_diversity',
                 'detection_source', 'outlier_reason']
    ).interactive().properties(height=460)
    st.altair_chart(scatter, use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Bảng So Sánh Hiệu Suất Mô Hình (trên tập dữ liệu sạch)")
    metrics_df = pd.DataFrame({
        "Model": [f"K-Means (K={n_clusters})",
                  f"DBSCAN (eps={eps}, min_samples={min_samples})",
                  f"Isolation Forest (contamination={contamination:.2f})"],
        "Outliers Detected": ["N/A", dbscan_count, iso_count],
        "Silhouette Score ↑":  [kmeans_sil, dbscan_sil, iso_sil],
        "Davies-Bouldin Index ↓": [kmeans_db, dbscan_db, iso_db],
    })
    st.table(metrics_df)
    st.markdown("""
    **💡 Giải thích chỉ số:**
    - **Silhouette Score** (↑ tốt hơn, từ -1→1): Đo mức độ phân tách rõ ràng giữa các cụm. Giá trị càng cao (gần 1) nghĩa là các cụm càng tách biệt và nội bộ càng chặt chẽ.
    - **Davies-Bouldin Index** (↓ tốt hơn, ≥ 0): Đo tỷ lệ giữa khoảng cách nội cụm và khoảng cách liên cụm. Giá trị càng thấp nghĩa là cụm càng chặt và xa nhau hơn.
    """)

# ═══ TAB 2 ═══════════════════════════════════════════════════════════════════
with tab2:
    st.header("Quản lý Dữ liệu")

    st.subheader("📁 Tất cả tài khoản (All Scanned Users)")
    display_cols = ['username', 'follow_ratio', 'avg_time_diff_minutes', 'engagement_rate',
                    'night_post_ratio', 'lexical_diversity', 'is_bot_declared', 'hybrid_cluster',
                    'detection_source', 'outlier_reason']
    available = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available], use_container_width=True, height=300)

    st.markdown("---")
    st.subheader("☠️ Bot Blacklist (Hybrid: Rule + ML)")

    blacklist_df = df[df['hybrid_cluster'] == -1].copy()
    bl_cols = ['username', 'follow_ratio', 'avg_time_diff_minutes', 'engagement_rate',
               'night_post_ratio', 'lexical_diversity', 'detection_source', 'outlier_reason']
    bl_available = [c for c in bl_cols if c in blacklist_df.columns]

    if not blacklist_df.empty:
        st.dataframe(
            blacklist_df[bl_available].style.set_properties(**{'background-color': '#1a0000', 'color': '#ff9999'}),
            use_container_width=True, height=320
        )
        csv = blacklist_df[bl_available].to_csv(index=False).encode('utf-8')
        st.download_button("📥 Tải xuống Blacklist (.CSV)", data=csv,
                           file_name='bot_blacklist.csv', mime='text/csv')
    else:
        st.success("✅ Không phát hiện tài khoản bot nào với cấu hình hiện tại.")

# ═══ TAB 3 ═══════════════════════════════════════════════════════════════════
NLP_LABELS     = ["Crypto Scam", "Hate Speech", "NSFW", "Normal Conversation"]
RISK_LABELS    = {"Crypto Scam", "Hate Speech", "NSFW"}
RISK_THRESHOLD = 0.70

with tab3:
    st.header("Tra cứu & Explainable AI (XAI)")
    search = st.text_input("Nhập Username cần kiểm tra:", key="xai_search")

    if search:
        row = df[df['username'] == search]
        if row.empty:
            st.warning("Không tìm thấy tài khoản này trong cơ sở dữ liệu!")
        else:
            row = row.iloc[0]
            is_bot = row['hybrid_cluster'] == -1
            if is_bot:
                st.error(f"🚨 CẢNH BÁO: '{search}' bị đánh dấu Bot/Spam.")
                st.info(f"**Lý do:** {row.get('outlier_reason', 'N/A')}  |  **Nguồn phát hiện:** `{row.get('detection_source', 'N/A')}`")
            else:
                st.success(f"✅ AN TOÀN: '{search}' có hành vi sinh hoạt bình thường.")

            # ── Behavior XAI Chart ──────────────────────────────────────────
            st.subheader("Giải thích Quyết định Hành vi (Behavioral XAI)")
            normal    = df[df['hybrid_cluster'] != -1]
            avg_normal = normal[FEATURES].mean()
            user_vals  = row[FEATURES]
            chart_data = pd.DataFrame({
                'Người thật (Trung bình)': avg_normal.values,
                f'Hành vi của {search}': user_vals.values
            }, index=['Follow Ratio', 'Avg Time Diff (min)', 'Engagement Rate', 'Night Post Ratio', 'Lexical Diversity'])
            st.bar_chart(chart_data)
            st.info("So sánh hành vi của tài khoản được tra cứu so với giá trị trung bình của nhóm người dùng bình thường (raw features, không chuẩn hóa).")

            # ── Deep NLP Content Scanner ────────────────────────────────────
            st.markdown("---")
            st.subheader("🧠 Deep NLP Content Scanner")
            st.markdown("Phân tích nội dung bài đăng thực tế bằng mô hình **Zero-Shot Classification (DistilBART-MNLI)**.")

            if st.button("🔍 Scan Content for Toxicity & Scams", type="primary", key="nlp_scan_btn"):
                posts = get_user_posts(JSON_PATH, search)

                if not posts:
                    st.warning("Không tìm thấy bài đăng nào cho tài khoản này trong dữ liệu thô.")
                else:
                    # Concatenate up to 5 most recent posts for context
                    sample_posts = posts[:5]
                    combined_text = " ".join(sample_posts)[:1500]  # truncate to keep inference fast

                    with st.spinner("🤖 Đang phân tích nội dung bằng mô hình NLP..."):
                        classifier = load_nlp_classifier()
                        result = classifier(
                            combined_text,
                            candidate_labels=NLP_LABELS,
                            multi_label=False
                        )

                    # Build score dict
                    scores = dict(zip(result['labels'], result['scores']))

                    # ── High-risk banners ──────────────────────────────────
                    any_high_risk = False
                    for label in NLP_LABELS:
                        if label in RISK_LABELS and scores.get(label, 0) >= RISK_THRESHOLD:
                            pct = scores[label] * 100
                            st.error(f"⚠️ HIGH RISK: {pct:.1f}% probability of **{label}** detected in content!")
                            any_high_risk = True

                    if not any_high_risk:
                        st.success("✅ Nội dung không phát hiện rủi ro cao (< 70% trên bất kỳ nhãn độc hại nào).")

                    # ── Score progress bars ────────────────────────────────
                    st.markdown("**Điểm phân loại chi tiết (Zero-Shot NLP):**")
                    label_colors = {
                        "Crypto Scam":       "🟠",
                        "Hate Speech":       "🔴",
                        "NSFW":              "🟣",
                        "Normal Conversation": "🟢",
                    }
                    for label in NLP_LABELS:
                        score = scores.get(label, 0.0)
                        icon  = label_colors.get(label, "⚪")
                        col_label, col_bar, col_pct = st.columns([2, 5, 1])
                        with col_label:
                            st.markdown(f"{icon} **{label}**")
                        with col_bar:
                            st.progress(float(score))
                        with col_pct:
                            st.markdown(f"`{score*100:.1f}%`")

                    # ── Sampled posts preview ──────────────────────────────
                    with st.expander(f"📄 Xem {len(sample_posts)} bài đăng được phân tích"):
                        for i, post in enumerate(sample_posts, 1):
                            st.markdown(f"**Post {i}:** {post[:300]}{'...' if len(post) > 300 else ''}")
                            st.markdown("---")

# ═══ TAB 4: BOTNET NETWORK GRAPH ═════════════════════════════════════════════
with tab4:
    st.header("🕸️ Botnet Network Graph")
    st.markdown(
        "Đồ thị mạng lưới tương tác giữa **500 tài khoản**. "
        "**Đỏ** = Bot/Anomaly, **Xanh** = Normal User. "
        "Cạnh **xanh đậm** = khoảng cách hành vi gần (< 0.1), "
        "cạnh **tím** = cùng hashtag (> 3 tags chung). "
        "Kéo thả, zoom để khám phá."
    )

    with st.spinner("⚙️ Đang xây dựng đồ thị mạng… (lần đầu có thể mất vài giây)"):
        hashtag_map = get_hashtags_map(JSON_PATH)

        graph_html = build_network_graph_html(
            usernames        = df['username'].tolist(),
            hybrid_labels    = df['hybrid_cluster'].tolist(),
            detection_sources= df['detection_source'].tolist(),
            outlier_reasons  = df['outlier_reason'].tolist(),
            feature_matrix   = df[FEATURES].values,
            hashtag_map      = hashtag_map,
            dist_threshold   = 0.10,
            min_shared_tags  = 2,
        )

    components.html(graph_html, height=640, scrolling=False)

    st.markdown("---")
    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Chú thích màu sắc:**")
        st.markdown("🔴 Bot / Anomaly&nbsp;&nbsp;&nbsp;🟢 Normal User")
    with col_r:
        st.markdown("**Chú thích cạnh:**")
        st.markdown("🔵 Behavioral proximity (dist < 0.10)&nbsp;&nbsp;&nbsp;🟣 Shared hashtags (> 2 tags)")
