import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import os
import json
import re
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA

# Configure page layout and style
st.set_page_config(page_title="Social Bot Detector Dashboard", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

# Load and process unscaled data from JSON
@st.cache_data
def get_processed_raw_data(json_path):
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_users = json.load(f)
        
    processed_features = []
    
    for user in raw_users:
        user_id = user['id']
        username = user['username']
        is_bot_declared = 1 if user['bot'] else 0
        
        followers = user['followers_count']
        following = user['following_count']
        follow_ratio = following / (followers + 1)
        
        posts = user['posts']
        total_posts_retrieved = len(posts)
        
        avg_time_diff_minutes = 1440.0
        engagement_rate = 0.0
        night_post_ratio = 0.0
        lexical_diversity = 1.0
        
        if total_posts_retrieved > 0:
            timestamps = []
            total_engagements = 0
            night_posts_count = 0
            all_words = []
            
            for post in posts:
                content = post.get('content', '')
                dt = pd.to_datetime(post['created_at']).tz_localize(None)
                timestamps.append(dt)
                total_engagements += (post['favourites_count'] + post['reblogs_count'] + post['replies_count'])
                if 1 <= dt.hour <= 5:
                    night_posts_count += 1
                words = re.findall(r'\w+', content.lower())
                all_words.extend(words)
            
            if total_posts_retrieved > 1:
                timestamps.sort()
                time_diffs = [ (timestamps[i] - timestamps[i-1]).total_seconds() / 60.0 for i in range(1, len(timestamps)) ]
                avg_time_diff_minutes = np.mean(time_diffs)
            
            engagement_rate = total_engagements / total_posts_retrieved
            night_post_ratio = night_posts_count / total_posts_retrieved
            if len(all_words) > 0:
                lexical_diversity = len(set(all_words)) / len(all_words)
 
        processed_features.append({
            'user_id': user_id,
            'username': username,
            'is_bot_declared': is_bot_declared,
            'follow_ratio': follow_ratio,
            'avg_time_diff_minutes': avg_time_diff_minutes,
            'engagement_rate': engagement_rate,
            'night_post_ratio': night_post_ratio,
            'lexical_diversity': lexical_diversity
        })
        
    return pd.DataFrame(processed_features)

# Define file paths dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_PATH = os.path.join(BASE_DIR, 'data', 'mastodon_dataset.json')

df_raw = get_processed_raw_data(JSON_PATH)

if df_raw is None:
    st.error("⚠️ Không tìm thấy file `data/mastodon_dataset.json`. Vui lòng chạy crawler trước!")
    st.stop()

# --- SIDEBAR INTERACTIVE HYPERPARAMETER TUNING ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2721/2721272.png", width=80)
    st.title("🛡️ Bot Detector Controls")
    st.markdown("Cấu hình trực tiếp các thuật toán Machine Learning.")
    st.markdown("---")
    
    st.header("⚙️ K-Means")
    n_clusters = st.slider("Số lượng cụm (n_clusters)", min_value=2, max_value=10, value=3, step=1)
    
    st.header("⚙️ DBSCAN")
    eps = st.slider("Bán kính lân cận (eps)", min_value=0.1, max_value=2.5, value=0.8, step=0.1)
    min_samples = st.slider("Mẫu tối thiểu (min_samples)", min_value=1, max_value=10, value=2, step=1)
    
    st.header("⚙️ Isolation Forest")
    contamination = st.slider("Tỷ lệ dị biệt (contamination)", min_value=0.01, max_value=0.30, value=0.10, step=0.01)

# --- RUN DYNAMIC ML MODELS ---
features = ['follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity']
X_raw = df_raw[features].values

# Scale features on the fly
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# 1. KMeans
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
df_raw['kmeans_cluster'] = kmeans_labels

# 2. DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(X_scaled)
df_raw['dbscan_cluster'] = dbscan_labels
dbscan_outliers = np.sum(dbscan_labels == -1)

# 3. Isolation Forest
iso = IsolationForest(contamination=contamination, random_state=42)
iso_labels = iso.fit_predict(X_scaled)
df_raw['iso_outlier'] = iso_labels
iso_outliers = np.sum(iso_labels == -1)

# Calculate PCA dynamically for scatter plot
pca = PCA(n_components=2)
pca_features = pca.fit_transform(X_scaled)
df_raw['pca_1'] = pca_features[:, 0]
df_raw['pca_2'] = pca_features[:, 1]
df_raw['is_bot'] = df_raw['dbscan_cluster'].apply(lambda x: 'Bot/Anomaly' if x == -1 else 'Normal User')

# Calculate performance metrics
# KMeans
kmeans_sil = f"{silhouette_score(X_scaled, kmeans_labels):.4f}"
kmeans_db = f"{davies_bouldin_score(X_scaled, kmeans_labels):.4f}"

# DBSCAN
unique_db = set(dbscan_labels)
if len(unique_db - {-1}) >= 1 and len(unique_db) > 1:
    dbscan_sil = f"{silhouette_score(X_scaled, dbscan_labels):.4f}"
    dbscan_db = f"{davies_bouldin_score(X_scaled, dbscan_labels):.4f}"
else:
    dbscan_sil = "N/A"
    dbscan_db = "N/A"

# Isolation Forest
unique_iso = set(iso_labels)
if len(unique_iso) > 1:
    iso_sil = f"{silhouette_score(X_scaled, iso_labels):.4f}"
    iso_db = f"{davies_bouldin_score(X_scaled, iso_labels):.4f}"
else:
    iso_sil = "N/A"
    iso_db = "N/A"

# --- MAIN DASHBOARD DISPLAY ---
st.title("🛡️ Real-Time Social Bot Detection Dashboard")
st.markdown("Hệ thống phân tích hành vi bất thường chạy trực tiếp thuật toán học máy.")

tab1, tab2, tab3 = st.tabs([
    "📈 Dashboard & Analytics", 
    "🗂️ Database & Blacklist", 
    "🔍 Account Lookup (XAI)"
])

# ==========================================
# TAB 1: DASHBOARD & ANALYTICS
# ==========================================
with tab1:
    st.header("Thống kê Tổng quan (Thời Gian Thực)")
    
    total_users = len(df_raw)
    bot_ratio = (dbscan_outliers / total_users) * 100 if total_users > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Tổng Tài Khoản", total_users)
    with col2:
        st.metric("Bot Phát Hiện (DBSCAN)", dbscan_outliers, delta=f"{bot_ratio:.1f}%", delta_color="inverse")
    with col3:
        st.metric("Outliers (Isolation Forest)", iso_outliers)
    with col4:
        st.metric("K-Means Clusters", n_clusters)
        
    st.markdown("---")
    
    # 2D Behavior Scatter Plot using dynamic PCA
    st.subheader("2D Behavior Projection Plot (Dynamic PCA)")
    st.markdown("Biểu đồ chiếu dữ liệu đa chiều xuống không gian 2 chiều bằng giải thuật PCA để phân tích trực quan ranh giới quyết định.")
    
    scatter_chart = alt.Chart(df_raw).mark_circle(size=60).encode(
        x=alt.X('pca_1', title='PCA 1 (Behavior Trend)'),
        y=alt.Y('pca_2', title='PCA 2 (Posting Activity)'),
        color=alt.Color('is_bot', scale=alt.Scale(domain=['Normal User', 'Bot/Anomaly'], range=['#00cc96', '#ff4b4b']), title='Label (DBSCAN)'),
        tooltip=['username', 'follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity']
    ).interactive().properties(height=450)
    
    st.altair_chart(scatter_chart, use_container_width=True)

    st.markdown("---")
    
    # Advanced Performance Metrics Table
    st.subheader("📊 Bảng So Sánh Hiệu Suất Mô Hình Thuật Toán")
    
    metrics_df = pd.DataFrame({
        "Model": [f"K-Means (K={n_clusters})", f"DBSCAN (eps={eps}, min_samples={min_samples})", f"Isolation Forest (cont={contamination:.2f})"],
        "Outliers Detected": ["N/A", dbscan_outliers, iso_outliers],
        "Silhouette Score": [kmeans_sil, dbscan_sil, iso_sil],
        "Davies-Bouldin Index": [kmeans_db, dbscan_db, iso_db]
    })
    
    st.table(metrics_df)
    
    st.markdown("""
    ### 💡 Giải thích các Chỉ số Đo lường Hiệu năng Cụm (Clustering Metrics Explained)
    
    * **Silhouette Score (Hệ số Dáng hình):**
      * Có giá trị từ **-1 đến 1**.
      * Giá trị **càng cao (gần 1)** thể hiện các cụm được phân tách rõ ràng, các điểm trong cụm rất gần nhau và cách biệt hẳn các cụm khác.
      * Hệ số âm hoặc gần 0 nghĩa là các cụm bị chồng lấn đáng kể hoặc các điểm bị gán sai cụm.
    
    * **Davies-Bouldin Index (Chỉ số Davies-Bouldin):**
      * Có giá trị tối thiểu là **0**.
      * Giá trị **càng thấp** thể hiện hiệu năng gom cụm tốt hơn (cụm chặt chẽ bên trong và xa nhau bên ngoài).
      * Phản ánh tỷ số giữa khoảng cách nội cụm và khoảng cách liên cụm.
    """)

# ==========================================
# TAB 2: DATABASE & BLACKLIST
# ==========================================
with tab2:
    st.header("Quản lý Dữ liệu")
    
    st.subheader("📁 Tất cả tài khoản đã quét (All Scanned Users)")
    st.dataframe(df_raw, use_container_width=True, height=300)
    
    st.markdown("---")
    st.subheader("☠️ Danh sách Đen (Detected Bots Blacklist)")
    st.markdown("Bảng phân loại các tài khoản dị thường cùng với bằng chứng hành vi cụ thể.")
    
    # Filter anomalies from DBSCAN
    blacklist_df = df_raw[df_raw['dbscan_cluster'] == -1].copy()
    
    # Function to generate dynamic evidence reasons based on raw thresholds
    def get_outlier_reasons(row):
        reasons = []
        if row['night_post_ratio'] > 0.8:
            reasons.append("High Night Activity (>80% đêm)")
        if row['lexical_diversity'] < 0.2:
            reasons.append("Repetitive Content/Vocabulary (Từ vựng <20% đa dạng)")
        if row['follow_ratio'] > 15.0:
            reasons.append("Spammy Following (Follows/Followers > 15)")
        if row['avg_time_diff_minutes'] < 5.0:
            reasons.append("Automation/Botting (Đăng bài <5 phút/lần)")
        if row['engagement_rate'] == 0.0 and row['follow_ratio'] > 5.0:
            reasons.append("Zero engagement with high following")
            
        if not reasons:
            reasons.append("General Behavior Outlier (Hành vi dị biệt tổng thể)")
            
        return " | ".join(reasons)

    if not blacklist_df.empty:
        blacklist_df['Reason for Outlier Flag'] = blacklist_df.apply(get_outlier_reasons, axis=1)
        
        # Style and render
        styled_blacklist = blacklist_df[['username', 'follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity', 'Reason for Outlier Flag']]
        st.dataframe(styled_blacklist.style.set_properties(**{'background-color': '#2a0000', 'color': '#ff9999'}), use_container_width=True, height=300)
        
        # Download button
        csv = styled_blacklist.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Tải xuống Blacklist (.CSV)",
            data=csv,
            file_name='bot_blacklist.csv',
            mime='text/csv',
        )
    else:
        st.success("✅ Tuyệt vời! Không phát hiện tài khoản bot nào với cấu hình DBSCAN hiện tại.")

# ==========================================
# TAB 3: ACCOUNT LOOKUP (XAI)
# ==========================================
with tab3:
    st.header("Tra cứu & Explainable AI (XAI)")
    search_username = st.text_input("Nhập Username cần kiểm tra (Ví dụ: copy 1 tên bên Tab 2 bỏ vào đây):")
    
    if search_username:
        user_data = df_raw[df_raw['username'] == search_username]
        
        if user_data.empty:
            st.warning("Không tìm thấy tài khoản này trong cơ sở dữ liệu!")
        else:
            user_data = user_data.iloc[0]
            is_bot = user_data['dbscan_cluster'] == -1
            
            if is_bot:
                st.error(f"🚨 CẢNH BÁO: '{search_username}' bị đánh dấu là Bot/Spam.")
            else:
                st.success(f"✅ AN TOÀN: '{search_username}' có hành vi sinh hoạt bình thường.")
            
            st.subheader("Giải thích Quyết định thuật toán")
            
            normal_users = df_raw[df_raw['dbscan_cluster'] != -1]
            normal_avg = normal_users[['follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity']].mean()
            user_stats = user_data[['follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity']]
            
            chart_data = pd.DataFrame({
                'Người thật (Trung bình)': normal_avg.values,
                f'Hành vi của {search_username}': user_stats.values
            }, index=['Tỷ lệ Follow', 'Khoảng cách Bài đăng', 'Tương tác TB', 'Hoạt động Đêm', 'Độ đa dạng từ vựng'])
            
            st.bar_chart(chart_data)
            st.info("Biểu đồ so sánh hành vi (unscaled) của tài khoản được tra cứu so với hành vi trung bình của nhóm người dùng bình thường.")
