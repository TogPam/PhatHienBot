import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import warnings
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore') # Tắt các cảnh báo lặt vặt của Python

# 1. Đọc dữ liệu từ Bước 1
df = pd.read_csv('data/mastodon_features_scaled.csv')
features = ['follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio']
X = df[features].values

# 2. Chạy thử K-Means để vẽ đồ thị Cùi chỏ (tìm số cụm tốt nhất)
wcss = []
for i in range(2, 9):
    kmeans_test = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans_test.fit(X)
    wcss.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(2, 9), wcss, marker='o', linestyle='--')
plt.title('Phuong phap cui cho (Elbow Method)')
plt.xlabel('So luong cum (K)')
plt.ylabel('WCSS')
plt.savefig('docs/elbow_method.png')
print("1. Đã vẽ xong đồ thị cùi chỏ. Hãy mở file 'docs/elbow_method.png' để xem.")

# 3. Chạy K-Means chính thức (Tạm chọn K=3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42)
df['kmeans_cluster'] = kmeans.fit_transform(X).argmin(axis=1)

score = silhouette_score(X, df['kmeans_cluster'])
print(f"2. Điểm đánh giá K-Means (K={optimal_k}): {score:.4f}")

# 4. Chạy thuật toán DBSCAN để tìm Tài khoản ảo (Outliers)
dbscan = DBSCAN(eps=0.8, min_samples=2) 
df['dbscan_cluster'] = dbscan.fit_predict(X)

outliers = len(df[df['dbscan_cluster'] == -1])
print(f"3. Thuật toán DBSCAN phát hiện: {outliers} tài khoản bất thường (nghi ngờ Bot/Spam).")

# 5. Lưu kết quả ra file mới
df.to_csv('data/mastodon_clustered_results.csv', index=False)
print("4. Đã lưu kết quả gán nhãn vào file 'data/mastodon_clustered_results.csv'.")


# ==========================================
# THUẬT TOÁN 1: DBSCAN
# ==========================================
dbscan = DBSCAN(eps=0.8, min_samples=2) 
df['dbscan_cluster'] = dbscan.fit_predict(X)
dbscan_bots = df[df['dbscan_cluster'] == -1]

# ==========================================
# THUẬT TOÁN 2: ISOLATION FOREST (Rừng Cô Lập)
# ==========================================
# contamination=0.1 nghĩa là ta giả định có khoảng 10% dữ liệu là Bot
iso_forest = IsolationForest(contamination=0.1, random_state=42)
# Isolation Forest trả về -1 cho Outlier (Bot) và 1 cho Normal (Người)
df['iso_outlier'] = iso_forest.fit_predict(X) 
iso_bots = df[df['iso_outlier'] == -1]

# ==========================================
# PHÂN TÍCH ĐỐI CHIẾU KẾT QUẢ (CROSS-VALIDATION)
# ==========================================
print("\n" + "="*50)
print("🏆 KẾT QUẢ ĐỐI ĐẦU THUẬT TOÁN")
print("="*50)
print(f"DBSCAN phát hiện: {len(dbscan_bots)} bots.")
print(f"Isolation Forest phát hiện: {len(iso_bots)} bots.")

# Tìm những tài khoản bị CẢ 2 thuật toán đánh dấu là Bot (Độ tin cậy tuyệt đối)
intersection_bots = df[(df['dbscan_cluster'] == -1) & (df['iso_outlier'] == -1)]
print(f"Sự đồng thuận (Cả 2 đều bắt được): {len(intersection_bots)} bots.")

# Kiểm tra xem 10 con TEST_BOT_SPAM có bị tóm gọn không
caught_test_bots = intersection_bots[intersection_bots['username'].str.contains("TEST_BOT")]
print(f"Kiểm chứng nhãn: Đã tóm được {len(caught_test_bots)}/10 Bot giả lập đã tiêm vào.")
print("="*50)

# Lưu kết quả
df.to_csv('data/mastodon_clustered_results.csv', index=False)
