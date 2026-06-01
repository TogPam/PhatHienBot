import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def analyze_clusters():
    # 1. Đọc dữ liệu đã được gán nhãn cụm từ Bước 2
    df = pd.read_csv('data/mastodon_clustered_results.csv')
    
    # 2. VẼ CHÂN DUNG CÁC CỤM K-MEANS (PROFILING)
    features = ['follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio']
    cluster_profile = df.groupby('kmeans_cluster')[features].mean()
    
    print("="*60)
    print(" ĐẶC ĐIỂM TRUNG BÌNH CỦA TỪNG NHÓM (K-MEANS)")
    print("="*60)
    print(cluster_profile)
    print("\n*Ghi chú: Vì dữ liệu đã chuẩn hóa (Standard Scaler):")
    print("   > 0 : Cao hơn mức trung bình của toàn bộ User.")
    print("   < 0 : Thấp hơn mức trung bình của toàn bộ User.")
    print("="*60)

    # 3. TRỰC QUAN HÓA DỮ LIỆU (PCA SCATTER PLOT)
    # Máy tính hiểu 4 chiều (features), nhưng con người chỉ nhìn được đồ thị 2D. 
    # Ta dùng thuật toán PCA để nén 4 chiều xuống 2 chiều (pca_1 và pca_2).
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(df[features])
    df['pca_1'] = pca_features[:, 0]
    df['pca_2'] = pca_features[:, 1]

    plt.figure(figsize=(10, 6))
    
    # Vẽ đồ thị phân tán (Scatter Plot)
    scatter = plt.scatter(df['pca_1'], df['pca_2'], 
                          c=df['kmeans_cluster'], 
                          cmap='viridis', 
                          alpha=0.7, 
                          edgecolors='black', 
                          linewidth=0.5)
    
    plt.colorbar(scatter, label='Cụm K-Means (0, 1, 2)')
    plt.title('Bieu do Khong gian Nguoi dung (PCA - 2D)')
    plt.xlabel('Thanh phan PCA 1 (Xu huong Hanh vi)')
    plt.ylabel('Thanh phan PCA 2 (Tan suat & Tuong tac)')
    
    plt.savefig('docs/cluster_visualization_2d.png')
    print("\n-> 1. Đã vẽ xong biểu đồ! Hãy mở file 'docs/cluster_visualization_2d.png' để xem các cụm phân tách nhau thế nào.")

    # 4. XUẤT DANH SÁCH ĐEN (BLACKLIST) DÀNH CHO DOANH NGHIỆP TỪ DBSCAN
    # DBSCAN đánh dấu nhiễu (outliers) là -1. Đây là những kẻ có hành vi cực kỳ dị thường.
    blacklist_df = df[df['dbscan_cluster'] == -1]
    
    # Chỉ trích xuất tên tài khoản để đưa cho bộ phận Quảng cáo
    export_columns = ['username', 'follow_ratio', 'night_post_ratio']
    blacklist_df[export_columns].to_csv('data/bot_blacklist.csv', index=False)
    
    print(f"-> 2. ỨNG DỤNG THỰC TẾ: Đã xuất danh sách {len(blacklist_df)} tài khoản ảo ra file 'data/bot_blacklist.csv'.")
    print("      (Doanh nghiệp có thể upload file này lên Facebook/Google Ads để chặn hiển thị quảng cáo).")

if __name__ == '__main__':
    analyze_clusters()
