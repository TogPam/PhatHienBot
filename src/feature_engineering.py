import json
import numpy as np
import pandas as pd
import re
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def load_and_transform_data(json_path):

    lexical_diversity = 1.0 
    all_text = ""

    # 1. Đọc dữ liệu JSON thô
    with open(json_path, 'r', encoding='utf-8') as f:
        raw_users = json.load(f)
        
    processed_features = []
    
    for user in raw_users:
        user_id = user['id']
        username = user['username']
        is_bot_declared = 1 if user['bot'] else 0  # Cờ bot có sẵn của Mastodon để đối chứng sau này
        
        # --- Đặc trưng 1: Tỷ lệ Follow (Follow Ratio) ---
        # Tránh lỗi chia cho 0 nếu followers_count = 0
        followers = user['followers_count']
        following = user['following_count']
        follow_ratio = following / (followers + 1)
        
        # Phân tích danh sách bài đăng của user
        posts = user['posts']
        total_posts_retrieved = len(posts)
        
        # Khởi tạo các giá trị mặc định nếu user chưa có bài đăng nào
        avg_time_diff_minutes = 1440.0  # Mặc định 1 ngày nếu không có bài đăng liên tiếp
        engagement_rate = 0.0
        night_post_ratio = 0.0
        lexical_diversity = 1.0
        
        if total_posts_retrieved > 0:
            # Chuyển đổi thời gian đăng bài sang kiểu dữ liệu datetime của Python
            timestamps = []
            total_engagements = 0
            night_posts_count = 0
            all_words = []
            
            for post in posts:
                content = post.get('content', '')
                all_text += " " + content
                dt = pd.to_datetime(post['created_at']).tz_localize(None)
                timestamps.append(dt)
                
                # Tính tổng tương tác của bài post đó (like + share + reply)
                total_engagements += (post['favourites_count'] + post['reblogs_count'] + post['replies_count'])
                
                # Check xem bài đăng có nằm trong khung giờ muộn (1h sáng - 5h sáng) không
                if 1 <= dt.hour <= 5:
                    night_posts_count += 1
                
                # Tách từ để tính độ đa dạng từ vựng
                words = re.findall(r'\w+', content.lower())
                all_words.extend(words)
            
            # --- Đặc trưng 2: Tần suất đăng bài (Posting Frequency) ---
            if total_posts_retrieved > 1:
                # Sắp xếp thời gian tăng dần để tính khoảng cách giữa các bài đăng
                timestamps.sort()
                time_diffs = [ (timestamps[i] - timestamps[i-1]).total_seconds() / 60.0 for i in range(1, len(timestamps)) ]
                avg_time_diff_minutes = np.mean(time_diffs)
            
            # --- Đặc trưng 3: Tỷ lệ tương tác trung bình trên mỗi bài viết ---
            engagement_rate = total_engagements / total_posts_retrieved
            
            # --- Đặc trưng 4: Tỷ lệ hoạt động về đêm ---
            night_post_ratio = night_posts_count / total_posts_retrieved

            # --- Đặc trưng 5: Độ đa dạng từ vựng ---
            if len(all_words) > 0:
                lexical_diversity = len(set(all_words)) / len(all_words)
 
        # Gom tất cả vào một bản ghi dữ liệu mẫu
        processed_features.append({
            'user_id': user_id,
            'username': username,
            'is_bot_declared': is_bot_declared, # Giữ lại để làm nhãn đối chứng (Ground Truth)
            'follow_ratio': follow_ratio,
            'avg_time_diff_minutes': avg_time_diff_minutes,
            'engagement_rate': engagement_rate,
            'night_post_ratio': night_post_ratio,
            'lexical_diversity': lexical_diversity
        })
        
    # 2. Chuyển đổi thành Pandas DataFrame để dễ xử lý hình học
    df = pd.DataFrame(processed_features)
    return df

if __name__ == '__main__':
    # Chạy thử nghiệm hàm trích xuất
    df_raw = load_and_transform_data('data/mastodon_dataset.json')
    
    print("--- 5 Dòng dữ liệu Vector Đặc trưng vừa trích xuất ---")
    print(df_raw[['username', 'follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity']].head())
    
    # 3. Thực hiện Feature Scaling (Chuẩn hóa dữ liệu)
    # Vì khoảng cách thời gian (phút) có thể lên tới hàng nghìn, trong khi tỷ lệ đêm chỉ từ 0 -> 1
    # Nếu không chuẩn hóa, thuật toán gom cụm sẽ bị lệch hoàn toàn theo cột thời gian.
    features_to_scale = ['follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity']
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_raw[features_to_scale])
    
    # Tạo DataFrame mới lưu trữ dữ liệu đã chuẩn hóa
    df_scaled = pd.DataFrame(scaled_data, columns=features_to_scale)
    df_scaled.insert(0, 'username', df_raw['username'])
    df_scaled.insert(1, 'is_bot_declared', df_raw['is_bot_declared'])
    
    # Lưu file đã tiền xử lý để chuẩn bị cho Bước 2 chạy Mô hình Gom cụm
    df_scaled.to_csv('data/mastodon_features_scaled.csv', index=False)
    print("\n Tiền xử lý hoàn tất! File 'data/mastodon_features_scaled.csv' đã sẵn sàng cho thuật toán.")
