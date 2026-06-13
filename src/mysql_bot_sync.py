import mysql.connector
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest

def sync_mysql_bots():
    print("="*50)
    print(" BẮT ĐẦU PHÂN TÍCH VÀ CẬP NHẬT BOT TỪ MYSQL")
    print("="*50)
    
    try:
        # 1. Kết nối CSDL
        db = mysql.connector.connect(
            host="localhost", user="root", password="", database="mxh_db"
        )
        cursor = db.cursor(dictionary=True)
        
        # 2. Lấy dữ liệu người dùng
        cursor.execute("SELECT id, username, followers_count, following_count, statuses_count FROM users")
        users = cursor.fetchall()
        
        if not users:
            print("Không có người dùng nào trong cơ sở dữ liệu.")
            return

        records = []
        for u in users:
            user_id = u['id']
            followers = u['followers_count']
            following = u['following_count']
            follow_ratio = following / (followers + 1)
            
            # Lấy tất cả bài viết của user này
            cursor.execute("SELECT content, created_at, replies_count, reblogs_count, favourites_count FROM posts WHERE user_id = %s", (user_id,))
            posts = cursor.fetchall()
            
            n = len(posts)
            avg_time_diff_minutes = 1440.0
            engagement_rate = 0.0
            night_post_ratio = 0.0
            lexical_diversity = 1.0
            
            if n > 0:
                timestamps, engagements, night_count, all_words = [], 0, 0, []
                for p in posts:
                    content = p['content'] if p['content'] else ''
                    
                    # Cố gắng parse thời gian từ CSDL
                    try:
                        dt = pd.to_datetime(p['created_at']).tz_localize(None)
                        timestamps.append(dt)
                        if 1 <= dt.hour <= 5:
                            night_count += 1
                    except:
                        pass # Bỏ qua nếu format thời gian bị lỗi
                    
                    engagements += (p['favourites_count'] or 0) + (p['reblogs_count'] or 0) + (p['replies_count'] or 0)
                    all_words.extend(re.findall(r'\w+', content.lower()))
                
                if n > 1 and len(timestamps) > 1:
                    timestamps.sort()
                    diffs = [(timestamps[i] - timestamps[i-1]).total_seconds() / 60.0 for i in range(1, len(timestamps))]
                    avg_time_diff_minutes = float(np.mean(diffs))
                    
                engagement_rate = engagements / n
                night_post_ratio = night_count / n
                if all_words:
                    lexical_diversity = len(set(all_words)) / len(all_words)
                    
            records.append({
                'user_id': user_id,
                'username': u['username'],
                'follow_ratio': follow_ratio,
                'avg_time_diff_minutes': avg_time_diff_minutes,
                'engagement_rate': engagement_rate,
                'night_post_ratio': night_post_ratio,
                'lexical_diversity': lexical_diversity
            })
            
        df = pd.DataFrame(records)
        features = ['follow_ratio', 'avg_time_diff_minutes', 'engagement_rate', 'night_post_ratio', 'lexical_diversity']
        
        # 3. Phân loại BOT (Sử dụng Isolation Forest giống app.py nhưng cho dữ liệu tự tạo)
        # Bỏ qua chuẩn hóa nếu dùng thuật toán rule-based hoặc isolation forest cơ bản
        if len(df) > 1:
            iso = IsolationForest(contamination=0.1, random_state=42)
            df['is_bot'] = iso.fit_predict(df[features].values) # -1 là dị thường (BOT), 1 là bình thường
            
            # Reset toàn bộ bot về 0 trước
            cursor.execute("UPDATE users SET bot = 0")
            
            # Cập nhật bot = 1 cho các user bị đánh dấu là -1
            bot_users = df[df['is_bot'] == -1]['user_id'].tolist()
            if bot_users:
                format_strings = ','.join(['%s'] * len(bot_users))
                update_query = f"UPDATE users SET bot = 1 WHERE id IN ({format_strings})"
                cursor.execute(update_query, tuple(bot_users))
                db.commit()
                print(f"-> Đã phát hiện và gắn nhãn BOT cho {len(bot_users)} tài khoản ảo từ Database tự tạo!")
            else:
                db.commit()
                print("-> Hệ thống an toàn, chưa phát hiện tài khoản ảo nào!")
        else:
            print("Cần ít nhất 2 user để chạy thuật toán phân tích.")

        cursor.close()
        db.close()
        print("="*50)
        
    except Exception as e:
        print("Lỗi:", e)

if __name__ == '__main__':
    sync_mysql_bots()
