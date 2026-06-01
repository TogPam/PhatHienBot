from mastodon import Mastodon
from bs4 import BeautifulSoup
import json
import time
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

mastodon = Mastodon(
    access_token='data/mastodon_usercred.secret',
    api_base_url='https://mastodon.social'
)

def crawl_data():
    dataset = []
    unique_user_ids = set()
    
    dataset_path = 'data/mastodon_dataset.json'
    
    # Load existing dataset if it exists
    if os.path.exists(dataset_path):
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                for user in dataset:
                    unique_user_ids.add(user['id'])
            print(f"✅ Đã tải {len(dataset)} users từ dữ liệu cũ.")
        except Exception as e:
            print(f"⚠️ Lỗi đọc file cũ: {e}. Bắt đầu lại từ đầu.")
            dataset = []
            unique_user_ids = set()

    target_total = 500
    current_count = len(dataset)
    
    if current_count >= target_total:
        print(f"🎉 Dữ liệu đã có đủ hoặc vượt mức {target_total} users ({current_count}). Không cần thu thập thêm.")
        return

    needed = target_total - current_count
    print(f"🚀 Bắt đầu thu thập thêm {needed} users để đạt mục tiêu {target_total}...")
    
    new_user_ids = set()
    hashtags = ['news', 'tech', 'art', 'politics', 'gaming', 'music', 'nature', 'food', 'travel', 'sports', 'crypto', 'science', 'movies', 'books', 'photography', 'programming', 'ai', 'history', 'fitness', 'pets']
    
    for tag in hashtags:
        if len(new_user_ids) >= needed:
            break
            
        print(f"Đang quét hashtag #{tag}...")
        try:
            toots = mastodon.timeline_hashtag(tag, limit=40)
            for toot in toots:
                if toot['account'] is not None:
                    uid = toot['account']['id']
                    if uid not in unique_user_ids and uid not in new_user_ids:
                        new_user_ids.add(uid)
                        
            print(f" -> Cần thu thập thêm: {len(new_user_ids)}/{needed}")
            time.sleep(1)
        except Exception as e:
            print(f"Bỏ qua #{tag} do lỗi: {e}")
            time.sleep(2)

    new_user_ids = list(new_user_ids)[:needed]
    print(f"\n✅ Đã gom đủ {len(new_user_ids)} ID mới. Bắt đầu tải lịch sử hoạt động chi tiết...")

    for i, uid in enumerate(new_user_ids):
        try:
            account = mastodon.account(uid)
            statuses = mastodon.account_statuses(uid, limit=20)
            
            user_data = {
                'id': account['id'],
                'username': account['username'],
                'bot': account['bot'], 
                'followers_count': account['followers_count'],
                'following_count': account['following_count'],
                'statuses_count': account['statuses_count'],
                'created_at': str(account['created_at']),
                'posts': []
            }
            
            for status in statuses:
                raw_html = status['content']
                clean_text = BeautifulSoup(raw_html, "html.parser").get_text() if raw_html else ""
                user_data['posts'].append({
                    'id': status['id'],
                    'created_at': str(status['created_at']),
                    'content': clean_text,
                    'replies_count': status['replies_count'],
                    'reblogs_count': status['reblogs_count'],
                    'favourites_count': status['favourites_count']
                })
                
            dataset.append(user_data)
            unique_user_ids.add(uid)
            
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)
                
            print(f"[{current_count + i + 1}/{target_total}] Đã lưu dữ liệu user: {account['username']}")
            time.sleep(2)
            
        except Exception as e:
            print(f"[{current_count + i + 1}/{target_total}] Bỏ qua user {uid} do lỗi: {e}")
            
    print(f"\n🎉 HOÀN TẤT CHIẾN DỊCH! Dữ liệu tổng cộng {len(dataset)} users đã an toàn trong '{dataset_path}'.")

if __name__ == '__main__':
    crawl_data()
