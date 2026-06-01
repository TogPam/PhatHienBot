from mastodon import Mastodon

# 1. Đăng ký App với server
Mastodon.create_app(
    'DataMiningProject',
    api_base_url='https://mastodon.social',
    to_file='data/mastodon_clientcred.secret'
)

# 2. Khởi tạo instance với thông tin vừa tạo
mastodon = Mastodon(
    client_id='data/mastodon_clientcred.secret',
    api_base_url='https://mastodon.social'
)

# 3. Lấy URL cấp quyền
print("Hãy copy link sau, dán vào trình duyệt, đăng nhập và nhấn Authorize:")
print(mastodon.auth_request_url())

# 4. Nhập mã trả về để lưu Access Token
auth_code = input("Dán mã code lấy được từ trình duyệt vào đây: ")
mastodon.log_in(
    code=auth_code,
    to_file='data/mastodon_usercred.secret'
)
print("Thành công! Token đã được lưu.")
