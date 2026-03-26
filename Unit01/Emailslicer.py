# 1. 取得使用者 Email 並去前後空白
email = input("What is your email address?: ").strip()

# 2. 核心邏輯：利用 @ 進行切割
# 使用 index() 找到 @ 的位置，或是直接用 split('@')
user_name = email[:email.index('@')]
domain_name = email[email.index('@') + 1:]

# 3. 格式化輸出基礎訊息
output = "Your username is '{}' and your domain name is '{}'".format(user_name, domain_name)
print(output)

# 4. 進階判斷：字典對照與自定義域名處理
domain_dict = {
    'gmail.com': 'Google',
    'yahoo.com.tw': 'Yahoo',
    'ntu.edu.tw': '臺大',
    'hotmail.com.tw': 'Hotmail',
    'hotmail.com': 'Hotmail'
}

# 判斷 domain_name 是否在字典的「鍵 (Key)」當中
if domain_name in domain_dict:
    # 取得字典中對應的名稱（例如：Google）
    output = "這是註冊在 {} 之下的 Email 地址".format(domain_dict[domain_name])
else:
    # 自定義域名處理：取 domain_name 中第一個點之前的文字（例如：myfantasy.com -> myfantasy）
    custom_name = domain_name.split('.')[0]
    output = "這是在 {} 之下自定義域".format(custom_name)

print(output)