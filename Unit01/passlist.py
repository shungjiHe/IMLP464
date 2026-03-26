m = {'柯南', '灰原', '步美', '美環', '光彦'}
e = {'柯南', '灰原', '丸尾', '野口', '步美'}

# 1. 兩者皆及格 (交集)
both_pass = list(m & e)
both_pass.sort()

# 2. 數學及格但英文不及格 (數學 - 英文)
math_only = list(m - e)
math_only.sort()

# 3. 英文及格但數學不及格 (英文 - 數學)
english_only = list(e - m)
english_only.sort()

# 輸出結果
print(f"數學及格但英文不及格：{math_only}")
print(f"英文及格但數學不及格：{english_only}")
print(f"兩者皆及格名單：{both_pass}")