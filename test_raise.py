def check_length(data, expected_len):
    if len(data) != expected_len:
        raise Exception(f'len(data) is {len(data)}, expected {expected_len}!')

print('第1步: 程序开始')

print('第2步: 准备调用 check_length...')
check_length([1, 2, 3], 5)  # 这里会抛出异常，没人捕获

print('第3步: 这行永远不会执行！')  # 程序已经在上面终止了
