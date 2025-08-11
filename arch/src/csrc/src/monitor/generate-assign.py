# Python脚本生成1024个assign语句并写入文件

# 定义数组的最大索引
max_index = 1023

# 打开文件准备写入
with open('assign_statements.txt', 'w') as file:
    # 打印每个assign语句到文件
    for i in range(max_index + 1):
        file.write(f'assign line_data_{i} = line_data[{i}];\n')

print('Assign语句已生成并写入到assign_statements.txt文件中。')