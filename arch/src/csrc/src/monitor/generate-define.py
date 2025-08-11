# Python脚本生成Verilog模块输出声明并写入文件

# 定义数组的最大索引
max_index = 1023

# 打开文件准备写入
with open('output_declarations.txt', 'w') as file:
    # 打印每个output声明到文件
    for i in range(max_index + 1):
        file.write(f'output [15:0] line_data_{i},\n')

# 删除文件中的最后一个逗号和换行符
with open('output_declarations.txt', 'r+') as file:
    content = file.read()
    # 确保只删除最后一个逗号和换行符
    if content and content[-2:] == ',\n':
        content = content[:-2]
    file.seek(0)
    file.write(content)
    file.truncate()

print('输出声明已生成并写入到output_declarations.txt文件中。')