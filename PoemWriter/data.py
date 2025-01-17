import json
import os
import glob
from opencc import OpenCC

# 创建一个OpenCC对象，用于繁体转简体
cc = OpenCC('t2s')  # 't2s' 表示从繁体（traditional）到简体（simplified）

# 定义输入文件夹路径和输出文件路径
input_folder_path = r'./chinese-poetry-master/元曲'
folder_path = r'data'
output_file_path = os.path.join(folder_path, 'poetry元曲.txt')

# 使用glob查找所有匹配的文件
file_pattern = os.path.join(input_folder_path, '*.json')
matching_files = glob.glob(file_pattern)

# 打开输出的文本文件，准备写入
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    # 遍历所有匹配的文件
    for file_name in matching_files:
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)

            # 遍历每一首诗
            for poem in data:
                title = cc.convert(poem['title'])  # 将标题转换为简体
                paragraphs = [''.join(paragraph) for paragraph in poem['paragraphs']]  # 合并段落内的诗句
                content = cc.convert(''.join(paragraphs))  # 将内容转换为简体
                
                # 用冒号连接标题与内容，并写入文本文件中，一首诗占一行
                outfile.write(f'{title}:{content}\n')

print("All matching files have been processed.")