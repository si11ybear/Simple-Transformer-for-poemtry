def merge_text_files(file1, file2, file3, file4, output_file):
    # 创建一个空列表来存储所有文件的内容
    combined_content = []

    # 读取第一个文件
    with open(file1, 'r', encoding='utf-8') as f1:
        combined_content.append(f1.read())

    # 读取第二个文件
    with open(file2, 'r', encoding='utf-8') as f2:
        combined_content.append(f2.read())

    # 读取第三个文件
    with open(file3, 'r', encoding='utf-8') as f3:
        combined_content.append(f3.read())

    # 读取第四个文件
    with open(file4, 'r', encoding='utf-8') as f4:
        combined_content.append(f4.read())

    # 将所有内容合并并写入输出文件
    with open(output_file, 'w', encoding='utf-8') as output:
        output.write("\n".join(combined_content))

# 使用示例
merge_text_files('poetry曹操诗集.txt', 'poetry纳兰性德.txt', 'poetry五代诗词.txt', 'poetry诗经.txt','ci.txt')