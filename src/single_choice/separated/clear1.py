import re


def add_newline_before_numbers(md_content):
    # 匹配中文编号（如一. 二. 三.）
    pattern_chinese = re.compile(r'([一二三四五六七八九十]+[、.])')
    # 匹配阿拉伯数字编号（如1. 2. 3.）
    pattern_arabic = re.compile(r'(\d+[、.](?!\d))')
    # 匹配英文字母编号（如A. B. C.）
    pattern_english = re.compile(r'([A-Za-z]\.)')

    pattern_jiexi = re.compile(r'(【解析】)')

    # 在匹配到的编号前添加换行符
    md_content = pattern_chinese.sub(r'\n\1', md_content)
    md_content = pattern_arabic.sub(r'\n\1', md_content)
    md_content = pattern_english.sub(r'\n\1', md_content)
    md_content = pattern_jiexi.sub(r'\n\1', md_content)

    # 去除可能多余的多个连续换行符，防止出现空行
    md_content = re.sub(r'\n+', '\n', md_content)

    return md_content


def process_markdown_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    new_content = add_newline_before_numbers(content)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(new_content)


# 示例用法
input_file = 'file.txt'
output_file = 'output2.txt'
process_markdown_file(input_file, output_file)
