import re
import os

def add_newline_before_numbers(md_content):
    # 匹配中文编号（如一. 二. 三.）
    pattern_chinese = re.compile(r'([一二三四五六七八九十]+[、.])')
    # 匹配阿拉伯数字编号（如1. 2. 3.）
    pattern_arabic = re.compile(r'(\d+[、.．](?!\d))')
    # 匹配英文字母编号（如A. B. C.）
    pattern_english = re.compile(r'([A-Za-z]+[.．])')

    pattern_jiexi = re.compile(r'(【解析】|解析)')

    pattern_daan = re.compile(r'(【答案】|答案)')

    # 在匹配到的编号前添加换行符
    md_content = pattern_chinese.sub(r'\n\1', md_content)
    md_content = pattern_arabic.sub(r'\n\1', md_content)
    md_content = pattern_english.sub(r'\n\1', md_content)
    md_content = pattern_jiexi.sub(r'\n\1', md_content)
    md_content = pattern_daan.sub(r'\n\1', md_content)

    # 去除可能多余的多个连续换行符，防止出现空行
    md_content = re.sub(r'\n+', '\n', md_content)

    return md_content


def remove_page_references(md_content):
    # 匹配页面引用（如P1, P2, P3-4）
    pattern_page = re.compile(r'\bP\d+(-\d+)?\b')

    # 替换匹配到的页面引用为空字符串
    md_content = pattern_page.sub('', md_content)


    return md_content


def process_markdown_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        content = file.read()

    new_content = add_newline_before_numbers(content)
    new_content = remove_page_references(new_content)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(new_content)


def main (paper_path, file_name):
    input_file = paper_path

    directory = f'paper/{file_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    output_file = f'paper/{file_name}/{file_name}.txt'
    process_markdown_file(input_file, output_file)
