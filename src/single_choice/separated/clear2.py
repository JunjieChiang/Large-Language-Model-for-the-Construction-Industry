import re


def clean_brackets_spaces(text):
    # 移除小括号中的多余空格
    text = re.sub(r'\(\s+|\s+\)', lambda m: '(' if m.group(0).startswith('(') else ')', text)
    return text


def clean_lines(text):
    lines = text.split('\n')
    cleaned_lines = []
    buffer = ''
    option_buffer = ''
    option_started = False

    for line in lines:
        line = line.strip()

        # 检查是否是题目编号或 "ID:"
        if re.match(r'^(\d+\.\s|ID:)', line):
            if buffer:
                cleaned_lines.append(buffer)
            buffer = line
            option_buffer = ''
            option_started = False

        # 检查是否是选项（A., B., C., D.）
        elif re.match(r'^[A-E]\.\s', line):
            if option_buffer:
                buffer += ' ' + option_buffer
            option_buffer = line
            option_started = True

        # 其他行追加到选项缓冲区
        else:
            if option_started:
                option_buffer += ' ' + line
            else:
                buffer += '\n' + line if buffer else line

    if option_buffer:
        buffer += ' ' + option_buffer
    if buffer:
        cleaned_lines.append(buffer)

    cleaned_text = '\n'.join(line for line in cleaned_lines if line.strip() != '')
    return cleaned_text


def clean_options_spacing(text):
    text = re.sub(r'([A-E]\.)\s+', r'\1 ', text)
    return text


def clean_markdown(text):
    text = re.sub(r'[|\-]', '', text)  # 移除所有的竖线和破折号
    text = clean_brackets_spaces(text)  # 移除小括号中的多余空格
    text = clean_lines(text)  # 分割并清理文本行
    text = clean_options_spacing(text)  # 修正选项前的空格
    return text


def clean_markdown_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            content = file.read()
    except FileNotFoundError:
        print(f"Error: The file {input_file} does not exist.")
        return
    except IOError:
        print(f"Error: Could not read the file {input_file}.")
        return

    cleaned_content = clean_markdown(content)

    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(cleaned_content)
    except IOError:
        print(f"Error: Could not write to the file {output_file}.")
        return

    print(f'清理后的Markdown文件已保存到 {output_file}')


if __name__ == "__main__":
    input_file = 'single_choice_questions.md'  # 替换为你的Markdown文件路径
    output_file = 'cleaned_markdown_file.md'  # 替换为输出文件的路径

    clean_markdown_file(input_file, output_file)
