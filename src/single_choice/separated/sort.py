import re

def classify_questions(markdown_content):
    single_choice_questions = []
    multiple_choice_questions = []
    subjective_questions = []

    lines = markdown_content.split('\n')
    current_section = None
    id_counter = 0

    for line in lines:
        if line.startswith('一、单项选择题'):
            current_section = 'single_choice'
            id_counter += 1
            single_choice_questions.append(f'ID: {id_counter}')
            continue
        elif line.startswith('二、多项选择题'):
            current_section = 'multiple_choice'
            multiple_choice_questions.append(f'ID: {id_counter}')
            continue
        elif line.startswith('三、案例分析题'):
            current_section = 'subjective'
            subjective_questions.append(f'ID: {id_counter}')
            continue

        if current_section == 'single_choice':
            single_choice_questions.append(line)
        elif current_section == 'multiple_choice':
            multiple_choice_questions.append(line)
        elif current_section == 'subjective':
            subjective_questions.append(line)

    return single_choice_questions, multiple_choice_questions, subjective_questions

def save_to_file(questions, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for question in questions:
            f.write(question + '\n')

def main():
    with open('output_2018工程管理.txt', 'r', encoding='utf-8') as f:
        markdown_content = f.read()

    single_choice_questions, multiple_choice_questions, subjective_questions = classify_questions(markdown_content)

    save_to_file(single_choice_questions, '2018工程管理单选题.txt')
    save_to_file(multiple_choice_questions, '2018工程管理多选题.txt')
    save_to_file(subjective_questions, '2018工程管理主观题.txt')

if __name__ == "__main__":
    main()
