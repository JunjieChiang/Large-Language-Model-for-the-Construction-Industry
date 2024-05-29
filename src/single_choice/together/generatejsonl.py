import re
import json
import os


def parse_markdown_to_json(markdown_text):
    questions = []
    current_id = None
    current_question = None
    question_counter = None
    capture_solution = False

    lines = markdown_text.split('\n')

    for line in lines:
        line = line.strip()

        if line.startswith("ID:"):
            current_id = line.split(":")[1].strip()
            question_counter = None  # Reset question counter for each new ID
            continue

        if question_counter is None:
            question_match = re.match(r'^(\d+)[.．](.*?)$', line)
            if question_match:
                question_counter = int(question_match.group(1))

        question_match = re.match(rf'^{question_counter}[.．](.*?)$', line)
        if question_match:
            if current_question:
                questions.append(current_question)
            question_text = question_match.group(1).strip()
            current_question = {
                "id": current_id,
                "qid": question_counter,
                "question": question_text,
                "options": {},
                "answer": "",
                "solution": ""
            }
            question_counter += 1
            capture_solution = False  # Reset solution capture flag
            continue

        option_match = re.match(r'^([A-D])\.(.*)', line)
        if option_match:
            option_id = option_match.group(1)
            option_text = option_match.group(2).strip()
            if current_question and option_id not in current_question["options"]:
                current_question["options"][option_id] = option_text
            continue

        answer_match = re.match(r'(^答案：(.*)|【答案】)', line)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            if current_question:
                current_question["answer"] = answer_text
            continue

        solution_match = re.match(r'(^解析：(.*)|【解析】)', line)
        if solution_match:
            solution_text = solution_match.group(1).strip()
            if current_question:
                current_question["solution"] = solution_text
                capture_solution = True
            continue

        if capture_solution:
            # Append to the solution if capture_solution is True
            current_question["solution"] += f" {line}"

    if current_question:
        questions.append(current_question)

    return questions


def read_markdown_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()
    return markdown_text


def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False)+'\n')

def main(file_path, file_name):
    markdown_file_path = file_path  # Replace this with your file path
    now_name, now_extension = os.path.splitext(os.path.basename(file_path))
    jsonl_file_path =  f'paper/{file_name}/{now_name}.jsonl' # Replace this with your desired output file path
    markdown_text = read_markdown_file(markdown_file_path)
    questions = parse_markdown_to_json(markdown_text)
    save_to_jsonl(questions, jsonl_file_path)
    print(f"Data has been saved to {jsonl_file_path}")