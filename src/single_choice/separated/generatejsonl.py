import re
import json


def parse_markdown_to_json(markdown_text):
    questions = []
    current_id = None
    current_question = None

    lines = markdown_text.split('\n')

    for line in lines:
        line = line.strip()

        if line.startswith("ID:"):
            current_id = line.split(":")[1].strip()
            continue

        question_match = re.match(r'^(\d+)\.(.*?)$', line)
        if question_match:
            if current_question:
                questions.append(current_question)
            current_qid = question_match.group(1)
            question_text = question_match.group(2).strip()
            current_question = {
                "id": current_id,
                "qid": int(current_qid),
                "question": question_text,
                "options": {}
            }
            continue

        option_match = re.match(r'^([A-D])\.(.*)', line)
        if option_match:
            option_id = option_match.group(1)
            option_text = option_match.group(2).strip()
            if current_question and option_id not in current_question["options"]:
                current_question["options"][option_id] = option_text

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
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    markdown_file_path = "2018工程管理单选题.txt"  # Replace this with your file path
    jsonl_file_path = "2018工程管理单选题.jsonl"  # Replace this with your desired output file path
    markdown_text = read_markdown_file(markdown_file_path)
    questions = parse_markdown_to_json(markdown_text)
    save_to_jsonl(questions, jsonl_file_path)
    print(f"Data has been saved to {jsonl_file_path}")


if __name__ == "__main__":
    main()
