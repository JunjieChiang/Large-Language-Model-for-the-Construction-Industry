import re
import json


def parse_markdown_to_json(markdown_text):
    questions = []
    current_id = None
    current_question = None
    is_parsing_solution = False

    lines = markdown_text.split('\n')

    for line in lines:
        line = line.strip()

        if line.startswith("ID:"):
            current_id = line.split(":")[1].strip()
            continue

        question_match = re.match(r'^(\d+)\.([A-D])$', line)
        if question_match:
            if current_question:
                questions.append(current_question)
            current_qid = question_match.group(1)
            answer = question_match.group(2)
            current_question = {
                "id": current_id,
                "qid": int(current_qid),
                "answer": answer,
                "solution": ""
            }
            is_parsing_solution = False
            continue

        if line.startswith("【解析】"):
            if current_question:
                current_question["solution"] = line[4:].strip()
                is_parsing_solution = True
            continue

        if is_parsing_solution:
            if current_question:
                current_question["solution"] += " " + line

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
    markdown_file_path = "cleaned_markdown_file2.md"  # Replace this with your file path
    jsonl_file_path = "output2.jsonl"  # Replace this with your desired output file path
    markdown_text = read_markdown_file(markdown_file_path)
    questions = parse_markdown_to_json(markdown_text)
    save_to_jsonl(questions, jsonl_file_path)
    print(f"Data has been saved to {jsonl_file_path}")


if __name__ == "__main__":
    main()
