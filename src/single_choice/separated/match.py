import json


def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data


def merge_questions_answers(questions, answers):
    answer_dict = {(answer['id'], answer['qid']): answer for answer in answers}

    merged_data = []
    for question in questions:
        key = (question['id'], question['qid'])
        if key in answer_dict:
            question['answer'] = answer_dict[key]['answer']
            question['solution'] = answer_dict[key]['solution']
        merged_data.append(question)

    return merged_data


def save_to_jsonl(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + '\n')


def main():
    questions_file_path = "output.jsonl"  # Replace with your questions file path
    answers_file_path = "output2.jsonl"  # Replace with your answers file path
    output_file_path = "merged_output.jsonl"  # Replace with your desired output file path

    questions = read_jsonl_file(questions_file_path)
    answers = read_jsonl_file(answers_file_path)

    merged_data = merge_questions_answers(questions, answers)
    save_to_jsonl(merged_data, output_file_path)

    print(f"Merged data has been saved to {output_file_path}")


if __name__ == "__main__":
    main()
