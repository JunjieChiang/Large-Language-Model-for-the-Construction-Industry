import faiss
from FlagEmbedding import FlagModel
from http import HTTPStatus
import json
import os
from openai import AzureOpenAI
import embedding
from config import get_args
import logging
from tqdm import tqdm

def setup_openai_client():

    client = AzureOpenAI(
        api_key=os.getenv('41df71f980554898b556b2ee3d3dc8d1'),
        azure_endpoint = os.getenv("https://openai-api-siat.openai.azure.com/"),
        api_version="2024-02-01"
    )

    return client

def get_completion(client, embedding_model, prompt, top_k):

    response = client.chat.completions.create(
        model=embedding_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return response.choices[0].message.content

def generate_questions(client, generative_model, nowsentence, k):
    """问题生成"""
    prompt_template = f'''[INST]
    请你根据我给你的句子，提出{k}个有意义的问题。
    例如：在进行环氧胶泥的施工时需要注意什么？
    我提供的句子如下：
    {nowsentence}
    你的回答只需要输出三个问题，不需要任何解释性的语句，Be Creative!
    [/INST]'''
    questions = get_completion(client, generative_model, prompt_template, k)

    return questions


def generate_answers(client, generative_model, questions, k):
    """根据问题，生成答案"""
    questions_text = '\n'.join(questions)
    prompt_template = f'''
      你的任务是将我的{k}问题，生成对应的{k}个json对象。
      json对象包含以下键：
      1."query": 中文的问题，由我给你提供,由我给你提供的是用数字编号区分好的句子，你的回答应该忽略数字编号
      2."pos": 多个字符串，根据我给你提供的句子的信息，你自己再组织成完整的中文句子（包含主谓宾）来回答问题。
      3."neg": 多个字符串，多个跟pos意思完全相反或毫不相关的完整的句子（包含主谓宾）
      我提供的问题如下:
      '{questions_text}'
      注意一下几点:
      - neg里面应该包含多个反例语句
      - pos里面也应该有多个正例句子
      - query中不应该包含数字编号
      - Both the query and answer should be in Chinese.
      特别注意，输出的json代码，一个对象弄到一行，也就是说你应该输出三行
      你的回答中请不要有任何解释的词或者句子，我只想要json对象并且以txt形式输出
      输出格式如下：
      {{"query": "", "pos":["", ""], "neg":["", ""]}}
        '''
    answers = get_completion(client, generative_model, prompt_template, k)

    return answers


def data_generate(save_path, client, generative_model, corpus, k):
    """制作数据集：根据每个段落生成potential问题及相应答案"""

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(save_path, 'finetune.jsonl'), 'w', encoding='utf-8') as file:
        for sentence in tqdm(corpus[:1], desc="Data Generated"):  # 处理corpus中前10个段落
            questions = generate_questions(client, generative_model, sentence, k)
            answers = generate_answers(client, generative_model, questions, k)
            print(answers)

            for answer in answers.split('\n'):
                if answer.strip():
                    json_line = json.dumps(json.loads(answer), ensure_ascii=False) + '\n'
                    file.write(json_line)

    print("Data generation and saving completed.")


if __name__ == "__main__":
    args = get_args()
    client = setup_openai_client()
    # model, index = embedding.load_embedding_model(args.embedding_model, args.dimension, args.data_path)
    corpus = embedding.load_sentences(args.data_path)
    data_generate(args.data_result, client, args.generate_model, corpus, args.k)
