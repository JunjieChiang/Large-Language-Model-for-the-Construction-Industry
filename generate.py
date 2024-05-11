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
        azure_endpoint=os.getenv("https://openai-api-siat.openai.azure.com/"),
        api_version="2024-02-01"
    )

    return client


def get_completion(client, embedding_model, prompt):
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
    prompt_template = f"""[INST] 
    请根据提供的句子生成{k}个相关的问题。
    如果提供的句子是：“在进行环氧胶泥的施工时，需要根据设计要求和标准要求选择合适的粘接力控制方案，并对其粘接力进行适当的调节和控制”，
    根据所提供的句子，你可能会问如下问题： 
    1. 在进行环氧胶泥的施工时需要注意什么？
    2. 如何选择合适的粘接力控制方案？ 
    3. 如何进行粘接力的调节和控制？
    现在，我提供的句子如下：
    {nowsentence} 
    根据上述句子，请试着提出{k}个有意义的问题。
    注意：
    1. 每次你只能提出一个问题，不允许用逗号问几个问题
    2. 回答不允许出现句子中没有的意思
    3. 回答使用中文
    [/INST]"""
    questions = get_completion(client, generative_model, prompt_template)

    return questions


def generate_answers(client, generative_model, questions, relevant_source, k):
    """根据问题，生成答案"""
    # questions_text = '\n'.join(questions)
    prompt_template = f'''
      根据下面的{k}个问题，生成对应的{k}个回答，并根据这些问题和回答构成{k}个JSON对象，每个JSON对象包含以下3个键：
      - "query": 中文的问题，由我给你提供，其中不需要任何数字编号。
      - "pos": 包含几个中文句子，是对于当前问题的回答，每个句子应当尽量完整。
      - "neg": 包含几个中文句子，是跟pos意思完全相反或毫不相关的完整的句子

      以下是提供的问题： 
      '{questions}'
      以下是与问题相关的信息，你的回答可以参考这些内容： 
      '{relevant_source}'

      注意：
      - Both the query and answer should be in Chinese.
      每个JSON对象的输出格式及示例：
      {{"query": "", "pos":["", ""], "neg":["", ""]}}
      请以这种格式输出{k}个JSON对象，每个对象为一行。
        '''
    answers = get_completion(client, generative_model, prompt_template)

    return answers


def get_relevant_source(questions, corpus, embedding_model, top_k):
    questions_embedding = embedding_model.encode([questions]).astype('float32')

    _, similar_indices = index.search(questions_embedding, top_k)
    result = []
    for i, j in enumerate(similar_indices[0]):
        result.append(corpus[j])
    result_merge = '\n'.join(result)

    return result_merge


def data_generate(save_path, client, generative_model, corpus, k, embedding_model, top_k):
    """制作数据集：根据每个段落生成potential问题及相应答案"""

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for sentence in tqdm(corpus, desc="Data Generated"):  # 处理corpus中前10个段落
        questions = generate_questions(client, generative_model, sentence, k)
        relevant_source = get_relevant_source(questions, corpus, embedding_model, top_k)
        answers = generate_answers(client, generative_model, questions, relevant_source, k)
        with open(os.path.join(save_path, 'finetune.txt'), 'a', encoding='utf-8') as file:
            full_text = ''.join(answers)
            file.write(full_text + '\n')


if __name__ == "__main__":
    args = get_args()
    client = setup_openai_client()

    # 在embedding.py里加载模型，把embedding,index保存在本地，然后在这里写一个函数读取就好
    embedding_model, index, corpus = embedding.load_embedding_model(args.embedding_model, args.dimension,
                                                                    args.data_path)

    data_generate(args.data_result, client, args.generate_model, corpus, args.k, embedding_model, args.top_k)
