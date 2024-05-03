import faiss
from FlagEmbedding import FlagModel
from http import HTTPStatus
import json
import os
from openai import AzureOpenAI
import argparse

parser = argparse.ArgumentParser(description='模型参数')
parser.add_argument('--embedding_model', type=str, default='retriever/bge-m3',help='embedding model')
parser.add_argument('--generate_model', type=str, default='gpt-35-turbo',help='generate model')
parser.add_argument('--data_path', type=str,default='./corpus(1).txt',help='input data path')
parser.add_argument('--top_k',type=int, default=5,help='the number of returned information chunks')
args = parser.parse_args()


client = AzureOpenAI(
  azure_endpoint = os.getenv("https://openai-api-siat.openai.azure.com/"),
  api_key=os.getenv("41df71f980554898b556b2ee3d3dc8d1"),
  api_version="2024-02-01"
)

def get_completion1(prompt):
  response = client.chat.completions.create(
  model=args.generate_model, # model = "deployment_name".
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
  )
  return response.choices[0].message.content

def get_completion2(prompt):
  response = client.chat.completions.create(
  model=args.generate_model, # model = "deployment_name".
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
  )
  return response.choices[0].message.content


def read_sentences_from_txt(file_path):
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentences.append(line.strip())  # 移除每行句子的换行符并存储
    return sentences

txt_file_path = args.data_path
sentences = read_sentences_from_txt(txt_file_path)

model = FlagModel(args.embedding_model,
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
corpus_embeddings = model.encode(sentences)
corpus_embeddings = corpus_embeddings.astype('float32')
# 使用 Faiss 构建索引
d = corpus_embeddings.shape[1]  # 向量维度 (num_sentences, embedding_dimension)
index = faiss.IndexFlatL2(d)  # 使用 L2 （欧式）距离度量构建 Flat 索引
print(index.is_trained)
index.add(corpus_embeddings)  # 将向量添加到索引中
print(index.ntotal)


for nowsentence in sentences[:10]:
    prompt_template = f'''[INST]
    请你根据我给你的句子，提出三个个有意义的问题
    接下来我给你一个例子：在进行环氧胶泥的施工时需要注意什么？
    我给你的句子是：{nowsentence}
    你的回答只需要输出三个问题，不需要任何解释性的语句，Be Creative!
    [/INST]
    '''
    response_query = get_completion1(prompt_template)
    query = response_query
    query_embedding = model.encode([query])
    query_embedding = query_embedding.astype('float32')
    k = args.top_k
    _, similar_indices = index.search(query_embedding,k) # 返回前k大索引
    result = []
    for i, j in enumerate(similar_indices[0]):
      result.append(sentences[j])
    lk="{"
    rk="}"
    result_merge = '\n'.join(result)
    prompt_template2= f'''
      你的任务是将我的三个问题，生成三个json对象
      json对象包含以下键：
      1."query": 中文的问题，由我给你提供,由我给你提供的是用数字编号区分好的句子，你的回答应该忽略数字编号
      2."pos": 多个字符串，根据我给你提供的句子的信息，你自己再组织成完整的中文句子（包含主谓宾）来回答问题。
      3."neg": 多个字符串，多个跟pos意思完全相反或毫不相关的完整的句子（包含主谓宾）
      中文问题是:'{query}'
      提供的句子是'{result_merge}'
      注意一下几点:
      - neg里面应该包含多个反例语句
      - pos里面也应该有多个正例句子
      - query中不应该包含数字编号
      - Both the query and answer should be in Chinese.
      特别注意，输出的json代码，一个对象弄到一行，也就是说你应该输出三行
      你的回答中请不要有任何解释的词或者句子，我只想要json对象并且以txt形式输出
      输出格式如下,为三行：
      {lk}"query": "","pos":[""],"neg":["","",""]{rk}
      {lk}"query": "","pos":[""],"neg":["","",""]{rk}
      {lk}"query": "","pos":[""],"neg":["","",""]{rk}
        '''
    response1 = get_completion2(prompt_template2)
    full_text = ''.join(response1)
    print(full_text)

