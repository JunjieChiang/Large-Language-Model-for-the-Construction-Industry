import json
import os
from openai import AzureOpenAI
os.environ['AZURE_OPENAI_API_KEY'] = '41df71f980554898b556b2ee3d3dc8d1'  # 替换为你的实际API密钥
os.environ['AZURE_OPENAI_ENDPOINT'] ='https://openai-api-siat.openai.azure.com/'
client = AzureOpenAI(
  azure_endpoint = os.getenv("https://openai-api-siat.openai.azure.com/"),
  api_key=os.getenv("41df71f980554898b556b2ee3d3dc8d1"),
  api_version="2024-02-01"
)

def read_jsonl_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]
    return data

def get_completion1(prompt):
  response = client.chat.completions.create(
  model="gpt-35-turbo", # model = "deployment_name".
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
  )
  return response.choices[0].message.content



jsonl_file_path = 'merged_output.jsonl'
dataset = read_jsonl_file(jsonl_file_path)


lk="{"
rk="}"
cnt=0
#比如问题是建筑装饰工业化的基础是什么？pos里面应该是建筑装饰工业化的基础是模块化设计。而neg里面应该是建筑装饰工业化的基础是批量化生产等句子。
#3."neg": 多个中文句子，根据我给你选项除去答案选项，注意你不需要关注回答的中文句子是否符合逻辑，你就把所有除去答案的选项也当做正确选项就行，并且组织成完整的中文句子（包含主谓宾）来回答
for nowjson in dataset[0:10]:
  if 'answer' in nowjson and 'A' in nowjson['options']:
      prompt_template = f'''
        我已经将一道单项选择题的信息转化为json对象了，
        你的任务是根据我提供的题目信息，生成用于微调的json对象。
        该json对象应包含以下键：

        1. "query": 根据我提供的问题信息，将其转化为符合逻辑的完整问句，注意不要包含选择题的小括号。例如，若问题信息是“不属于砌体结构主要构造措施的是()”，那么应转化为“不属于砌体结构主要构造措施的是什么？”

        2. "pos": 一个中文句子，根据提供的答案选项和问题，组织成一个有逻辑的完整句子，回答该问题。然后根据答案解析，对回答进行解释，但注意不要出现类似“答案选择B”的解释。

        中文问题是: '{nowjson['question']}'
        提供的选项是: '{nowjson['options']}'
        这道题的答案选项是: {nowjson['answer']}
        这道题的答案解析是: {nowjson['solution']}

        特别注意：
        1. 输出的json代码应为一行，即每个对象应在一行内输出。

        2. 输出格式如下，请严格按照格式输出:
        {lk}"query": "","pos":[""]{rk}
      '''
    # prompt_template2

      response=get_completion1(prompt_template)
      full_text = ''.join(response)
      print(full_text+'\n')

