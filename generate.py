import os
import embedding
from config import get_args
from tqdm import tqdm
from models import *


def generate_questions(llm, nowsentence, k):
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

    questions = llm.get_completion(prompt_template)

    return questions


def generate_answers(llm, questions, relevant_source, k):
    """根据问题，生成答案"""
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

    answers = llm.get_completion(prompt_template)

    return answers


def get_relevant_source(questions, index, corpus, embedding_model, top_k):
    questions_embedding = embedding_model.encode([questions]).astype('float32')

    _, similar_indices = index.search(questions_embedding, top_k)
    result = []
    for i, j in enumerate(similar_indices[0]):
        result.append(corpus[j])
    result_merge = '\n'.join(result)

    return result_merge


def main():

    args = get_args()

    if args.model_configs == None:
        args.model_configs = f"model_configs/{args.generative_model}.json"

    if not os.path.exists(args.data_result):
        os.makedirs(args.data_result)

    # load resources
    embedding_model, index, corpus = embedding.load_embedding_model(args.embedding_model, args.dimension,
                                                                    args.data_path)
    # load LLM configuration
    llm = init_model_config(args.model_configs)

    # start to generate data
    for sentence in tqdm(corpus[5000:5002], desc="Data Generated"):
        questions = generate_questions(llm, sentence, args.k)
        relevant_source = get_relevant_source(questions, index, corpus, embedding_model, args.top_k)
        answers = generate_answers(llm, questions, relevant_source, args.k)
        with open(os.path.join(args.data_result, 'finetune.txt'), 'a', encoding='utf-8') as file:
            full_text = ''.join(answers)
            file.write(full_text + '\n')



if __name__ == "__main__":

    args = get_args()
    main()