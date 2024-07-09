import os
import embedding
import faiss
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
    prompt_template = f'''[INST] 
      根据下面的{k}个问题，生成对应的{k}个回答，并根据这些问题和回答构成{k}个JSON对象，每个JSON对象包含以下2个键：
      - "user": 模拟用户提出的中文问题，由我给你提供，其中不需要任何数字编号。
      - "response": 该字段应包含一个列表，该列表是一个完整的句子，结合下面提供的问题及其相关信息回答"user"中的问题。

      以下是提供的问题： 
      '{questions}'
      以下是与问题相关的信息，你的回答可以参考这些内容： 
      '{relevant_source}'

      注意：
      - Both the query and answer should be in Chinese.
      - 生成的{k}个JSON数据应为单行格式，以便于处理和分析
      
      示例输出：
      {{"user": "常见的高分子防水卷材有哪些？", "response": ["常见的高分子防水卷材包括三元乙丙、聚氯乙烯、氯化聚乙烯、氯化聚乙烯-橡胶共混及三元丁橡胶防水卷材。这些材料具有良好的防水性能，可广泛应用于建筑、隧道、地下工程等领域。"]}}
      [/INST]'''

    answers = llm.get_completion(prompt_template)

    return answers


def process_choice_question(llm, choice_question):
    """从试卷文本中生成Q&A数据"""
    prompt_template = f'''
    任务说明：
    根据提供的选择题信息，你需要生成一个用于微调的JSON对象。这个JSON对象将包含两个键："user"和"response"，分别代表经过重新表述的问题和解释。
    
    以下是给定的题目信息： 
      '{choice_question}'
    
    详细要求如下：
    - "user": 将提供的问题转换成一个清晰、完整的问句。需要去除原问题中的小括号，并确保问题是直接可回答的形式。例如，如果原问题是“建筑装饰工业化的基础是()”，应转化为“建筑装饰工业化的基础是什么？”
    - "response": 该字段应包含一个列表，该列表是一个完整的句子，结合答案选项直接回答"user"中的问题，并使用"solution"字段提供的内容来详细解释答案的原因和逻辑。同时解释中不要引用选项标识(如"A", "B", "C", "D")
    
    示例输入：
    - 问题: "通过对钢化玻璃进行均质处理可以什么？"
    - 答案解析: "通过对钢化玻璃进行均质处理可以大大降低钢化玻璃的自爆率。"

    示例输出：
    {{"user": "通过对钢化玻璃进行均质处理可以达到什么效果？", "response": ["通过对钢化玻璃进行均质处理，可以大大降低其自爆率。均质处理通过均匀加热钢化玻璃，使其内部应力均衡，从而减少自爆的可能性。"]}}
    
    注意：
    - 生成的JSON数据应为单行格式，以便于处理和分析
    - 避免提及“因此，答案...”的陈述以及答案选项（A、B、C、D）
    '''

    answers = llm.get_completion(prompt_template)

    return answers


def process_subjective_question(llm, subjective_question):
    """从试卷文本简答题中生成Q&A数据"""
    prompt_template = f'''
    任务说明：
    根据提供的考点信息，你需要生成一个用于微调的JSON对象。这个JSON对象将包含两个键："user"和"response"，分别代表经过重新表述的问题和解释。

    以下是给定的题目信息： 
      '{subjective_question}'

    详细要求如下：
    - "user": 该字段模拟用户的提问，将提供的问题转换成一个清晰、完整的问句，确保问题是直接可回答的形式。
    - "response": 该字段应包含一个列表，该列表是一个完整的句子。你应该使用"pos"字段提供的内容来详细解释和回答"user"中的问题，需要注意原因和逻辑。

    示例输入：
    {{"query": 通过对钢化玻璃进行均质处理可以达到什么效果？", "pos": ["通过对钢化玻璃进行均质处理可以大大降低钢化玻璃的自爆率。"]}}

    示例输出：
    {{"user": "通过对钢化玻璃进行均质处理可以达到什么效果？", "response": ["通过对钢化玻璃进行均质处理，可以大大降低其自爆率。均质处理通过均匀加热钢化玻璃，使其内部应力均衡，从而减少自爆的可能性。"]}}

    注意：
    - 生成的JSON数据应为单行格式，以便于处理和分析
    '''

    answers = llm.get_completion(prompt_template)

    return answers


def get_relevant_source(questions, corpus, embedding_model, top_k):
    questions_embedding = embedding_model.encode([questions]).astype('float32')
    index = faiss.read_index(args.knowledge_index)
    _, similar_indices = index.search(questions_embedding, top_k)
    result = []
    for i, j in enumerate(similar_indices[0]):
        if 0 <= j < len(corpus):
            result.append(corpus[j])

    result_merge = '\n'.join(result)

    return result_merge


def generate_from_corpus(llm, corpus, embedding_model):
    # start to generate data
    for sentence in tqdm(corpus[:100], desc="Data Generated"):
        questions = generate_questions(llm, sentence, args.k)
        relevant_source = get_relevant_source(questions, corpus, embedding_model, args.top_k)
        generation = generate_answers(llm, questions, relevant_source, args.k)
        with open(os.path.join(args.data_result, 'generated_from_corpus_2.jsonl'), 'a', encoding='utf-8') as file:
            full_text = ''.join(generation)
            file.write(full_text + '\n')


def generate_from_choice_question(llm, examination_data):
    # start to generate data
    for now_choice_question in tqdm(examination_data[135:], desc="Data Generated"):
        if 'answer' in now_choice_question and 'A' in now_choice_question['options']:
            generation = process_choice_question(llm, now_choice_question)

        with open(os.path.join(args.data_result, 'generated_from_choice_question.jsonl'), 'a', encoding='utf-8') as file:
            full_text = ''.join(generation)
            file.write(full_text + '\n')


def generate_from_subjective_question(llm, subjective_question):
    # start to generate data
    for now_subjective_data in tqdm(subjective_question[437:], desc="Data Generated"):
        generation = process_subjective_question(llm, now_subjective_data)

        with open(os.path.join(args.data_result, 'generated_from_subjective_question.txt'), 'a', encoding='utf-8') as file:
            full_text = ''.join(generation)
            file.write(full_text + '\n')



if __name__ == "__main__":
    args = get_args()

    # load LLM configuration
    if args.model_configs == None:
        args.model_configs = f"model_configs/{args.generative_model}.json"

    if not os.path.exists(args.data_result):
        os.makedirs(args.data_result)

    llm = init_model_config(args.model_configs)

    # generate data from corpus
    if args.from_corpus == "True":
        corpus = embedding.load_sentences(args.data_path)
        embedding_model = embedding.load_embedding_model(args.embedding_model)
        generate_from_corpus(llm=llm, corpus=corpus, embedding_model=embedding_model)

    # generate data from test paper
    if args.from_choice_question == "True":
        examination = embedding.load_exam_data(args.exam_data)
        generate_from_choice_question(llm=llm, examination_data=examination)

    if args.from_subjective_question == "True":
        subjective_question = embedding.load_exam_data(args.subjective_question)
        generate_from_subjective_question(llm=llm, subjective_question=subjective_question)
