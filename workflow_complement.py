import numpy as np
import faiss
from embedding import load_embedding_model, load_sentences, create_index_knowledge_base, load_embeddings, setup_logging
import config
from tqdm import tqdm

# 设置日志
setup_logging()

# 加载配置参数
args = config.get_args()

# 文件路径
summarized_ms_path = 'example/method_statement/Method Statement for Feasibility Study on Temporary Widening at SSK Drive Summary.txt'
original_ms_path = 'example/method_statement/Method Statement for Feasibility Study on Temporary Widening at SSK Drive.txt'

# 读取TXT文件内容
def read_txt_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    return content

# 创建知识库索引
def create_knowledge_base_index(model_path, dimension, knowledge_source_path, index_save_path, embeddings_save_path):
    model = load_embedding_model(model_path)
    knowledge_sources = load_sentences(knowledge_source_path)

    embeddings = []
    for knowledge in tqdm(knowledge_sources, desc="Creating embeddings from knowledge source"):
        embedding = model.encode(knowledge)
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype('float32')
    save_embeddings(embeddings, embeddings_save_path)

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, index_save_path)

    logging.info(f"Model {model_path} loaded and FAISS index created with {index.ntotal} vectors.")

# 加载知识库索引
def load_knowledge_base_index(index_path):
    index = faiss.read_index(index_path)
    return index

# 检索并完善文档
def complement_workflow(summarized_text, knowledge_base_index, model, knowledge_sources):
    summarized_lines = summarized_text.split('\n')
    enhanced_text = []

    for line in summarized_lines:
        if line.strip():
            # 生成当前行的嵌入
            embedding = model.encode(line)

            # 在知识库中检索相关信息
            D, I = knowledge_base_index.search(np.array([embedding]), k=1)  # 检索最相似的1个结果
            retrieved_text = knowledge_sources[I[0][0]]

            enhanced_text.append(line + " " + retrieved_text)
        else:
            enhanced_text.append(line)

    return '\n'.join(enhanced_text)

# 主流程
def main():
    # 读取文件内容
    summarized_text = read_txt_file(summarized_ms_path)
    original_text = read_txt_file(original_ms_path)

    # 创建知识库索引
    create_knowledge_base_index(
        model_path=args.embedding_model,
        dimension=args.dimension,
        knowledge_source_path=original_ms_path,
        index_save_path='example/KB/workflow_complement/workflow_index.index',
        embeddings_save_path='example/KB/workflow_complement/workflow_embeddings.npy'
    )

    # 加载知识库索引
    knowledge_base_index = load_knowledge_base_index(args.knowledge_index)
    knowledge_sources = load_sentences(original_ms_path)

    # 加载嵌入模型
    model = load_embedding_model(args.embedding_model)

    # 完善总结文档
    enhanced_text = complement_workflow(summarized_text, knowledge_base_index, model, knowledge_sources)
    print(f"Enhanced Method Statement:\n{enhanced_text}")

if __name__ == "__main__":
    main()
