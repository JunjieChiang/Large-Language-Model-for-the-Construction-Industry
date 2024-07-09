import os
import sys
import json
from tqdm import tqdm
from sentence_transformers import util

# 确定项目的根目录并添加到sys.path中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# 导入嵌入模型和配置
import embedding
import config

# 获取配置参数
args = config.get_args()

# 加载嵌入模型
model = embedding.load_embedding_model(args.embedding_model)

# 定义过滤关键词
keywords = ["下列...正确的是", "因此，答案为", "选项"]
keyword_embeddings = model.encode(keywords)

# 定义相似度阈值
similarity_threshold = 0.5

# 使用函数过滤数据并保存结果

input_file_path = os.path.join(args.data_result, 'generated_from_choice_question.jsonl')  # 输入文件路径
output_file_path = os.path.join(args.data_result, 'cleaned_generated_from_choice_question.jsonl')  # 输出文件路径
filtered_out_file = os.path.join(args.data_result, 'filtered_out_generated_from_choice_question.jsonl')

filtered_data = []
filtered_out_data = []

# 打开输入文件进行处理
with open(input_file_path, 'r', encoding='utf-8') as f:
    for line in tqdm(f, desc="Searching"):
        data = json.loads(line)
        response_text = data['response']

        # 将response字段转换为嵌入向量
        response_embedding = model.encode(response_text)

        # 计算与每个关键词的相似度
        is_filtered_out = False
        for keyword_embedding in keyword_embeddings:
            cosine_score = util.pytorch_cos_sim(response_embedding, keyword_embedding)
            if cosine_score.item() > similarity_threshold:
                filtered_out_data.append(data)
                is_filtered_out = True
                break

        if not is_filtered_out:
            filtered_data.append(data)

# 保存过滤后的数据
with open(output_file_path, 'w', encoding='utf-8') as f:
    for item in filtered_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# 保存被过滤的数据
with open(filtered_out_file, 'w', encoding='utf-8') as f:
    for item in filtered_out_data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

print(f"Filtered data saved to {output_file_path}")
print(f"Filtered out data saved to {filtered_out_file}")
