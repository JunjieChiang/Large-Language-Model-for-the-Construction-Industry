from FlagEmbedding import FlagModel
import json
import faiss
import logging
import numpy as np

def load_embeddings(load_path):
    return np.load(load_path)

def save_embeddings(embeddings, save_path):
    np.save(save_path, embeddings)


def setup_logging():

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_embedding_model(model_path, dimension, data_path ):

    logging.info("Loading model and encoding data...")
    model = FlagModel(model_path,
                      query_instruction_for_retrieval="为这个JSON数据生成表示以用于检索相关属性：",
                      use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
    sentences = load_sentences(data_path)
    #embeddings = model.encode(sentences).astype('float32')
    #save_embeddings(embeddings, '/content/RE-Generate/example/corpus_embeddings.npy')
    embeddings = load_embeddings('example/corpus_embeddings.npy')
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    logging.info(f"Model {model_path} loaded and FAISS index created with {index.ntotal} vectors.")

    return model, index, sentences

def load_sentences(file_path):

    sentences = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            sentences.append(line.strip())

    return sentences

def load_exam_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = [json.loads(line) for line in file]

    return data