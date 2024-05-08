import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Parameter Configuration')
    parser.add_argument('--embedding_model', type=str, default='retriever/bge-m3', help='embedding model')
    parser.add_argument('--generate_model', type=str, default='gpt-35-turbo', help='generate model')
    parser.add_argument('--data_path', type=str, default='example/corpus.txt', help='input data path')
    parser.add_argument('--top_k', type=int, default=5, help='the number of returned information chunks')
    parser.add_argument('--k', type=int, default=3, help='the number of data generated by one api call')
    parser.add_argument('--dimension', type=int, default=1024, help='please refer to the embedding model')
    parser.add_argument('--data_result', type=str, default='result/finetune', help='the save path of generated data path')

    return parser.parse_args()
