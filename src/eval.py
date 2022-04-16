import argparse
from sklearn.metrics import classification_report
import torch
from model import ABAE
from word2vec import W2VEmbReader
from data_loader import DataProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import collections

LABELS_MAP = {
    0: "Misc", 1: "Price", 2: "Staff", 3: "Anecdotes", 4: "Staff",
    5: "Food", 6: "Ambience", 7: "Misc", 8: "Misc", 9: "Food",
    10: "Food", 11: "Misc", 12: "Misc", 13: "Misc"
}


def evaluation_with_F1(model, test_loader, labels, labels_map):
    predict_labels = []
    for batch, mask in tqdm(test_loader):
        predict_labels.extend(torch.argmax(model.predict_sentences(batch, mask), dim=-1).tolist())
    predict_labels_map = [labels_map[predict_labels[i]] for i in range(len(predict_labels))]
    print('\n' + classification_report(labels, predict_labels_map))
    return predict_labels_map, predict_labels


def load_model(args):
    saved_model = torch.load(args.model_path, map_location='cpu')
    model_params, model_args, vocab = saved_model['model'], saved_model['args'], saved_model['vocab']
    w2v = W2VEmbReader(os.path.join(model_args.data_path + model_args.domain, 'word2vec'),
                       emb_dim=model_args.embedding_size)
    model = ABAE(wv_dim=model_args.embedding_size, asp_count=model_args.aspect_count, ortho_reg=model_args.ortho_reg,
                 maxlen=model_args.maxlen,
                 init_aspects_matrix=w2v.get_aspect_matrix(model_args.aspect_count))
    model.load_state_dict(model_params)
    return model, w2v, model_args, vocab


def do_eval(args):
    model, w2v, model_args, vocab = load_model(args)
    data_processor = DataProcessor(model_args)
    test_dataset = data_processor.read_dataset(vocab, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    # Load test labels
    test_labels = []
    with open(os.path.join(model_args.data_path, model_args.domain + "/test_label.txt"), 'r') as f:
        for i, line in enumerate(f):
            test_labels.append(line.strip())
    with open(args.output_file, 'w') as f:
        predict_map, predict_labels = evaluation_with_F1(model, test_loader, test_labels, LABELS_MAP)
        f.write(classification_report(test_labels, predict_map))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Model path to load")

    args = parser.parse_args()
    do_eval(args)
