import argparse
import os
from word2vec import pretrain_word2vec, W2VEmbReader
from trainer import ABAETrainer
from data_loader import DataProcessor
from utils import init_logger, set_seed


def main(args):
    init_logger()
    set_seed(args)
    if args.do_train_word2vec:
        pretrain_word2vec(args)
    w2v = W2VEmbReader(os.path.join(args.data_path, args.domain + '/' + args.word2vec_file), args.embedding_size)
    data_processor = DataProcessor(args)
    vocab, train_dataset, test_dataset = data_processor.get_data()
    trainer = ABAETrainer(args, w2v, vocab, train_dataset, test_dataset)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default='../preprocessed_data', type=str, help="Data folder path")
    parser.add_argument("--domain", default="restaurant", type=str, help="Default domain folder")
    parser.add_argument("--seed", default=1, type=int, help="random seed for initialization")
    parser.add_argument("--embedding_size", default=200, type=int, help="Word embedding size")
    parser.add_argument("--window_size", default=5, type=int, help="Word2Vec window size config")
    parser.add_argument("--min_count", default=5, type=int, help="Word2Vec minimum word frequency")
    parser.add_argument("--num_workers", default=7, type=int, help="Word2Vec training worker")
    parser.add_argument("--vocab_size", default=9000, type=int, help="Maximum vocab size")
    parser.add_argument("--aspect_count", default=20, type=int, help="Number of aspects")
    parser.add_argument("--ortho_reg", default=0.1, type=float, help="Ortho regularization weight term")
    parser.add_argument("--maxlen", default=50, type=int, help="Max sequence length")
    parser.add_argument("--batch_size", default=64, type=int, help="Batch size")
    parser.add_argument("--device", default='cpu', type=str, help="GPU or CPU to train model")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate for Adam")
    parser.add_argument("--negative_samples", default=20, type=int, help="Number of negative sample for each sentence")
    parser.add_argument("--num_epochs", default=1, type=int, help="Number of running epoch")
    parser.add_argument("--do_train", action="store_true", help="Whether or not training model")
    parser.add_argument("--do_eval", action="store_true", help="Whether or not do eval")
    parser.add_argument("--word2vec_file", default='word2vec')
    parser.add_argument("--do_train_word2vec", action="store_true", help="Whether or not do train word2vec")
    parser.add_argument("--train_file", default="train.txt")
    parser.add_argument("--dev_file", default="dev.txt")
    parser.add_argument("--test_file", default="test.txt")
    parser.add_argument("--padding_token", default=0)
    parser.add_argument("--output_dir", default="../experiment", help="Save model and file log")

    args = parser.parse_args()
    main(args)
