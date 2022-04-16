import re
import logging
import torch
import numpy as np
import random
from model import ABAE

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')


def init_logger():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')


def is_number(token):
    return bool(num_regex.match(token))


def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
