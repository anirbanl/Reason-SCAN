import argparse
import os

from torch.optim.lr_scheduler import LambdaLR

from dataloader import dataloader
from model.config import cfg
from model.model import GSCAN_model
from model.utils import *
import random

def train(train_data_path: str, val_data_paths: dict, use_cuda: bool, resume_from_file: str, is_baseline: bool):
    device = torch.device(type='cuda') if use_cuda else torch.device(type='cpu')

    logger.info("Loading Training set...")
    train_iter, train_input_vocab, train_target_vocab = dataloader(train_data_path,
                                                                   batch_size=cfg.TRAIN.BATCH_SIZE,
                                                                   use_cuda=use_cuda)
    val_iters = {}
    for split_name, path in val_data_paths.items():
        val_iters[split_name], _, _ = dataloader(path, batch_size=cfg.VAL_BATCH_SIZE, use_cuda=use_cuda,
                                                 input_vocab=train_input_vocab, target_vocab=train_target_vocab)

    pad_idx, sos_idx, eos_idx = train_target_vocab.stoi['<pad>'], train_target_vocab.stoi['<sos>'], \
                                train_target_vocab.stoi['<eos>']

    train_input_vocab_size, train_target_vocab_size = len(train_input_vocab.itos), len(train_target_vocab.itos)

    logger.info("Loading Dev. set...")

    val_input_vocab_size, val_target_vocab_size = train_input_vocab_size, train_target_vocab_size
    logger.info("Done Loading Dev. set.")

    model = GSCAN_model(pad_idx, eos_idx, train_input_vocab_size, train_target_vocab_size, is_baseline=is_baseline)

    model = model.cuda() if use_cuda else model

    log_parameters(model)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(trainable_parameters, lr=cfg.TRAIN.SOLVER.LR,
                                 betas=(cfg.TRAIN.SOLVER.ADAM_BETA1, cfg.TRAIN.SOLVER.ADAM_BETA2))
    scheduler = LambdaLR(optimizer,
                         lr_lambda=lambda t: cfg.TRAIN.SOLVER.LR_DECAY ** (t / cfg.TRAIN.SOLVER.LR_DECAY_STEP))

    cfg.RESUME_FROM_FILE = resume_from_file
    assert os.path.isfile(cfg.RESUME_FROM_FILE), "No checkpoint found at {}".format(cfg.RESUME_FROM_FILE)
    logger.info("Loading checkpoint from file at '{}'".format(cfg.RESUME_FROM_FILE))
    optimizer_state_dict = model.load_model(cfg.RESUME_FROM_FILE)
    optimizer.load_state_dict(optimizer_state_dict)
    start_iteration = model.trained_iterations
    print("start iteration is .. ", start_iteration)
    logger.info("Loaded checkpoint '{}' (iter {})".format(cfg.RESUME_FROM_FILE, start_iteration))

    logger.info("Prediction starts..")
    model.eval()
    print(val_iters)
    for split_name, val_iter in val_iters.items():
        output_file_name = f"{split_name}_predict.json"
        output_file_path = os.path.join(os.path.dirname(resume_from_file), output_file_name)
        logger.info(f"Output file path: {output_file_path}")
        output_file = predict_and_save(
            val_iter, model=model,
            output_file_path = output_file_path,
            max_decoding_steps=120, pad_idx=pad_idx,
            sos_idx=sos_idx,
            eos_idx=eos_idx,
            input_vocab=train_input_vocab,
            target_vocab=train_target_vocab)
        logger.info("Saved predictions to {}".format(output_file))


def main(flags, use_cuda):
    train_data_path = os.path.join(flags.data_dir, "train.json")

    test_splits = [
        flags.test_split
    ]
    val_data_paths = {split_name: os.path.join(flags.data_dir, split_name + '.json') for split_name in test_splits}

    if cfg.MODE == "train":
        train(train_data_path=train_data_path, val_data_paths=val_data_paths, use_cuda=use_cuda,
              resume_from_file=flags.load, is_baseline=flags.is_baseline)

    elif cfg.MODE == "predict":
        raise NotImplementedError()

    else:
        raise ValueError("Wrong value for parameters --mode ({}).".format(cfg.MODE))


if __name__ == "__main__":
    # torch.manual_seed(cfg.SEED)
    FORMAT = "%(asctime)-15s %(message)s"
    logging.basicConfig(format=FORMAT, level=logging.DEBUG,
                        datefmt="%Y-%m-%d %H:%M")
    logger = logging.getLogger(__name__)
    use_cuda = True if torch.cuda.is_available() else False
    logger.info("Initialize logger")

    if use_cuda:
        logger.info("Using CUDA.")
        logger.info("Cuda version: {}".format(torch.version.cuda))

    parser = argparse.ArgumentParser(description="LGCN models for GSCAN")
    parser.add_argument('--load', type=str, help='Path to model')
    parser.add_argument('--baseline', dest='is_baseline', action='store_true')
    parser.add_argument('--data_dir', type=str, help='Path to dataset')
    parser.add_argument('--seed', type=int, help='random seeds')
    parser.add_argument('--test_split', type=str, help='split to evaluate')
    parser.set_defaults(is_baseline=False)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    main(args, use_cuda)
