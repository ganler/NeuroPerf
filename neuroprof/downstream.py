from neuroprof.dataset import X86PerfDownstreamDataset
from neuroprof.data_gen import PerfData

import os
from datetime import datetime

from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from rich.progress import Progress

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import nn
import torch


def new_model():
    # Set a configuration for our RoBERTa model
    config = RobertaConfig(
        vocab_size=2048,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    # Initialize the model from a configuration without pretrained weights
    return RobertaForMaskedLM(config=config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Downstream task training')
    parser.add_argument(
        '--model', type=str, default='./pretrained_model/checkpoint-7552', help='checkpoint dir')
    parser.add_argument('--epoch', type=int, default=10, help='epoch')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--save_dir', type=str, default='downstream_model',
                        help='path to save the downstream model')
    parser.add_argument('--batch_size', type=int,
                        default=16, help='batch size')
    parser.add_argument('--log_freq', type=int,
                        default=100, help='sample size')
    args = parser.parse_args()

    if args.model == 'None':
        print('Training downstream task w/o pre-training...')

    model = RobertaForMaskedLM.from_pretrained(
        args.model) if args.model != 'None' else new_model()
    model = model.cuda()
    model.train()

    from transformers import RobertaTokenizerFast
    # Create the tokenizer from a trained one
    tokenizer = RobertaTokenizerFast(
        tokenizer_file='./byte-level-bpe.tokenizer.json')

    train_dataset = X86PerfDownstreamDataset(evaluate=False)
    eval_dataset = X86PerfDownstreamDataset(evaluate=True)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5)

    writer = SummaryWriter(log_dir=os.path.join(
        args.save_dir, 'runs', datetime.now().strftime('%b%d-%H-%M')))

    with Progress() as progress:
        training_task = progress.add_task(
            "[red]Training...", total=args.epoch * len(train_loader))
        for n_epoch in range(args.epoch):
            running_loss = 0.0
            for i_batch, data_batch in enumerate(train_loader):
                input_ids, attention_mask, labels = data_batch
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                labels = labels.cuda()

                optimizer.zero_grad()
                loss = loss_fn(model(input_ids=input_ids,
                                     attention_mask=attention_mask).logits.mean(2), labels)
                loss.backward()
                optimizer.step()
                # print statistics
                running_loss += loss.item()

                progress.update(training_task, advance=1)

                if i_batch % args.log_freq == args.log_freq - 1:    # print every 2000 mini-batches
                    n_progress = len(train_loader) * n_epoch + i_batch
                    writer.add_scalar(
                        'Loss/train', running_loss / args.log_freq, n_progress)
                    with torch.no_grad():
                        eval_running_loss = 0.0
                        distance_err = 0
                        n_accurate = 0
                        n_total = 0
                        for i_eval, data_eval in enumerate(eval_loader):
                            input_ids, attention_mask, labels = data_eval
                            input_ids = input_ids.cuda()
                            attention_mask = attention_mask.cuda()
                            labels = labels.cuda()

                            out = model(input_ids=input_ids,
                                        attention_mask=attention_mask).logits.mean(2)
                            loss = loss_fn(out, labels)

                            eval_running_loss += loss.item()

                            pred_bottleneck = torch.argmax(out, dim=1)
                            label_bottleneck = torch.argmax(labels, dim=1)
                            distance_err += (pred_bottleneck -
                                             label_bottleneck).abs().sum().item()
                            n_accurate += (pred_bottleneck ==
                                           label_bottleneck).sum().item()
                            n_total += labels.shape[0]

                        writer.add_scalar(
                            'Loss/eval', eval_running_loss / len(eval_loader), n_progress)
                        writer.add_scalar(
                            'Accuracy/eval', n_accurate / n_total, n_progress)
                        writer.add_scalar(
                            'AvgTokenDistErr/eval', distance_err / n_total, n_progress)
                    running_loss = 0.0

    dirname = f'checkpoint-{n_epoch * len(train_loader)}'
    if args.model == 'None':
        dirname += '-scratch'
    model.save_pretrained(
        os.path.join(args.save_dir, dirname))
