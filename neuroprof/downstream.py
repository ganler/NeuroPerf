from transformers import RobertaConfig
from transformers import RobertaForMaskedLM
from neuroprof.dataset import X86PerfFineTuneDataset
import torch

if __name__ == '__main__':
    import argparse
    # args
    parser = argparse.ArgumentParser(description='Downstream')
    parser.add_argument('--model', type=str, help='checkpoint dir')
    args = parser.parse_args()

    # Set a configuration for our RoBERTa model
    config = RobertaConfig(
        vocab_size=2048,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        type_vocab_size=1,
    )
    # Initialize the model from a configuration without pretrained weights
    model = RobertaForMaskedLM(config=config).cuda()
    print('Num parameters: ', model.num_parameters())

    from transformers import RobertaTokenizerFast
    # Create the tokenizer from a trained one
    tokenizer = RobertaTokenizerFast.from_pretrained(
        'tokenizer_model', max_len=512)

    train_dataset = X86PerfFineTuneDataset(evaluate=False)
    eval_dataset = X86PerfFineTuneDataset(evaluate=True)

    checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(checkpoint['model_state_dict'])
