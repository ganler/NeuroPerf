from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from pathlib import Path
import torch


class X86PerfFineTuneDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        # Create the tokenizer from a trained one
        tokenizer = RobertaTokenizerFast(
            tokenizer_file='./byte-level-bpe.tokenizer.json')

        self.examples = []

        files = list(Path("./DATASET/text/").glob("**/*.txt"))
        if evaluate:
            files = files[:len(files) // 10]
        else:
            files = files[len(files) // 10:]

        for path in files:
            self.examples.append(
                tokenizer.encode_plus(path.read_text(), max_length=512, truncation=True, padding=True).input_ids)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])
