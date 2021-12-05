from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from pathlib import Path
import torch

from multiprocessing import Pool, cpu_count
import time

__GLOBAL_TOKENIZER__ = RobertaTokenizerFast(
    tokenizer_file='./byte-level-bpe.tokenizer.json')
def tokenize_them_all(file_list):
    ret = []
    for path in file_list:
        ret.append(__GLOBAL_TOKENIZER__.encode_plus(Path(path).read_text(), max_length=512, truncation=True, padding=True).input_ids)
    return ret

class X86PerfFineTuneDataset(Dataset):
    def __init__(self, evaluate: bool = False):
        # Create the tokenizer from a trained one
        self.examples = []

        files = [str(x) for x in Path("./DATASET/text/").glob("**/*.txt")]
        if evaluate:
            files = files[:len(files) // 10]
        else:
            files = files[len(files) // 10:]

        
        t0 = time.time()

        n_process = max(1, min(len(files) // 128, cpu_count() - 1))
        if n_process <= 1:
            print('... loading dataset :: [single-threads]')
            for path in files:
                self.examples.append(
                    __GLOBAL_TOKENIZER__.encode_plus(Path(path).read_text(), max_length=512, truncation=True, padding=True).input_ids)
        else:
            print(f'... loading dataset :: [{n_process}-threads]')
            size = (len(files) + n_process - 1) // n_process
            with Pool(n_process) as p:
                task_list = []
                for i in range(n_process):
                    task_list.append(files[i*size:(i+1)*size])
                for v in p.map(tokenize_them_all, task_list):
                    self.examples += v
        data_loading_time = time.time() - t0
        print(f'... finished in {data_loading_time}s --- {len(files) / data_loading_time} files / s')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])
