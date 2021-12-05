from os import linesep
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast, BatchEncoding
from tokenizers import Encoding, Tokenizer
from pathlib import Path
import torch
import numpy as np

from multiprocessing import Pool, cpu_count
import time
import pickle

__G_FAST_TOKENIZER__ = RobertaTokenizerFast(
    tokenizer_file='./byte-level-bpe.tokenizer.json')

__G_LONG_TOKENIZER__ = Tokenizer.from_file('./byte-level-bpe.tokenizer.json')
__G_LONG_TOKENIZER__.enable_truncation(max_length=512 * 8) # Longer range.

def encode_string(string):
    return __G_FAST_TOKENIZER__.encode_plus(
        string, max_length=512, truncation=True, padding=True).input_ids

def encode_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
        
        bottleneck_id = np.argmax([v for _, v in data.data])
        lines = [line for line, _ in data.data]
        
        def get_line_range(line_id):
            line_id = min(line_id, len(lines) - 1)
            lrange = (0, 0)
            for line in lines[:line_id]:
                lrange[0] += len(line)
            lrange[1] = lrange[0] + len(lines[line_id])
            return lrange

        ########################################################################
        # lines[bottleneck_id] is unique in the file!

        # line_start = max(0, bottleneck_id - 1)
        # for _ in range(128):
        #     if line_start == 0 or lines[line_start - 1] == lines[bottleneck_id]:
        #         break
        #     line_start -= 1
        
        # line_end = min(len(lines) - 1, bottleneck_id + 1)
        # for _ in range(128):
        #     if line_end == len(lines) - 1 or lines[line_end] == lines[bottleneck_id]:
        #         break
        #     line_end += 1
        ########################################################################
        
        line_start = max(0, bottleneck_id - 128)
        line_end = min(len(lines) - 1, bottleneck_id + 128)

        original_bottleneck_range = get_line_range(bottleneck_id)
        lower_lim = get_line_range(bottleneck_id - 128)[0]
        upper_lim = get_line_range(bottleneck_id + 128)[1]
        offset_bottleneck_range = (
            original_bottleneck_range[0] - lower_lim, 
            original_bottleneck_range[1] - upper_lim)

        cache = __G_LONG_TOKENIZER__.encode_plus(
            '\n'.join(lines[line_start:line_end]))

        if cache.offsets[-1] >= offset_bottleneck_range[1]:
            return BatchEncoding(cache.truncate(512)), bottleneck_id - line_start
        else:
            bottleneck_token_id_end = None
            for i, offset in enumerate(cache.offsets):
                if offset[1] >= offset_bottleneck_range[1]:
                    bottleneck_token_id_end = i
                    break
            cache_valid_start = max(0, bottleneck_token_id_end - 512)
            return BatchEncoding(Encoding(cache.ids[cache_valid_start:(cache_valid_start + 512)])), bottleneck_id - line_start


def encode_text(path):
    return encode_string(Path(path).read_text())

def tokenize_them_texts(file_list):
    ret = []
    for path in file_list:
        ret.append(encode_text(path))
    return ret

class X86PerfFineTuneDataset(Dataset):
    def __init__(self, evaluate: bool = False, train_percent = 0.9, pretraining = True): # TODO: Pretraining...
        # Create the tokenizer from a trained one
        self.examples = []

        files = [str(x) for x in Path("./DATASET/text/").glob("**/*.txt")]
        n_eval = int(len(files) * (1 - train_percent))
        if evaluate:
            files = files[:n_eval]
        else:
            files = files[n_eval:]

        
        t0 = time.time()

        n_process = max(1, min(len(files) // 128, cpu_count() - 1))
        if n_process <= 1:
            print('... loading dataset :: [single-threads]')
            for path in files:
                self.examples.append(encode_text(path))
        else:
            print(f'... loading dataset :: [{n_process}-threads]')
            size = (len(files) + n_process - 1) // n_process
            with Pool(n_process) as p:
                task_list = []
                for i in range(n_process):
                    task_list.append(files[i*size:(i+1)*size])
                for v in p.map(tokenize_them_texts, task_list):
                    self.examples += v
        data_loading_time = time.time() - t0
        print(f'... finished in {data_loading_time}s --- {len(files) / data_loading_time} files / s')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i])
