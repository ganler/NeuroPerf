from abc import ABC, abstractmethod
import os
import subprocess


class CProgramGen(ABC):
    @abstractmethod
    def gen() -> str:
        raise NotImplementedError


__CSMITH_HOME__ = '/home/ganler/Documents/csmith/build'


class CSmith(CProgramGen):
    @staticmethod
    def gen() -> str:
        os.system(f'{__CSMITH_HOME__}/src/csmith > random.c')
        assert os.path.isfile('./random.c')
        return f'./random.c -I{__CSMITH_HOME__}/include'


__YARPGEN_HOME__ = '/home/ganler/Documents/yarpgen/build'


class Yarpgen(CProgramGen):
    @staticmethod
    def gen() -> str:
        os.system(f'{__YARPGEN_HOME__}/yarpgen >/dev/null 2>&1')
        assert os.path.isfile('./driver.cpp')
        assert os.path.isfile('./func.cpp')
        assert os.path.isfile('./init.h')

        return './driver.cpp ./func.cpp'


from typing import List, Tuple


class PerfData:
    def __init__(self, data):
        # A list of assembly strings.
        self.data: List[Tuple(str, float)] = data

    def __str__(self):
        ret = ''
        for asm, perc in self.data:
            ret += f'{perc * 100:.2f}\t{asm}\n'
        return ret

    def __repr__(self):
        return self.__str__()


def perf_datagen(program):
    os.system(
        f'clang -g -O1 -gline-tables-only {program} -o a.out >/dev/null 2>&1')
    assert os.path.isfile('./a.out')
    os.system('perf record -e cycles:u ./a.out  >/dev/null 2>&1')
    raw_data = subprocess.check_output(
        ['perf', 'annotate', '--stdio']).decode()

    lines = raw_data.split('\n')
    ranges = []
    cur_range_start = None
    for i, line in enumerate(lines):
        if 'Percent |	Source code & Disassembly of' in line:
            if cur_range_start is not None:
                ranges.append((cur_range_start, i))
                cur_range_start = None

            if cur_range_start is None and 'a.out' in line:
                cur_range_start = i + 8

    ret_data = []

    for begin, end in ranges:
        data = []
        __START_TOKEN__ = '$START'
        start_addr_token = 'UNKNOWN'
        for k, c in enumerate(lines[begin]):
            if c.isalpha():
                start_addr_token = lines[begin][k:-1]
                break
        for i in range(begin, end):
            line = lines[i].replace(start_addr_token, __START_TOKEN__)
            if len(line) == len(lines[i]):  # no change
                line = lines[i].replace(start_addr_token[:-2], __START_TOKEN__)
            tokens = line.split()
            if len(tokens) == 0:
                continue
            try:
                float(tokens[0])
            except ValueError:
                continue

            # OK : we got an instruction like:
            #     0.00 :   1564:   xor    %rdx,%rcx
            for j, t in enumerate(tokens):
                if t == '#':
                    tokens = tokens[:j]
                    break

            # print(tokens)
            data.append(
                (
                    '\t'.join(tokens[3:]),
                    float(tokens[0])
                ))
        ret_data.append(PerfData(data))

    return ret_data


if __name__ == '__main__':
    __DATASET__ = 'DATASET'
    if not os.path.exists(__DATASET__):
        os.mkdir(__DATASET__)

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', type=int, default=10)

    import uuid
    import pickle

    args = parser.parse_args()

    from rich.progress import Progress

    n_cont_fail = 0

    with Progress() as progress:

        task = progress.add_task(
            "[green]Generating datasets...", total=args.num)
        while not progress.finished:
            data = None
            try:
                data = perf_datagen(Yarpgen().gen())
            except Exception as e:
                print(e)
                n_cont_fail += 1
                if n_cont_fail > 20:
                    print('Too many continuous failures, aborting.')
                    break
                continue

            for d in data:
                with open(f'{__DATASET__}/{uuid.uuid4()}.pkl', 'wb') as f:
                    pickle.dump(d, f)
                    progress.update(task, advance=1)
