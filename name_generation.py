import numpy as np
from numpy.typing import NDArray
def get_names(
        text_file: str
    ) -> list[str]:
    with open(text_file) as f:
        names = [name.strip() for name in f.readlines()]
    return names

def get_map(
        names: list[str]
    ) -> tuple[dict, dict]:
    chars = sorted(set("".join(names)))
    chars.append("<START>")
    chars.append("<END>")
    ctoi = {c: i for i, c in enumerate(chars)}
    itoc = {i: c for c, i in ctoi.items()}
    return ctoi, itoc

def compute_prob_stat_mat(
        names: list[str], 
        ctoi: dict[str, int]
    ) -> NDArray:
    ctoi, _ = get_map(names)
    stat_mat = np.zeros((len(ctoi), len(ctoi)), dtype=np.int64)    
    for name in names:
        name = ["<START>"] + list(name) + ["<END>"]
        for char, next_char in zip(name, name[1:]):
            row_idx = ctoi[char]
            col_idx = ctoi[next_char]
            stat_mat[row_idx, col_idx] += 1
    stat_mat = stat_mat[:-1]
    
    return stat_mat / stat_mat.sum(axis=1, keepdims=True)

def generate_name(
        num: int,
        ctoi: dict[str, int], 
        itoc: dict[int, str], 
        prob_stat_mat: NDArray,
        seed: int = 83479
    ) -> list[str]:
    rng = np.random.default_rng(seed)
    names = []
    for _ in range(num):
        new_name = ""
        idx = ctoi["<START>"]
        indexes = [i for i in range(len(ctoi))]
        while True:
            p = prob_stat_mat[idx]
            idx = rng.choice(
                indexes,
                size=1,
                p=p
            ).item()
            if idx == ctoi["<END>"]:
                break
            new_name += itoc[idx]
        names.append(new_name)
    
    return names