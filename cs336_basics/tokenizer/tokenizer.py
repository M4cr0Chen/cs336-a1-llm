import os
from collections import Counter, defaultdict
from multiprocessing import Manager, Process, Queue
from queue import Empty

import regex as re
from tqdm import tqdm, trange

from cs336_basics.tokenizer.merge_fn import (
    build_pair_heap,
    merge_pairs_with_heap_index,
    pop_most_frequent_pair,
)

from cs336_basics.tokenizer.utils import (
    find_chunk_boundaries,
    string_to_bytes,
    save_vocab_and_merges,
)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
NUM_PROCESSES = min(4, os.cpu_count() or 1)

def init_vocab(special_tokens: list[str] | None = None) -> dict[int, bytes]:
    vocab: dict[int, bytes] = {x : bytes([x]) for x in range(256)}
    curr_index = 256

    if special_tokens:
        for token in special_tokens:
            token_bytes = token.encode("utf-8")
            vocab[curr_index] = token_bytes
            curr_index += 1
    
    return vocab

def update_vocab(vocab: dict[int, bytes], pair: tuple[int, int]) -> int:
    new_id = len(vocab)
    vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
    return new_id

def split_by_special_tokens(text: str, special_tokens: list[str], include_special: bool = False) -> list[str]:
    if not special_tokens:
        return [text]
    
    special_tokens_sorted = sorted(special_tokens, key = len, reverse = True)
    pattern = "|".join(re.escape(t) for t in special_tokens_sorted)

    if include_special:
        special_chunks = re.split(f"({pattern})", text)
    else:
        special_chunks = re.split(pattern, text)
    
    return special_chunks

def pre_tokenize(string: str, special_tokens: list[str], including_special: bool = False) -> Counter:
    word_counter = Counter()

    parts = split_by_special_tokens(string, special_tokens, include_special=including_special)

    for part in parts:
        if including_special and part in special_tokens:
            word_counter[tuple(string_to_bytes(part))] += 1
        else:
            for match in re.finditer(PAT, part):
                word = match.group(0)
                word_encoded = tuple(string_to_bytes(word, return_int=True))
                word_counter[word_encoded] += 1
    return word_counter

def pre_tokenize_string_worker(*args):
    input_path, special_tokens, queue, start, end, include_special = args

    # Read the chunk from the file
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    word_counter = pre_tokenize(chunk, special_tokens, include_special)

    # Put the result in the queue
    queue.put(word_counter)


def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str] | None = None,
        verbose: bool = False,
        **kwargs,
):
    num_merges = vocab_size - 256 - (len(special_tokens) if special_tokens else 0)
    vocab: dict[int, bytes] = init_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []

    # Pre-tokenization
    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(
            f, desired_num_chunks=kwargs.get("desired_num_chunks", NUM_PROCESSES), split_special_token=b"\n"
        )
    
    manager = Manager()
    queue = manager.Queue()
    processes: list[Process] = []
    
    for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
        p = Process(
            target=pre_tokenize_string_worker,
            args=(input_path, special_tokens, queue, start, end, False),
        )
        processes.append(p)
        p.start()

    word_counter = Counter() 
    for _ in range(len(processes)):
        try:
            partial_counter = queue.get(timeout=10)
            word_counter.update(partial_counter)
        except Empty:
            continue
    for p in processes:
        p.join()

    pairs_counter = Counter()
    pair_to_words: dict[tuple[int, int], set[tuple[int, ...]]] = defaultdict(set)
    for word in word_counter:
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_to_words[pair].add(word)
            pairs_counter[pair] += word_counter[word]
    

    pair_heap = build_pair_heap(pairs_counter, vocab)

    for i in trange(num_merges):
        most_frequent_pair = pop_most_frequent_pair(pair_heap, pairs_counter)
        new_id = update_vocab(vocab, most_frequent_pair)

        word_counter, pairs_counter, pair_heap, pair_to_words = merge_pairs_with_heap_index(
            word_counter, pairs_counter, most_frequent_pair, new_id, vocab, pair_heap, pair_to_words
        )

        merges.append((vocab[most_frequent_pair[0]], vocab[most_frequent_pair[1]]))

    if kwargs.get("save_path"):
        save_vocab_and_merges(vocab, merges, kwargs["save_path"])
        with open(os.path.join(kwargs["save_path"], "special_tokens.txt"), "w", encoding="utf-8") as f:
            if special_tokens:
                for token in special_tokens:
                    f.write(f"{token}\n")

    return vocab, merges
