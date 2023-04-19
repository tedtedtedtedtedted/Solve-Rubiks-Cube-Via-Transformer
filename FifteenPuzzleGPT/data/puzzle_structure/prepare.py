# TODO: Modify to use input cube structure randomly generated.


"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# Download the tiny cube-structure dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

with open(input_file_path, 'r') as f:
    # data = f.read() # Ted: Removed.
    # Ted: Below added for train & validation sets.
    lines = f.readlines()
    num_lines = len(lines)
    print("num_lines: " + str(num_lines))  # Ted: DEBUG.
# print(f"length of dataset in characters: {len(data):,}")

empty_space_char = 0
size = 4

tokens_cube = [str(i) for i in range(size*size)] + \
    ["l", "r", "d", "u", "DONE", "I_SE", "I_SB", "\n"]

# TODO: Under this setting, cannot treat whole file as a big string.


# TODO: How does trainner know tokens are NOT char-level?
tokens_size = len(tokens_cube)
#print("all the unique tokens:", ''.join(tokens))
#print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(tokens_cube)}
itos = {i: ch for i, ch in enumerate(tokens_cube)}


def encode(s):
    s_list = (s.strip()).split(" ")
    # encoder: take a string, output a list of integers # TODO: WB end-of-line character?
    return [stoi[c] for c in s_list]


def decode(l):
    # ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    # return ' '.join([itos[i] for i in l]) # Ted: Modified.
    return [itos[i] for i in l]  # Ted: Modified.


# create the train and test splits
#n = len(data)
#train_data = data[:int(n*0.9)]
#val_data = data[int(n*0.9):]
# Ted: Cannot naively treat file as one big string in our setting:
train_data = lines[:round(num_lines * 0.9)]
val_data = lines[round(num_lines * 0.9):]
# Ted: Convert above data back to string.
train_data = ''.join(train_data)
val_data = ''.join(val_data)


# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': tokens_size,  # TODO: Maybe future change the name?
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# length of dataset in characters:  1115394
# all the unique characters:
#  !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz
# vocab size: 65
# train has 1003854 tokens
# val has 111540 tokens
