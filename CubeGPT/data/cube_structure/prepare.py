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
    #data = f.read() # Ted: Removed.
    # Ted: Below added for train & validation sets.
    lines = f.readlines()
    num_lines = len(lines)
    print("num_lines: " + str(num_lines)) # Ted: DEBUG.
# print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
#chars = sorted(list(set(data)))
#tokens_char = # TODO: Char tokens can also try.
tokens_cube = ["YOG", "YO", "YOB", "YG", "Y", "YB", "YRG", "YR", "YRB", "OG", "O", "OB", "G", "B", "RG", "R", "RB", "WOG", "WO", "WOB", "WG", "W", "WB", "WRG", "WR", "WRB",
               "OGY", "OY", "OBY", "GY",      "BY", "RGY", "RY", "RBY", "GO",      "BO",           "GR",      "BR", "OGW", "OW", "OBW", "GW",      "BW", "RGW", "RW", "RBW",
               "GYO",       "BYO",                  "GYR",       "BYR",                                             "GWO",       "BWO",                  "GWR",       "BWR",
               "YGO",       "YBO",                  "YGR",       "YBR",                                             "WGO",       "WBO",                  "WGR",       "WBR",
               "OYG",       "OYB",                  "RYG",       "RYB",                                             "OWG",       "OWB",                  "RWG",       "RWB",
               "GOY",       "BOY",                  "GRY",       "BRY",                                             "GOW",       "BOW",                  "GRW",       "BRW",
               "U-", "D-", "F-", "B-", "L-", "R-", "V-", "H-", "S-", # Actions.
               "u-", "d-", "f-", "b-", "l-", "r-", "v-", "h-", "s-",
               "DONE", # Also an action specifying done.
               "I_SB", "I_SE" # Separators for state.
               # Canceled: End of token. # Although this will never be in action space of transformer, for simplicity we comprimise, since we need to encode training file into ".bin" file.
               ] # Cube-color tokens. 





# TODO: Under this setting, cannot treat whole file as a big string.





# TODO: How does trainner know tokens are NOT char-level?
tokens_size = len(tokens_cube)
#print("all the unique tokens:", ''.join(tokens))
#print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(tokens_cube) }
itos = { i:ch for i,ch in enumerate(tokens_cube) }
def encode(s):
    s_list = (s.strip()).split(" ") # Ted: <strip()> removes '\n' and blank space!
    #print("Inside encode: ") # DEBUG.
    #print(s_list) # DEBUG.
    return [stoi[c] for c in s_list] # encoder: take a string, output a list of integers # TODO: WB end-of-line character?
def decode(l):
    #''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string
    #return ' '.join([itos[i] for i in l]) # Ted: Modified.
    return [itos[i] for i in l] # Ted: Modified.

# create the train and test splits
#n = len(data)
#train_data = data[:int(n*0.9)]
#val_data = data[int(n*0.9):]
# Ted: Cannot naively treat file as one big string in our setting:



train_data = lines[:round(num_lines * 0.9)]
val_data = lines[round(num_lines * 0.9):]


#print(train_data) # DEBUG.
# print(train_data[0]) # DEBUG.
# print(len(train_data[0])) # DEBUG.
# print(train_data[442]) # DEBUG.
# print(len(train_data[442])) # DEBUG.
# print(train_data[-1]) # DEBUG.
# print(len(train_data[-1])) # DEBUG.
# print(val_data[0]) # DEBUG.
# print(len(val_data[0])) # DEBUG.
# print(val_data[330]) # DEBUG.
# print(len(val_data[330])) # DEBUG.
# print(val_data[-1]) # DEBUG.
# print(len(val_data[-1])) # DEBUG.
# 
# print("encode length: ")
# print(len(encode(train_data[0]))) # DEBUG.
# print(len(encode(train_data[442]))) # DEBUG.
# print(len(encode(train_data[-1]))) # DEBUG.
# print(len(encode(val_data[0]))) # DEBUG.
# print(len(encode(val_data[330]))) # DEBUG.
# print(len(encode(val_data[-1]))) # DEBUG.
# 
# print("encode raw: ")
# print(encode(train_data[0])) # DEBUG.
# print(encode(train_data[442])) # DEBUG.
# print(encode(train_data[-1])) # DEBUG.
# print(encode(val_data[0])) # DEBUG.
# print(encode(val_data[330])) # DEBUG.
# print(encode(val_data[-1])) # DEBUG.


# Ted: Convert above data back to string.
train_data = ''.join(train_data)
val_data = ''.join(val_data)
train_data = train_data.replace(' \n', '')
val_data = val_data.replace(' \n', '')
#print(val_data) # DEBUG.




# encode both to integers
train_ids = encode(train_data)
val_ids = encode(val_data)
#print(train_ids) # DEBUG.
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint32)
val_ids = np.array(val_ids, dtype=np.uint32)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': tokens_size, # TODO: Maybe future change the name?
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
