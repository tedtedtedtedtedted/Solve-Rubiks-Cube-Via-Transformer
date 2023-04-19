"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import random
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from data.cube_structure.prepare import stoi, itos # Ted: Python allows importing variables from another file! This is the tokenization. 
from cube_utilities import internal_to_color, color_to_internal, cube_permute, action_space, action_space_strict, is_solved

torch.set_printoptions(threshold=torch.inf) # TODO: DEBUG. Remove later. This is to print tensor fully for debugging.

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout if self.training else 0
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        #self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        #print(hasattr(torch.nn.functional, 'scaled_dot_product_attention')) # DEBUG.
        #print(not self.training) # DEBUG
        #self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and not self.training # Ted: Dangerous to modify! Wrong to do because <__init__()> will not be called after initialization. 
        #print(self.flash)

        #if not self.flash: # TODO: Come back later and modify here.
            #print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            #self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
            #                            .view(1, 1, config.block_size, config.block_size))
           
        # Ted: Below shift masking, assuming the first 28 tokens are not to predict, but rather given as puzzle context.
        size_puzzle_tokens = 1 + 26 + 1 # TODO: Bug! Only mask during training! Here mask in both training and inference time!
        extended_mask = torch.tril(torch.ones(config.block_size + size_puzzle_tokens, config.block_size))
        mask = torch.narrow(extended_mask, 0, size_puzzle_tokens, config.block_size)  
        mask = mask.view(1, 1, config.block_size, config.block_size)
        self.register_buffer("bias", mask)
        # print(self.bias) # DEBUG.


    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # print("B: " + str(B) + "; T: " + str(T) + "; C: " + str(C)) # DEBUG.

        # print("x: " + str(x)) # DEBUG.
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # print("q: " + str(q)) # DEBUG.
        # print("k: " + str(k)) # DEBUG.
        # print("v: " + str(v)) # DEBUG.

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # self.flash = not self.training # Problem is that dropout layer is already initialized.
        # print("training time self.flash: " + str(self.flash))
#        if self.flash:
#            # efficient attention using Flash Attention CUDA kernels
#            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True) # Ted: Why does NanoGPT set <attn_mask = None>? Because we use flash attention during inference time not training time.
#        else:
            # manual implementation of attention
        # Ted: Always use home-made masking, even with slow attention.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Ted: Dim: [batch_size, num_heads, seq_len, seq_leln].
        # print("att raw: ") # DEBUG.
        # print(att) # DEBUG.
        #if self.training == True:
        if True: # TODO: Change it back!!! DEBUG. Reason is that there is huge discrepancy between inference time evaluation and train time evaluation.
            #print("Trainig time. Masking") # DEBUG.
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf')) # Ted: Only take what we need from the mask.
        #else:
            #print("Inference time. No masking") # DEBUG.
        att = F.softmax(att, dim=-1)
        #print("att_dim: ") # DEBUG.
        #print(att.size()) # DEBUG.
        #print("att: ") # DEBUG.
        #print(att) # DEBUG.
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        #print("att @ v: ") # DEBUG.
        #print(y) # DEBUG.
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout if self.training else 0)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # Ted: Skip connection.
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout if self.training else 0),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
#        self.cross_entropy_weight = torch.tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#                                                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#                                                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#                                                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#                                                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#                                                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#                                                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
#                                                  1., 1., 1., 1., 1., 1., 1., 1., 7., 7.,
#                                                  7., 7., 7., 7., 7., 7., 7., 7., 7., 7.,
#                                                  7., 7., 7., 7., 7., 7., 7., 1., 1.]).cuda() # More weights for 19 actions including "DONE". # TODO: Can try even more emphasis and also on special tokens.
        self.cross_entropy_weight = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                  0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                                  0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
                                                  1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
                                                  1., 1., 1., 1., 1., 1., 0.1, 0., 0.]).cuda() 
        #self.cross_entropy_weight = self.cross_entropy_weight / self.cross_entropy_weight.sum()
        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size() # Ted: In "train_cube.py" <model(X, Y)>, <X> is <idx> here so training batch.
        #print("batch size: " + str(b)) # DEBUG.
        #print("seq length: " + str(t)) # DEBUG.
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd) # Ted: Before embedding [batch_size, seq_len]
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb) # Ted: Pytorch broadcast addition.
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            #print("logits:") # DEBUG.
            #print(logits.view(-1, logits.size(-1)).size()) # DEBUG.
            #print(logits.view(-1, logits.size(-1))) # DEBUG.
            #print("Targets:") # DEBUG.
            #print(targets.view(-1).size()) # DEBUG.
            #print(targets.view(-1)) # DEBUG.
            # TODO: Weighted cross entropy!
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, weight=self.cross_entropy_weight)
            #print("training? " + str(self.training)) # DEBUG.
            print("loss: " + str(loss)) # DEBUG.
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim # Ted: select last token of sequence.
            loss = None

        return logits, loss

    def crop_block_size(self, block_size): # Ted: Need this to crop actually!!!
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, inference_method, max_new_tokens=35, temperature=1.0, top_k=None): # TODO: Add max_new_tokens to "config.yaml" and maybe allow larger number like 100 or 200.
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
    
        assert idx.size(dim=1) >= 1 + 26 + 1  # Ted: Assume input (token) length is at least 1 + 26 + 1.
        curr_state = idx[0].tolist()
        #curr_state = curr_state[(-1 - 26):-1] # Omit last token which is end-state-separator.
        curr_state = curr_state[(-1 - 26):-1] # Omit last token which is end-state-separator.
        curr_state = [itos[i] for i in curr_state] # Ted: Revert back to internal representation from tokens. 1. Need mapping 2. Need input.
        curr_state = internal_to_color(curr_state) # Ted: Need in color representation not internal representation, and in string format!



        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            #print("idx_cond: " + str(idx_cond)) # DEBUG.
            logits, _ = self(idx_cond) # Ted: Calling prediction.
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
#            if top_k is not None: # TODO: Uncomment!
#                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
#                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            
            #print("logits: " + str(logits)) # DEBUG.
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            #print("probs: " + str(probs)) # DEBUG.
            idx_next = torch.multinomial(probs, num_samples=1) # Ted: Return a tensor of one number which is the position index <i> and occur with probability a (multinomial distribution) function of <prob>.
            
            if inference_method == "token": # Generate token-by-token.
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)
            else:
                # Ted: Found it! Autoregressive with transition function here!
                move = itos[int(idx_next.item())] # Ted: <int()> cast is unnecessary but avoid complaining.
                print("move: " + move) # DEBUG.
                if (move not in action_space):
                    # Randomly sample an action or do nothing? Report this? Maybe report when assessing model stability, but good trained model shouldn't let this happen, at least not very often. Try random moves for now to break symmetry.
                    print("Invaild move, replace with random") # DEBUG.
                    move = random.choice(action_space_strict) # replace with random move.
                # Concatenate action.
                print("true move: " + move) # DEBUG.
                if (move == 'DONE'): # Just return and be done.
                    if (is_solved(curr_state)):
                        return idx
                    else:
                        print("Not DONE, replace with random") # DEBUG.
                        move = random.choice(action_space_strict) # replace with random move.
                        
                idx_next = torch.tensor([[stoi[move]]]) # Update <idx_next> (i.e. move).
                idx = torch.cat((idx, idx_next.cuda()), dim=1) # Concatenate new action.
                # Concatenate next state and separators.
                separator_state_begin = "I_SB"
                separator_state_end = "I_SE"
                separator_state_begin = torch.tensor([[stoi[separator_state_begin]]])
                separator_state_end = torch.tensor([[stoi[separator_state_end]]])
                idx = torch.cat((idx, separator_state_begin.cuda()), dim=1) # Concatenate new state-begin separator.
                curr_state = cube_permute(curr_state, move) # Ted: Note <cube_permute> takes in color representation, not internal representation.
                state_tensor = color_to_internal(curr_state)
                state_tensor = torch.tensor([[stoi[c] for c in state_tensor]]) # Ted: Encode.
                idx = torch.cat((idx, state_tensor.cuda()), dim=1)
                idx = torch.cat((idx, separator_state_end.cuda()), dim=1) # Concatenate new state-end separator.

            
        return idx
