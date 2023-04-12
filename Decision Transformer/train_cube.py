"""
To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False
"""
import os
import time
import math
import pickle
from contextlib import nullcontext
import numpy as np
import torch
from transformers import DecisionTransformerModel, DecisionTransformerConfig, Trainer, TrainingArguments
import logging  # Better than printing because it is saved in a log file as well
from dataclasses import dataclass

from cube_utilities import *

import hydra


def create_model_from_scratch(model_args, meta_vocab_size):
    """Creates a new model to train"""



def load_model(model_args, config):
    """Loads an old model"""
    

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, context, eval_iters):
    # TODO: Complete this

    """ Taken from CubeGPT if useful here
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with context:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
    """


# learning rate decay scheduler (cosine with warmup) # Ted: Dynamic learning rate IMO.
def get_lr(it, learning_rate, warmup_iters, lr_decay_iters, min_lr):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def get_batch(data, device, block_size, batch_size):
    """Get a batch from the inputted data.
    This is modified to simply take in the array.
    """


@hydra.main(version_base=None, config_path="config", config_name="config")
def train_from_scratch(config):
    """hydra decorated functions can only take in one parameter.
    This is one way to get around this."""
    return train(config, True)


@hydra.main(version_base=None, config_path="config", config_name="config")
def train_resume(config):
    """hydra decorated functions can only take in one parameter.
    This is one way to get around this."""
    return train(config, False)


@dataclass
class DecisionTransformerDataCollator:
    return_tensors: str = "pt"
    max_len: int = 30 #subsets of the episode we use for training
    state_dim: int = 26  # size of state space, 26 squares per state
    act_dim: int = 1  # size of action space
    max_ep_len: int = 100 # max episode length in the dataset
    scale: float = 1000.0  # normalization of rewards/returns
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, states, actions) -> None:
        self.states = states
        self.actions = actions
        # calculate dataset stats for normalization of states
        traj_lens = []
        for obs in states:
            traj_lens.append(len(obs))
        self.n_traj = len(traj_lens)
        states = np.vstack(states)
        traj_lens = np.array(traj_lens)
        self.p_sample = traj_lens / sum(traj_lens)

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # this is a bit of a hack to be able to sample of a non-uniform distribution
        batch_inds = np.random.choice(
            np.arange(self.n_traj),
            size=batch_size,
            replace=True,
            p=self.p_sample,  # reweights so we sample according to timesteps
        )
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        
        for ind in batch_inds:
            # for feature in features:
            state_trajectory = self.states[int(ind)]
            action_trajectory = self.actions[int(ind)]
            si = random.randint(0, len(state_trajectory) - 1)

            reward_trajectory = [-1 for _ in range(len(state_trajectory))]
            reward_trajectory[-1] = 1000

            dones = [False for _ in range(len(state_trajectory))]
            dones[-1] = True

            # get sequences from dataset
            s.append(np.array(state_trajectory[si : si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(action_trajectory[si : si + self.max_len]).reshape(1, -1, 1))
            r.append(np.array(reward_trajectory[si : si + self.max_len]).reshape(1, -1, 1))

            d.append(np.array(dones[si : si + self.max_len]).reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(
                self._discount_cumsum(np.array(reward_trajectory[si:]), gamma=1.0)[
                    : s[-1].shape[1]   # TODO check the +1 removed here
                ].reshape(1, -1, 1)
            )
            if rtg[-1].shape[1] < s[-1].shape[1]:
                print("if true")
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, self.state_dim)), s[-1]], axis=1)
            a[-1] = np.concatenate(
                [np.ones((1, self.max_len - tlen, self.act_dim)) * -10.0, a[-1]],
                axis=1,
            )
            r[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, self.max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, self.max_len - tlen, 1)), rtg[-1]], axis=1) / self.scale
            timesteps[-1] = np.concatenate([np.zeros((1, self.max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).float()
        a = torch.from_numpy(np.concatenate(a, axis=0)).float()
        r = torch.from_numpy(np.concatenate(r, axis=0)).float()
        d = torch.from_numpy(np.concatenate(d, axis=0))
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).float()
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).long()
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).float()

        return {
            "states": s,
            "actions": a,
            "rewards": r,
            "returns_to_go": rtg,
            "timesteps": timesteps,
            "attention_mask": mask,
        }


class TrainableDT(DecisionTransformerModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, **kwargs):
        output = super().forward(**kwargs)
        # add the DT loss
        action_preds = output[1]
        action_targets = kwargs["actions"]
        attention_mask = kwargs["attention_mask"]
        act_dim = action_preds.shape[2]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        
        loss = torch.nn.CrossEntropyLoss()

        
        return {"loss": loss(action_preds, action_targets)}

    def original_forward(self, **kwargs):
        return super().forward(**kwargs)



def train(config, start_from_scratch):
    """Trains a model on the current configurations.
    config: The dictionary of configurations.
    start_from_scratch: If False, load a previous checkpoint. Otherwise, start from scratch.
    output: The model (avoids needing to get the model from a file in Agent.py)
    """
    torch.manual_seed(1337 + config['seed_offset'])
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # Define the number of examples to generate
    num_examples = 1000
    len_examples = 20

    # Generate the challenges
    challenges = [challenge_generator(len_examples, 'internal_repr', False)[::-1] + ["END"] for _ in range(num_examples)]

    states = [list(map(tokenize_state, challenge[::2])) for challenge in challenges]
    actions = [list(map(tokenize_action, challenge[1::2])) for challenge in challenges]

    collator = DecisionTransformerDataCollator(states, actions)

    config = DecisionTransformerConfig(state_dim=collator.state_dim, act_dim=collator.act_dim)
    model = TrainableDT(config)

    training_args = TrainingArguments(
        output_dir="output/",
        remove_unused_columns=False,
        num_train_epochs=120,
        per_device_train_batch_size=64,
        learning_rate=1e-4,
        weight_decay=1e-4,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.25,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=challenges
    )

    trainer.train()
    torch.save(model.state_dict(), "trained_model")
    breakpoint()

    return model


if __name__ == '__main__':
    train_from_scratch()