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
    act_dim: int = 19  # size of action space
    max_ep_len: int = 100 # max episode length in the dataset
    scale: float = 1000.0  # normalization of rewards/returns
    p_sample: np.array = None  # a distribution to take account trajectory lengths
    state_mean: np.array = None  # to store state means
    state_std: np.array = None  # to store state stds
    n_traj: int = 0 # to store the number of trajectories in the dataset

    def __init__(self, states, actions, state_dim=26) -> None:
        self.state_dim = state_dim
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

        self.state_mean, self.state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6

    def _discount_cumsum(self, x, gamma):
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[-1] = x[-1]
        for t in reversed(range(x.shape[0] - 1)):
            discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
        return discount_cumsum

    def __call__(self, features):
        batch_size = len(features)
        # a batch of dataset features
        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for state, action in features:
            # for feature in features:
            state_trajectory = state
            action_trajectory = action
            si = 0

            reward_trajectory = [-1 for _ in range(len(state_trajectory))]
            reward_trajectory[-1] = 10

            dones = [False for _ in range(len(state_trajectory))]
            dones[-1] = True

            # get sequences from dataset
            s.append(np.array(state_trajectory[si : si + self.max_len]).reshape(1, -1, self.state_dim))
            a.append(np.array(action_trajectory[si : si + self.max_len]).reshape(1, -1, self.act_dim))
            r.append(np.array(reward_trajectory[si : si + self.max_len]).reshape(1, -1, 1))

            s[-1] = (s[-1] - self.state_mean) / self.state_std

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
            # TODO: Check if we should we add back the normalization?
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
    num_examples = config['num_examples']
    len_examples = config['len_examples']

    # Generate the challenges
    challenges = [challenge_generator(len_examples, config['representation'], False)[::-1] + ["END"] for _ in range(num_examples)]

    if config['representation'] == 'color':
        state_tokenizer = tokenize_colors
        state_dim = 54
    else:
        state_tokenizer = tokenize_state
        state_dim = 26

    states = [list(map(state_tokenizer, challenge[::2])) for challenge in challenges]
    actions = [list(map(tokenize_action, challenge[1::2])) for challenge in challenges]

    collator = DecisionTransformerDataCollator(states, actions, state_dim)

    config = DecisionTransformerConfig(
        state_dim=collator.state_dim,
        act_dim=collator.act_dim,
        n_head=config['n_heads'],
        n_inner=config['n_inner'],
        n_layer=config['n_layer']
    )
    model = TrainableDT(config)

    training_args = TrainingArguments(
        output_dir="output/",
        remove_unused_columns=False,
        num_train_epochs=50,
        per_device_train_batch_size=64,
        learning_rate=1e-2,
        weight_decay=1e-8,
        warmup_ratio=0.1,
        optim="adamw_torch",
        max_grad_norm=0.5,
        logging_steps=10,
        logging_dir="log"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=[(s, a) for s in states for a in actions]
    )

    result = trainer.train()
    trainer.save_model("trained_model")

    logSave = open('lossoutput.txt', 'a')
    logSave.write(str(trainer.state.log_history) + '\n')
    logSave.close()

    return model

def get_action(model, states, actions, rewards, returns_to_go, timesteps):
    # This implementation does not condition on past rewards

    states = states.reshape(1, -1, model.config.state_dim)
    actions = actions.reshape(1, -1, model.config.act_dim)
    returns_to_go = returns_to_go.reshape(1, -1, 1)
    timesteps = timesteps.reshape(1, -1)

    states = states[:, -model.config.max_length :]
    actions = actions[:, -model.config.max_length :]
    returns_to_go = returns_to_go[:, -model.config.max_length :]
    timesteps = timesteps[:, -model.config.max_length :]
    padding = model.config.max_length - states.shape[1]
    # pad all tokens to sequence length
    attention_mask = torch.cat([torch.zeros(padding), torch.ones(states.shape[1])])
    attention_mask = attention_mask.to(dtype=torch.long).reshape(1, -1)
    states = torch.cat([torch.zeros((1, padding, model.config.state_dim)), states], dim=1).float()
    actions = torch.cat([torch.zeros((1, padding, model.config.act_dim)), actions], dim=1).float()
    returns_to_go = torch.cat([torch.zeros((1, padding, 1)), returns_to_go], dim=1).float()
    timesteps = torch.cat([torch.zeros((1, padding), dtype=torch.long), timesteps], dim=1)

    state_preds, action_preds, return_preds = model.original_forward(
        states=states,
        actions=actions,
        rewards=rewards,
        returns_to_go=returns_to_go,
        timesteps=timesteps,
        attention_mask=attention_mask,
        return_dict=False,
    )

    return action_preds[0, -1]

def run_test(model, num_shuffles, device):
    state_dim = 26  # Number of tokens in every state
    act_dim = 19  # Number of tokens in every action
    max_ep_len = 30
    scale = 1
    target_return = 10 - num_shuffles   # We get 10 reward at the end, and -1 otherwise
    episode_return, episode_length = 0, 0

    internal_state = challenge_generator(num_shuffles, 'internal_repr', False)[-1].strip().split(" ")
    state = np.array(tokenize_state(" ".join(internal_state)))
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)

    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = get_action(
            model,
            states,
            actions,
            rewards,
            target_return,
            timesteps,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        internal_state = internal_cube_permute(internal_state, get_action_list()[np.argmax(action)])
        state = np.array(tokenize_state(" ".join(internal_state)))
        done = is_done(internal_state)
        if done:
            reward = 10
        else:
            reward = -1

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    breakpoint()

    logging.info(states)
    logging.info(actions)
    logging.info(rewards)


if __name__ == '__main__':
    train_from_scratch()

    # model = TrainableDT.from_pretrained("trained_model")
    # run_test(model, 1, "cpu")