from Filter import Filter
from Policy import Policy
from cube_utilities import challenge_generator, init_state, internal_cube_permute

import torch
import os
from model import GPT, GPTConfig
import subprocess

class Agent:
    def __init__(self, filter: Filter, policy: Policy, episode_length: int, difficulty_scheduler: function, file_name: str):
        """
        filter: The Filter object that filters the learning history
        policy: The Policy object that decides on the action take at each step
        episode_length: The maximum length of an episode
        difficulty_scheduler: A function that takes in the current number of iterations
            ran, and returns the number of moves we should shuffle this iteration.
        file_name: The name of the file that stores the learning history
        """
        self.filter = filter
        self.policy = policy
        self.difficulty_scheduler = difficulty_scheduler
        self.episode_length = episode_length
        self.file_name = file_name
        self.num_iterations_ran = 0
        self.model = None
        """
        We want model to support two tasks:
        
        model.train: Train on some data
        model.generate: Output moves
        """

    def consolidate(self, num_iterations: int):
        """Run the consolidation step of the algorithm
        num_iterations: The number of iterations we run for
        """
        self.generate_learning_history(num_iterations)
        self.train()

    def train(self):
        """Train the model on the data in self.file_name"""
        # data = load_file(self.file_name)
        command_prepare_data = "python data/cube_structure/prepare.py"

        # Train the model on the data
        command = """python train.py config/train_shakespeare_char.py --device=cpu --compile=False
        --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --n_head=4
        --n_embd=128 --max_iters=2000 --lr_decay_iters=2000 --dropout=0.0"""
        output = subprocess.check_output(command, shell=True)
        print(output.decode())

        # Update the model
        self.update_model('ckpt.pt')

    def update_model(self, file_name):
        """Update self.model using the checkpoint in file_name
        """
        ckpt_path = os.path.join('out', 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        gptconf = GPTConfig(**model_args)
        self.model = GPT(gptconf)

    def generate_learning_history(self, num_iterations: int):
        """Repeatedly play for a specific number of iterations, then filter out the learning history
        num_iterations: The number of iterations we run for
        """
        for _ in range(num_iterations):
            self.run_episode()
            self.num_iterations_ran += 1

        self.filter.filter_data(self.file_name)

    def run_episode(self):
        """Run a single episode of the consolidation step, storing the result in the learning history file"""
        start_state = challenge_generator(
            self.difficulty_scheduler(self.num_iterations_ran),
            "internal_repr",
            False
        )

        # The learning history starts off with a (start, goal) pair
        learning_history = [(start_state, init_state("internal_repr")), start_state]

        # We initialize like this to keep learning_history the same length
        # (e.g. for convenience in the autoencoder filter)
        # as well as to avoid the more expensive appends.
        learning_history += [''] * (self.episode_length * 2 - 1)

        current_state = start_state
        for i in range(1, 1 + self.episode_length):
            if current_state == init_state("internal_repr"):
                # We have solved the cube, so do nothing now.
                learning_history[2 * i + 1] = current_state
            else:
                action = self.policy.next_action(current_state)
                current_state = internal_cube_permute(current_state, [action])

                learning_history[2 * i] = action
                learning_history[2 * i + 1] = current_state

    def predict(self, problem):
        """
        - Given a problem (start_state, end_state), predict the solution.
        - Will use correction on state tokens so really generated action tokens are what matter.
        - Most likely still predict state token because chain-of-thought (i.e. ask model to explain itself is good.
        """





        # TODO: Write the learning history to the file self.file_name.
