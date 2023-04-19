import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

from train_cube import train_from_scratch
from cube_utilities import challenge_generator, init_state
from cube_utilities import cube_generate_training_file



class Agent:
    def __init__(self):
        """
        filter: The Filter object that filters the learning history
        policy: The Policy object that decides on the action take at each step
        episode_length: The maximum length of an episode
        difficulty_scheduler: A function that takes in the current number of iterations
            ran, and returns the number of moves we should shuffle this iteration.
        file_name: The name of the file that stores the learning history
        """
        #self.filter = filter
        #self.policy = policy
        #self.difficulty_scheduler = difficulty_scheduler
        #self.episode_length = episode_length
        #self.file_name = file_name
        #self.num_iterations_ran = 0
        self.model = None # TODO: Load "ckpt.pt" as done in "sample.py".

        


    def consolidate(self, num_iterations: int):
        """Run the consolidation step of the algorithm
        num_iterations: The number of iterations we run for
        """
        #self.generate_learning_history(num_iterations)
        #self.train()
        pass

    def train(self):
        """Train the model on the data in self.file_name"""
        self.model = train_from_scratch() # No need to pass config parameter since hydra.main will automatically fill it in. TODO: Is simply return model sufficient?

    def generate_training_file(self, num_examples: int, num_permute: int):
        cube_generate_training_file(num_examples, num_permute) 
         

    def generate_learning_history(self, num_iterations: int):
        """Repeatedly play for a specific number of iterations, then filter out the learning history
        num_iterations: The number of iterations we run for
        """
        #for _ in range(num_iterations):
        #    self.run_episode()
        #    self.num_iterations_ran += 1

        #self.filter.filter_data(self.file_name)
        pass
    
    def run_episode(self):
        """Run a single episode of the consolidation step, storing the result in the learning history file"""
#        start_state = challenge_generator(
#            self.difficulty_scheduler(self.num_iterations_ran),
#            "internal_repr",
#            False
#        )
#
#        # The learning history starts off with a (start, goal) pair
#        learning_history = [(start_state, init_state("internal_repr")), start_state]
#
#        # We initialize like this to keep learning_history the same length
#        # (e.g. for convenience in the autoencoder filter)
#        # as well as to avoid the more expensive appends.
#        learning_history += [''] * (self.episode_length * 2 - 1)
#
#        current_state = start_state
#        for i in range(1, 1 + self.episode_length):
#            if current_state == init_state("internal_repr"):
#                # We have solved the cube, so do nothing now.
#                learning_history[2 * i + 1] = current_state
#            else:
#                action = self.policy.next_action(current_state)
#                current_state = internal_cube_permute(current_state, [action])
#
#                learning_history[2 * i] = action
#                learning_history[2 * i + 1] = current_state
#
#        # TODO: Write the learning history to the file self.file_name.
#        return
        pass

    def solve(self, puzzle_file: str, inference_method: str):
        """
        - Given a problem (start_state, end_state), predict the solution.
        - Will use correction on state tokens so really generated action tokens are what matter.
        - Most likely still predict state token because chain-of-thought (i.e. ask model to explain itself is good.
        """
        # Prepare puzzle file.

        
        out_dir = 'out-cube_structure' # ignored if init_from is not 'resume'
        seed = 1337
        device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
        # dtype = 'bfloat16' # 'float32' or 'bfloat16' or 'float16'
        #dtype = 'float16' # 'float32' or 'bfloat16' or 'float16'
        dtype = 'float32' # 'float32' or 'bfloat16' or 'float16'



        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        print(checkpoint['model_args'])
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        #print(state_dict) # DEBUG.
        model.load_state_dict(state_dict)

        model.eval()
        #print("model.training: " + str(model.training)) # DEBUG.
        model.to(device)
        #print("model.training: " + str(model.training)) # DEBUG.

        # look for the meta pickle in case it is available in the dataset folder
        load_meta = False
        #if 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
        assert load_meta == True
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ' '.join([itos[i] for i in l])



        # encode the beginning of the prompt
        assert puzzle_file.startswith('PUZZLE_') == True
        with open(puzzle_file, 'r', encoding='utf-8') as f:
            puzzle = f.read().strip().split(" ")
            print("puzzle: " + str(puzzle))
        puzzle_ids = encode(puzzle)
        #print("puzzle_ids: " + str(puzzle_ids)) # DEBUG.
        x = (torch.tensor(puzzle_ids, dtype=torch.long, device=device)[None, ...])
        # print(x) # DEBUG. 



        # <model.generate()>
        with torch.no_grad():
            with ctx:
                y = model.generate(x, inference_method, max_new_tokens=20, temperature=0.8)
                print(decode(y[0].tolist()))




if __name__ == "__main__":
    agent = Agent()
    agent.solve("PUZZLE_1.txt", "action") # "token" or "action" for <inference_method>.
    #agent.solve("PUZZLE_1.txt", "token") 







