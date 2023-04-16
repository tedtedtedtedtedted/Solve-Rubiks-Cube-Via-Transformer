from Filter import Filter
from Policy import Policy
from train_cube import train_from_scratch
from cube_utilities import challenge_generator, init_state, internal_cube_permute
from cube_utilities import cube_generate_training_file


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
        #self.filter = filter
        #self.policy = policy
        #self.difficulty_scheduler = difficulty_scheduler
        self.episode_length = episode_length
        self.file_name = file_name
        self.num_iterations_ran = 0
        self.model = None

    def consolidate(self, num_iterations: int):
        """Run the consolidation step of the algorithm
        num_iterations: The number of iterations we run for
        """
        #self.generate_learning_history(num_iterations)
        #self.train()
        pass

    def train(self):
        """Train the model on the data in self.file_name"""
        self.model = train_from_scratch(...) # TODO: Pass <config> parameter.

    def generate_training_file(num_examples: int, num_permute: int):
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

    def predict(self, problem):
        """
        - Given a problem (start_state, end_state), predict the solution.
        - Will use correction on state tokens so really generated action tokens are what matter.
        - Most likely still predict state token because chain-of-thought (i.e. ask model to explain itself is good.
        """
        pass
