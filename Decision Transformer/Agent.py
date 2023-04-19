from train_cube import TrainableDT, run_test, train
import numpy as np
import hydra
import logging
import pickle

class Agent:
    def __init__(self, episode_length, config):
        """
        episode_length: The maximum length of an episode
        file_name: The name of the file that stores the learning history
        """
        self.episode_length = episode_length
        self.num_iterations_ran = 0
        self.model = None
        self.config = config
        self.replay_buffer_states = []
        self.replay_buffer_actions = []

        self.initialize_model = True

    def consolidate(self, num_iterations: int, difficulty_scheduler, epsilon_scheduler):
        """Run the consolidation step of the algorithm
        num_iterations: The number of iterations we run for
        """
        self.generate_learning_history(num_iterations, difficulty_scheduler, epsilon_scheduler)
        logging.info(f"{{num_iterations: {self.num_iterations_ran}, epsilon: {epsilon_scheduler(self.num_iterations_ran)}, shuffles: {difficulty_scheduler(self.num_iterations_ran)}, num_examples: {len(self.replay_buffer_states)}}}")
        self.model = train(self.config, self.initialize_model, False, self.replay_buffer_states, self.replay_buffer_actions)

        if self.initialize_model:
            self.initialize_model = False

    def generate_learning_history(self, num_iterations: int, difficulty_scheduler, epsilon_scheduler):
        """Repeatedly play for a specific number of iterations, then filter out the learning history
        num_iterations: The number of iterations we run for
        """
        for _ in range(num_iterations):
            self.run_episode(difficulty_scheduler, epsilon_scheduler)

        self.num_iterations_ran += num_iterations

    def run_episode(self, difficulty_scheduler, epsilon_scheduler):
        is_done, _, states, actions = run_test(
            self.model,
            difficulty_scheduler(self.num_iterations_ran),
            "cpu",
            0,
            1,
            True,
            epsilon_scheduler(self.num_iterations_ran)
        )
        if is_done:
            self.replay_buffer_states.append(states.tolist())
            self.replay_buffer_actions.append(actions.tolist())


@hydra.main(version_base=None, config_path="config", config_name="config_agent")
def train_agent(config):
    difficulty_scheduler = lambda n: 1 + round((n/1000) ** 0.4)
    epsilon_scheduler = lambda n: 1 if n < 1000 else pow(0.95, (n/1000))
    agent = Agent(30, config)
    length_of_consolidation = 1000
    num_consolidations = 50
    for _ in range(num_consolidations):
        agent.consolidate(length_of_consolidation, difficulty_scheduler, epsilon_scheduler)

        with open('agent.pkl', 'wb') as f:
            pickle.dump(agent, f)

if __name__ == '__main__':
    train_agent()
