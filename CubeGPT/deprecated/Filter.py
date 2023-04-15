"""
A collection of different filters
"""
from cube_utilities import init_state

class Filter:
    def filter_data(self, input_data: str):
        """Take in the name of the file with all the data, process it,
        and create a filtered file.
        input_data: The name of the file
        """
        learning_history = None # TODO: Deal with inputting the file
        filtered_learning_history = self._filter(learning_history)
        # TODO: Deal with writing to the file

    def _filter(self, learning_history):
        """Given the total learning history, remove unsuccessful attempts, and then
        apply the specific filter."""
        filtered_learning_history = [learning_history[0]]  # Copy the (s, g) pair
        for episode in learning_history[1:]:
            final_state = episode[-1]
            if final_state == init_state("internal_repr") and self._is_filtered(episode):
                filtered_learning_history.append(episode)

        return filtered_learning_history

    def _is_filtered(self, episode):
        raise NotImplementedError


class NaiveFilter(Filter):
    def _is_filtered(self, input_data):
        """A filter that does nothing, but remove unsuccessful attempts"""
        return False


class AutoEncoderFilter(Filter):
    """Trains an autoencoder to compress the learning history along side the
    transformer, and uses its current performance as a measure of complexity,
    filtering if it passes some threshold."""
    def __init__(self, threshold):
        self.threshold = threshold
        self.data = []

    def _train(self):
        """Trains the autoencoder on the data in self.data for some number of iterations"""
        # TODO: Train on current data
        pass

    def _loss(self, input_data):
        """Finds how off the autoencoder is at reproducing input_data"""
        # TODO: Complete
        pass

    def _is_filtered(self, input_data):
        self.data.append(input_data)
        self._train()

        compressibility = self._loss(input_data)
        return compressibility > self.threshold