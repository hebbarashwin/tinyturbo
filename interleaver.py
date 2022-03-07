import torch
import numpy as np


class Interleaver:
    """ Random Interleaver.

    Parameters
    ----------
    length : int
        Length of the interleaver.

    seed : int
        Seed to initialize the random number generator
        which generates the random permutation for
        interleaving.

    p_array (optional) :
        Permutation array.
        If not specified, a random permutation is chosen

    Returns
    -------
    random_interleaver : RandInterlv object
        A random interleaver object.

    Note
    ----
    The random number generator is the
    RandomState object from NumPy,
    which uses the Mersenne Twister algorithm.

    """
    def __init__(self, length, seed, p_array = None):
        rand_gen = np.random.mtrand.RandomState(seed)
        if p_array is None:
            self.p_array = rand_gen.permutation(np.arange(length))
        else:
            self.p_array = p_array
        self.rev_p_array = np.argsort(self.p_array)

    def interleave(self, input):
        """ Interleave input array using the specific interleaver.

        Parameters
        ----------
        in_array : 2D torch tensor or np array (:, length)
            Input data to be interleaved.

        Returns
        -------
        out_array : 2D torch tensor or np array (:, length)
            Interleaved output data.

        """

        out = input[:, self.p_array]
        return out

    def deinterleave(self, input):
        """ De-interleave input array using the specific interleaver.

        Parameters
        ----------
        in_array : 2D torch tensor (:, length)
            Input data to be de-interleaved.

        Returns
        -------
        out_array : 2D torch tensor (:, length)
            De-Interleaved output data.

        """
        out = input[:, self.rev_p_array]
        return out

    def set_p_array(self, p_array):
        """ Set interleaver and de-interleaver permutation
        
        """

        self.p_array = p_array
        self.rev_p_array = np.argsort(self.p_array)
