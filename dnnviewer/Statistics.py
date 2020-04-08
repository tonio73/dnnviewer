import numpy as np


class Statistics:

    @staticmethod
    def get_strongest(weights, topn):
        """ Get the top n strongest (max absolute value) weights of each row """
        nstrongest_idx = np.argpartition(np.abs(weights), -topn, axis=0)[-topn:]
        nstrongest = np.array([[weights[nstrongest_idx[i, j], j] for j in range(nstrongest_idx.shape[1])]
                               for i in range(topn)])

        return nstrongest_idx, nstrongest
