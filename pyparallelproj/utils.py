class EventMultiplicityCounter:

    def __init__(self, xp):
        self._xp = xp

    def count(self, events):
        """ Count the multiplicity of events in an LM file

            Parameters
            ----------

            events : 2D numpy/cupy array
              of LM events of shape (n_events, 5) where the second axis encodes the event 
              (e.g. detectors numbers and TOF bins)
        """

        if self._xp.__name__ == 'cupy':
            from .utils_cupy import cupy_unique_axis0

            if not isinstance(events, self._xp.ndarray):
                # if the event array is not a cupy array, we first have to send it to the GPU
                events_d = self._xp.array(events)
            else:
                events_d = events

            tmp_d = cupy_unique_axis0(events_d,
                                      return_counts=True,
                                      return_inverse=True)
            mu_d = tmp_d[2][tmp_d[1]]

            if not isinstance(events, self._xp.ndarray):
                # if the event array is not a cupy array, we have to get the result back from the GPU
                mu = self._xp.asnumpy(mu_d)
            else:
                mu = mu_d
        elif self._xp.__name__ == 'numpy':
            tmp = self._xp.unique(events,
                                  axis=0,
                                  return_counts=True,
                                  return_inverse=True)
            mu = tmp[2][tmp[1]]

        return mu
