def tonumpy(x, xp):
    """return numpy representation of numpy or cupy array

    Parameters
    ----------
    x : numpy or cupy array
    xp : numpy or cupy module

    Returns
    -------
    numpy ndarray
    """    
    if xp.__name__ == 'numpy':
        return x
    elif xp.__name__ == 'cupy':
        return xp.asnumpy(x)
    else:
        raise ValueError