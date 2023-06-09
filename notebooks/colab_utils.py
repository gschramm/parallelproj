def in_colab():
    """test if we are in google colab or not"""
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    return IN_COLAB


def tonumpy(x):
    """ function that converts a cupy or numpy array to an numpy array"""
    try:
        if cp.get_array_module(x).__name__ == 'cupy':
            return cp.asnumpy(x)
        else:
            return x
    except:
        return x