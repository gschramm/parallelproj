def in_colab():
    """test if we are in google colab or not"""
    try:
        import google.colab
        IN_COLAB = True
    except:
        IN_COLAB = False
    return IN_COLAB