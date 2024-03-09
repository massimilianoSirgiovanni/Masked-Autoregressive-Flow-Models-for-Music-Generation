from torch import cuda
from gc import collect

def cleanCache(clean=True):
    if clean:
        if cuda.is_available():
            cuda.empty_cache()
        else:
            collect()