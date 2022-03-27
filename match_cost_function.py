import numpy as np


def ssd_similarity_measure(patch1: np.ndarray, patch2: np.ndarray) -> np.ndarray:

    assert patch1.shape == patch2.shape

    ssd_value = np.sum(np.power((patch1 - patch2), 2))

    return ssd_value


def sad_similarity_measure(patch1: np.ndarray, patch2: np.ndarray) -> np.ndarray:

    assert patch1.shape == patch2.shape
  
    sad_value = np.sum(np.abs((patch1 - patch2)))

    return sad_value


def ncc_similarity_measure(patch1: np.ndarray, patch2: np.ndarray) -> np.ndarray:

    assert patch1.shape == patch2.shape
    
    patches_std = patch1.std() * patch2.std()

    if (patches_std == 0):
        return 0
    
    ncc_value = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean())) / patches_std
    
    return ncc_value