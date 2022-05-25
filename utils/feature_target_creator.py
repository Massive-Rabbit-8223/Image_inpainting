"""
Author: Patrick Mederitsch
Matr.Nr.: K51831808
Exercise 4
"""

import numpy as np

def ex4(image_array: np.ndarray, offset: tuple, spacing: tuple) -> tuple:
    """
    This function creates two input arrays and one target array from a given input image. For this, the function
    has to remove (=set values to zero) all pixels that do not lie on a grid specified by offset and spacing.

    image_array:    A numpy array of shape (M, N, 3) and numeric datatype, which contains the RGB image data.
    offset:         A tuple containing 2 int values. These two values specify the offset of the first
                    grid point in x and y direction.
    spacing:        A tuple containing 2 int values. These two values specify the spacing between
                    two successive grid points in x and y direction.

    The function should return a 3-tuple (input_array, known_array, target_array).

    input_array:    Should be a 3D numpy array of shape (3, M, N) and the same datatype as image_array. 
                    It should have the same pixel values as image-array, with the exception that the 
                    to-be-removed pixel values off the specified grid are set to 0.
    known_array:    Should be a 3D numpy array of same shape and datatype as input_array, where pixels on 
                    the specified grid should have value 1 and all other unknown pixels have value 0.
    target_array:   Should be a 1D numpy array of the same datatype as image_array. It should hold 
                    the R-, G-, and B-pixel values at the off-grid locations (the pixels that were
                    set to 0 in input_array). The length of the 1D numpy array target_array is 
                    the number of removed pixels times 3.
    """

    if not isinstance(image_array, np.ndarray):
        raise TypeError("image_array is not a numpy array!")
    if len(image_array.shape) != 3:
        raise NotImplementedError("image_array is not a 3D array!")
    if image_array.shape[2] != 3:
        raise NotImplementedError("image_array, the size of the 3rd dimension is not equal to 3!")

    try:    # test if values are convertible to int objects
        for i in range(2):
            int(offset[i])
            int(spacing[i])
    except:
        raise ValueError("The values in offset and spacing are not convertible to int objects!")

    if (min(offset) < 0) or (max(offset) > 32):
        raise ValueError("The value for offset is outside the range [0, 32]!")
    if (min(spacing) < 2) or (max(spacing) > 8):
        raise ValueError("The value for spacing is outside the range [2, 8]!")

    
    #input_array = np.copy(image_array)

    ### create mask with specified offset and spacing ###
    known_array = np.zeros_like(image_array)

    image_rows = np.arange(image_array.shape[0])
    image_cols = np.arange(image_array.shape[1])

    rows_keep = np.where(((image_rows)%spacing[1])== 0)[0]+offset[1]   # check if last rows are still within bound due to the offset
    cols_keep = np.where(((image_cols)%spacing[0])== 0)[0]+offset[0]   # check if last rows are still within bound due to the offset

    rows_keep = rows_keep[rows_keep <= max(image_rows)]
    cols_keep = cols_keep[cols_keep <= max(image_cols)]

    known_array[rows_keep] = known_array[rows_keep] + 1
    known_array[:, cols_keep] = known_array[:, cols_keep] + 1
    known_array[known_array<2] = 0

    known_array = known_array/2

    input_array = np.multiply(image_array, known_array)
    input_array = np.transpose(input_array, (2,0,1))

    #print((known_array.transpose((2,0,1))<1).shape)
    target_array = image_array.transpose((2,0,1))[known_array.transpose((2,0,1)) < 1]

    # raise ValueError: if the number of the remaining known image pixels would be smaller than 144.
    if known_array.sum(axis=(0,1))[0] < 144:
        raise ValueError(f"The number of known pixels after removing must be at least 144 but is {len(np.where(known_array != 1)[0])}")

    return (input_array.astype(image_array.dtype), known_array.transpose((2,0,1)).astype(image_array.dtype), target_array.astype(image_array.dtype))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    R = np.ones((20,20))*0.2
    G = np.ones((20,20))*0.3
    B = np.ones((20,20))*0.5

    img = np.stack([R,G,B], axis=2)
    print(img.shape)
    

    i, k, t = ex4(img, (1,0), (2,3))
    print(t[-30:])
    print(k.shape)
    #plt.imshow(np.transpose(k, (1,2,0)))
    #plt.show()
    #plt.imshow(np.transpose(i, (1,2,0)))
    #plt.show()