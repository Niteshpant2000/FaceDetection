import numpy as np

def integral(image):
    iiv=np.zeros(image.shape) # integral image value
    crs=np.zeros(image.shape) # cummulative row sum
    for row in range(len(image)):
        for col in range(len(image[row])):
            if(row-1>=0):
                crs[row][col]=crs[row-1][col]+image[row][col] 
            else:
                crs[row][col]=image[row][col]
            if(col-1>=0):
                iiv[row][col]=iiv[row][col-1]+crs[row][col]
            else:
                iiv[row][col]=crs[row][col]
    return iiv
