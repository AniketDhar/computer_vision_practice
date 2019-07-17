import numpy as np
from scipy.ndimage import gaussian_filter
from PIL import Image
import matplotlib.pyplot as plt


def compute_harris_response(im, sigma=3):
    '''Compute Harris response on the grayscale image'''

    # derivatives
    imx = np.zeros(im.shape)
    gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    
    imy = np.zeros(im.shape)
    gaussian_filter(im, (sigma, sigma), (1, 0), imy)

    # compute Harris matrix components

    Wxx = gaussian_filter(imx*imx, sigma)
    Wxy = gaussian_filter(imx*imy, sigma)
    Wyy = gaussian_filter(imy*imy, sigma)

    # determinant and trace
    Wdet = Wxx = Wyy - Wxy**2
    Wtr = Wxx + Wyy

    return Wdet / Wtr


def get_harris_points(harris_im, min_dist=10, threshold=0.1):
    '''Returns harris corners
    min-dist is min number of pixels separating corners and image boundary'''

    # find top corner candidates above a threshold
    corner_threshold = harris_im.max() * threshold
    harrisim_t = (harris_im > corner_threshold) * 1
    
    # get coordinates of candidates
    coords = np.array(harrisim_t.nonzero()).T
    # ...and their values
    candidate_values = [harrisim[c[0],c[1]] for c in coords]

    # sort candidates
    index = np.argsort(candidate_values)

    # store allowed point locations in array
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1
    
    # select the best points taking min_distance into account
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0

    return filtered_coords


def plot_harris_points(image, filtered_coords):
    """ Plots corners found in image. """
    
    plt.figure()
    plt.gray()
    plt.imshow(image)
    plt.plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
    plt.axis('off')
    plt.show()



im = np.array(Image.open('EmpireStateBuilding.jpg').convert('L'))
harrisim = compute_harris_response(im)
filtered_coords = get_harris_points(harrisim, 6)
plot_harris_points(im, filtered_coords)

