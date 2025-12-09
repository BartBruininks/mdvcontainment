"""
Perlin-noise based creation of synthetic labeling data.
"""

# Python External
import numpy as np


def make_complex_3D(roll=0):
    """
    Returns and writes the atomgroup for a complex system(3D).
    """

    # Creating the boolean mask.
    shape = (23, 10 , 10)
    voxels = np.zeros(shape)
    
    # Add the cylinders
    voxels[3:7, 3:7, :] = 1
    voxels[4:6, 4:6, :] = 0

    # Add the nested islands
    voxels[15:21, 2:8, 2:8] = 1
    voxels[16:20, 3:7, 3:7] = 0
    voxels[17:19, 4:6, 4:6] = 1

    # Add dividing plane
    voxels[10:13, :, :] = 1

    # Add island in dividing plane
    voxels[11, 5, 5] = 0
    
    # Translate periodically 
    voxels = np.roll(voxels, roll, 0)
    voxels = np.roll(voxels, roll, 1)
    voxels = np.roll(voxels, roll, 1)
    return voxels.astype(bool)

def _generate_perlin_noise_3d(shape, res, tileable=(True, True, True), seed=0):
    """
    Returns correlated noise. The res should be all factors of the shape.
    e.g. shape = (10,10,10) --> res = (2, 1, 5)

    Copied from:
    https://github.com/pvigier/perlin-numpy/blob/master/perlin3d.py
    """
    def f(t):
        return 6*t**5 - 15*t**4 + 10*t**3

    np.random.seed(seed)
    delta = (res[0] / shape[0], res[1] / shape[1], res[2] / shape[2])
    d = (shape[0] // res[0], shape[1] // res[1], shape[2] // res[2])
    grid = np.mgrid[0:res[0]:delta[0],0:res[1]:delta[1],0:res[2]:delta[2]]
    grid = grid.transpose(1, 2, 3, 0) % 1
    # Gradients
    theta = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    phi = 2*np.pi*np.random.rand(res[0]+1, res[1]+1, res[2]+1)
    gradients = np.stack((np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)), axis=3)
    if tileable[0]:
        gradients[-1,:,:] = gradients[0,:,:]
    if tileable[1]:
        gradients[:,-1,:] = gradients[:,0,:]
    if tileable[2]:
        gradients[:,:,-1] = gradients[:,:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1).repeat(d[2], 2)
    g000 = gradients[    :-d[0],    :-d[1],    :-d[2]]
    g100 = gradients[d[0]:     ,    :-d[1],    :-d[2]]
    g010 = gradients[    :-d[0],d[1]:     ,    :-d[2]]
    g110 = gradients[d[0]:     ,d[1]:     ,    :-d[2]]
    g001 = gradients[    :-d[0],    :-d[1],d[2]:     ]
    g101 = gradients[d[0]:     ,    :-d[1],d[2]:     ]
    g011 = gradients[    :-d[0],d[1]:     ,d[2]:     ]
    g111 = gradients[d[0]:     ,d[1]:     ,d[2]:     ]
    # Ramps
    n000 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g000, 3)
    n100 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]  ), axis=3) * g100, 3)
    n010 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g010, 3)
    n110 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]  ), axis=3) * g110, 3)
    n001 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g001, 3)
    n101 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]  , grid[:,:,:,2]-1), axis=3) * g101, 3)
    n011 = np.sum(np.stack((grid[:,:,:,0]  , grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g011, 3)
    n111 = np.sum(np.stack((grid[:,:,:,0]-1, grid[:,:,:,1]-1, grid[:,:,:,2]-1), axis=3) * g111, 3)
    # Interpolation
    t = f(grid)
    n00 = n000*(1-t[:,:,:,0]) + t[:,:,:,0]*n100
    n10 = n010*(1-t[:,:,:,0]) + t[:,:,:,0]*n110
    n01 = n001*(1-t[:,:,:,0]) + t[:,:,:,0]*n101
    n11 = n011*(1-t[:,:,:,0]) + t[:,:,:,0]*n111
    n0 = (1-t[:,:,:,1])*n00 + t[:,:,:,1]*n10
    n1 = (1-t[:,:,:,1])*n01 + t[:,:,:,1]*n11
    return ((1-t[:,:,:,2])*n0 + t[:,:,:,2]*n1)

def create_3d_boolean_grid(shape, true_prob=0.5, res=[2,2,2], seed=0):
    """Create a 3D boolean grid with the given shape and probability of True values."""
    return (_generate_perlin_noise_3d(shape, res, seed=seed) + 0.5) > true_prob

