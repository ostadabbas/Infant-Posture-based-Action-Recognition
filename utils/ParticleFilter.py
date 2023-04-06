from filterpy.monte_carlo import systematic_resample
import numpy as np
from numpy.linalg import norm
from numpy.random import randn
import scipy.stats
from numpy.random import uniform
import pickle
import copy
from math import *


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


##################################################################################################
def predict(particles, u, std):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity)`"""

    N = len(particles)
    # update heading
    # move in the (noisy) commanded direction
    dist = (u[1]) + (randn(N) * std[1])
    particles[:, 0] += np.cos(u[0]) * dist
    particles[:, 1] += np.sin(u[0]) * dist


##################################################################################################
def update(particles, weights, z, R, landmarks):
    for i, landmark in enumerate(landmarks):
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300      # avoid round-off to zero
    weights /= sum(weights) # normalize

##################################################################################################    
def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var

##################################################################################################
def simple_resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1. # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

##################################################################################################
def neff(weights):
    return 1. / np.sum(np.square(weights))

##################################################################################################
def resample_from_index(particles, weights, indexes):
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights.fill(1.0 / len(weights))

##################################################################################################
def read_pickle(keyps, f, j):    
        x0=keyps[f][0][j]
        y0=keyps[f][1][j]
        x1=keyps[f+1][0][j]
        y1=keyps[f+1][1][j]
        
        cord = np.array([x0,y0])
        orientation = atan2((y1 - y0),(x1 - x0))
        dist= sqrt((x1-x0) ** 2 + (y1-y0) ** 2)
        u = np.array([orientation,dist])
        return (cord, u)
            
##################################################################################################
##################################################################################################
def particle_filter(keyps, j, initial_x, h, w, N=300,sensor_std_err=0.3,xlim=(-356, 356), ylim=(-356, 356)):
    landmarks = np.array([[-10, -10], [-10, w+10], [h+10,-10], [h+10,w+10]])
    NL = len(landmarks)   
    # create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(
            mean=initial_x, std=(5, 5, np.pi/4), N=N)
    weights = np.ones(N) / N    
    kp = copy.deepcopy(keyps)
    for x in range(len(keyps)-1):
        keypt_pos, uv = read_pickle(keyps,x,j)
        zs = (norm(landmarks - keypt_pos, axis=1) + 
              (randn(NL) * sensor_std_err))
        if uv[1] ==0:
            uv[1]==2
        # move diagonally forward to (x+1, x+1)
        predict(particles, u=uv, std=(0, 0))

        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err, 
               landmarks=landmarks)
        # resample if too few effective particles
        if neff(weights) < N/2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            #assert np.allclose(weights, 1/N)
        mu, var = estimate(particles, weights) 
        kp[x,:2,j] = mu
    return kp
