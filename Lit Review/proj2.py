"""
Project 2 Problem 2
A Demo code for V-Cycle
Edited by Han Chen, Sep 26, 2012
"""
from numpy import *
from pylab import *
import sys
import pdb

class JacobSolver:

    def __init__(self, N):
        '''
        Initialize Jacobian solver
        '''
        self._N = N
        self._dx = 1./(N -1)
        self._phi = zeros([N,N,N,N])
        self._f = None
        self._ns = None

    def NeighborSum(self):
        N = self._N
        tsum = self._phi[0:N-2,1:N-1,1:N-1,1:N-1] + self._phi[2:N-0,1:N-1,1:N-1,1:N-1]
        xsum = self._phi[1:N-1,0:N-2,1:N-1,1:N-1] + self._phi[1:N-1,2:N-0,1:N-1,1:N-1]
        ysum = self._phi[1:N-1,1:N-1,0:N-2,1:N-1] + self._phi[1:N-1,1:N-1,2:N-0,1:N-1]
        zsum = self._phi[1:N-1,1:N-1,1:N-1,0:N-2] + self._phi[1:N-1,1:N-1,1:N-1,2:N-0]
        self._ns = tsum + xsum + ysum + zsum


    def JacobUpdate(self):
        '''
        Jacobian iteration
        '''
        assert(self._f != None)
        N = self._N
        dx = self._dx
        self.NeighborSum()
        try:
            self._phi[1:N-1,1:N-1,1:N-1,1:N-1] = (self._ns + self._f * dx * dx) / 8.
        except:
            pdb.set_trace()


    def fullResidual(self):
        '''
        r = b - Ax
        '''
        assert( self._f!=None)
        self.NeighborSum()
        N, dx = self._N, self._dx
        res = - self._f - ( self._ns - 8.*self._phi[1:N-1,1:N-1,1:N-1,1:N-1] ) / (dx*dx)
        return res

    def restrictResidual(self):
        '''
        Restriction
        '''
        res = self.fullResidual()
        return res[1::2,1::2,1::2,1::2]


def interp(res):
    '''
    Interpolation
    '''
    csize = shape(res)[0]    # coarse size
    rsize = csize*2 - 1      # refined size
    newres = zeros([rsize,rsize,rsize,rsize])
    newres[::2 ,  ::2,  ::2,  ::2] = res
    newres[1::2,  ::2,  ::2,  ::2] = newres[:-1:2,   ::2,   ::2,   ::2]
    newres[  : , 1::2,  ::2,  ::2] = newres[  :  , :-1:2,   ::2,   ::2]
    newres[  : ,   : , 1::2,  ::2] = newres[  :  ,    : , :-1:2,   ::2]
    newres[  : ,   : ,   : , 1::2] = newres[  :  ,    : ,    : , :-1:2]
    return newres



def getSource(N):
    '''
    block defined by corners, generate source
    '''
    # label each block with index
    ind1D = linspace(0,1,N)
    ind4D = reshape( outer(ind1D, ones([N,N,N])), [N,N,N,N])
    indt = swapaxes(ind4D,0,0)
    indx = swapaxes(ind4D,0,1)
    indy = swapaxes(ind4D,0,2)
    indz = swapaxes(ind4D,0,3)
    # define one source block
    #                 t    x    y    z
    block = array([[ 0.,  0.,  0.,  0.],
                    [1., 1., 1., 1.]])
    # define source term
    t_check = double(indt >= block[0,0]) * double(indt <= block[1,0])
    x_check = double(indx >= block[0,1]) * double(indx <= block[1,1])
    y_check = double(indy >= block[0,2]) * double(indy <= block[1,2])
    z_check = double(indz >= block[0,3]) * double(indz <= block[1,3])
    f = t_check * x_check * y_check * z_check
    return f[1:N-1,1:N-1,1:N-1,1:N-1]


def MultiGrid(N, T, res=None):
    '''
    V-Cycle solver
    '''

    print "V downward %d" % N
    if N <= 3:
        return zeros([N,N,N])
    if res is None:
        res = getSource(N)

    solver0 = JacobSolver(N)
    solver0._f = res

    for i in range(T):
        solver0.JacobUpdate()

    corr_phi = MultiGrid((N+1)/2, T, solver0.restrictResidual())
    solver0._phi -= interp(corr_phi)

    print "V upward %d" % N
    for i in range(T):
        solver0.JacobUpdate()

    return solver0._phi



if __name__ == '__main__':
    N = 33
    T = 20
    phi = MultiGrid(N,T)
    contourf(phi[1,15])
    colorbar()
    show()


   
    







