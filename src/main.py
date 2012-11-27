import numpy as np
from pylab import *
from scipy.sparse import lil_matrix, dia_matrix
from scipy.sparse.linalg import spsolve
from os import *
from sys import *
import getopt



# INDEXING IS IN Y,X ORDER!!!!!



class LRAProblem:

	def __init__(self, num_mesh_x=5, num_mesh_y=5):
		'''
		Initialize LRA geometry and materials
		'''

		# Define the boundaries of the geometry [cm]		
		self._xmin = 0.;
		self._xmax = 165.;
		self._ymin = 0.;
		self._ymax = 165.;

    	# number of mesh per coarse grid cell in LRA problem 
		self._num_mesh_x = num_mesh_x
		self._num_mesh_y = num_mesh_y

		# mesh spacing - each LRA coarse grid cell is 5cm x 5cm
		self._dx = 5. / self._num_mesh_x
		self._dy = 5. / self._num_mesh_y

		# number of mesh cells in x,y dimension for entire LRA geometry
		self._num_x_cells = self._num_mesh_x * 11
		self._num_y_cells = self._num_mesh_y * 11

    	# Create a numpy array for materials ids
		self._material_ids = np.array([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
									   [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
									   [3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
									   [3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5],
									   [2, 5, 5, 5, 5, 2, 2, 3, 3, 5, 5],
									   [2, 5, 5, 5, 5, 2, 2, 3, 3, 5, 5],
									   [5, 5, 5, 5, 5, 5, 5, 3, 3, 5, 5],
									   [5, 5, 5, 5, 5, 5, 5, 3, 3, 5, 5],
									   [5, 5, 5, 5, 5, 5, 5, 3, 3, 5, 5],
									   [5, 5, 5, 5, 5, 5, 5, 3, 3, 5, 5],
									   [2, 5, 5, 5, 5, 2, 2, 3, 3, 5, 5]])

    	# Dictionary with keys (material ids) to diffusion coefficients
		self._D = {1: [1.255, 0.211], 
         		   2: [1.268, 0.1902],
         		   3: [1.259, 0.2091],
         		   4: [1.259, 0.2091],
         		   5: [1.257, 0.1592]}

    	# Dictionary with keys (material ids) to absorption cross-sections
		self._SigmaA = {1: [0.008252, 0.1003], 
                        2: [0.007181, 0.07047],
              			3: [0.008002, 0.08344],
              			4: [0.008002, 0.073324],
              			5: [0.0006034, 0.01911]}

    	# Dictionary with keys (material ids) to fission cross-sections
		self._NuSigmaF = {1: [0.004602, 0.1091], 
                		  2: [0.004609, 0.08675],
                		  3: [0.004663, 0.1021],
                		  4: [0.004663, 0.1021],
               		 	  5: [0., 0.]}

    	# Dictionary with keys (material ids) to group 2 to 1 scattering           
    	# cross-sections
		self._SigmaS21 = {1: 0.02533, 
                		  2: 0.02767,
                		  3: 0.02617,
                		  4: 0.02617,
                		  5: 0.04754}

    	# Array with the material id for each fine mesh cell
		self._materials = zeros([self._num_x_cells, self._num_y_cells], float)
		for i in range(self._num_x_cells):
			for j in range(self._num_y_cells):
				self._materials[j,i] = self._material_ids[j  / self._num_mesh_x][i / self._num_mesh_y]

		# Sparse destruction matrix used for solver (initialized empty)
		self._M = lil_matrix((10, 10))
		self._F = lil_matrix((10, 10))


	def plotMaterials(self):
		'''
		Plot a coarse 2D grid of the materials in the LRA problem
		'''	
		
		figure()
		pcolor(linspace(self._xmin, self._xmax, 12), linspace(self._ymin, self._ymax, 12), \
				self._material_ids.T, edgecolors='k', linewidths=1)
		axis([0, 165, 0, 165])
		title('2D LRA Benchmark Materials')
		show()


	def plotMesh(self):
		'''
		Plot the fine 2D mesh used to solve the LRA problem
		'''	

		figure()
		pcolor(linspace(0, 165, self._num_x_cells), linspace(0, 165, self._num_y_cells), \
				self._materials.T, edgecolors='k', linewidths=0.25)
		axis([0, 165, 0, 165])
		title('2D LRA Benchmark Mesh')
		show()


	def spyMF(self):
		fig = figure()
		fig.add_subplot(121)		
		spy(self._M)
		fig.add_subplot(122)		
		spy(self._F)


	def computeDCouple(self, D1, D2, delta):
		'''
		Compute the diffusion coefficient coupling two adjacent cells
		'''
		return (2.0 * D1 * D1) / (delta * (D1 + D2))


	def setupMFMatrices(self):
		'''
		'''

		# Create arrays for each of the diagonals of the production and destruction matrices
		size = 2 * self._num_x_cells * self._num_y_cells
		M_diag = np.zeros(size)
		M_udiag = np.zeros(size)
		M_2udiag = np.zeros(size)
		M_ldiag = np.zeros(size)
		M_2ldiag = np.zeros(size)
		M_3ldiag = np.zeros(size)
		F_diag1 = np.zeros(size)
		F_diag2 = np.zeros(size)
		

		for i in range(size):

			# energy group 1
			if i < size / 2:
				x = i % self._num_x_cells
				y = i / self._num_y_cells
				mat_id = self._materials[y,x]

				# 2D lower - leakage from top cell
				if y > 0:
					M_2ldiag[i-self._num_x_cells] = -self._dx * self.computeDCouple(self._D[mat_id][0], \
												self._D[self._materials[y-1,x]][0], self._dy)

				# lower - leakage from left cell
				if x > 0:
					M_ldiag[i-1] = -self._dy * self.computeDCouple(self._D[mat_id][0], \
												self._D[self._materials[y,x-1]][0], self._dx)

				# 2D upper - leakage from bottom cell
				if y < self._num_y_cells-1:
					M_2udiag[i+self._num_x_cells] = -self._dx * self.computeDCouple(self._D[mat_id][0], \
												self._D[self._materials[y+1,x]][0], self._dy)

				# upper - leakage from right cell
				if x < self._num_x_cells-1:
					M_udiag[i+1] = -self._dy * self.computeDCouple(self._D[mat_id][0], \
												self._D[self._materials[y,x+1]][0], self._dx)

				# diagonal
				M_diag[i] = (self._SigmaA[mat_id][0] + \
                                self._SigmaS21[mat_id]) * self._dx * self._dy


				# leakage into cell above
				if y > 0:
					M_diag[i] += self._dx * self.computeDCouple(self._D[mat_id][0], \
												self._D[self._materials[y-1,x]][0], self._dy)

				# leakage into cell below
				if y < self._num_y_cells-1:
					M_diag[i] += self._dx * self.computeDCouple(self._D[mat_id][0], \
												self._D[self._materials[y+1,x]][0], self._dy)

				# leakage into cell to the left
				if x > 0:
					M_diag[i] += self._dy * self.computeDCouple(self._D[mat_id][0], \
												self._D[self._materials[y,x-1]][0], self._dy)

				# leakage into cell to the right
				if x < self._num_x_cells-1:
					M_diag[i] += self._dy * self.computeDCouple(self._D[mat_id][0], \
												self._D[self._materials[y,x+1]][0], self._dy)

				# leakage into vacuum for cells at the topmost edge of the geometry
				if (y == 0):
					M_diag[i] += (2.0 * self._D[mat_id][0] / self._dx) * \
									(1.0 / (1.0 + (4.0 *  self._D[mat_id][0] / self._dx)))

				# leakage into vacuum for cells at the rightmost edge of the geometry
				if (x == self._num_x_cells-1):
					M_diag[i] += (2.0 * self._D[mat_id][0] / self._dy) * \
									(1.0 / (1.0 + (4.0 *  self._D[mat_id][0] / self._dx)))

				# fission production
				F_diag1[i] = self._NuSigmaF[mat_id][0]

				# fission production
				F_diag2[i+size/2] = self._NuSigmaF[mat_id][1]

			# energy group 2
			else:
				x = (i-(size/2)) % self._num_x_cells
				y = (i-(size/2)) / self._num_y_cells
				mat_id = self._materials[y,x]

				# Group 1 scattering into group 2
				M_3ldiag[i-size/2] = -self._SigmaS21[mat_id] * self._dx * self._dy

				# 2self._D lower - leakage from top cell
				if y > 0:
					M_2ldiag[i-self._num_x_cells] = -self._dx * self.computeDCouple(self._D[mat_id][1], \
												self._D[self._materials[y-1,x]][1], self._dy)

				# lower - leakage from left cell
				if x > 0:
					M_ldiag[i-1] = -self._dy * self.computeDCouple(self._D[mat_id][1], \
												self._D[self._materials[y,x-1]][1], self._dx)

				# 2self._D upper - leakage from bottom cell
				if y < self._num_y_cells-1:
					M_2udiag[i+self._num_x_cells] = -self._dx * self.computeDCouple(self._D[mat_id][1], \
												self._D[self._materials[y+1,x]][1], self._dy)

				# upper - leakage from right cell
				if x < self._num_x_cells-1:
					M_udiag[i+1] = -self._dy * self.computeDCouple(self._D[mat_id][1], \
												self._D[self._materials[y,x+1]][1], self._dx)

				# diagonal
				M_diag[i] = self._SigmaA[mat_id][1] * self._dx * self._dy

				# leakage into cell above
				if y > 0:
					M_diag[i] += self._dx * self.computeDCouple(self._D[mat_id][1], \
												self._D[self._materials[y-1,x]][1], self._dy)

				# leakage into cell below
				if y < self._num_y_cells-1:
					M_diag[i] += self._dx * self.computeDCouple(self._D[mat_id][1], \
												self._D[self._materials[y+1,x]][1], self._dy)

				# leakage into cell to the left
				if x > 0:
					M_diag[i] += self._dy * self.computeDCouple(self._D[mat_id][1], \
												self._D[self._materials[y,x-1]][1], self._dy)

				# leakage into cell to the right
				if x < self._num_x_cells-1:
					M_diag[i] += self._dy * self.computeDCouple(self._D[mat_id][1], \
												self._D[self._materials[y,x+1]][1], self._dy)

				# leakage into vacuum for cells at the topmost edge of the geometry
				if (y == 0):
					M_diag[i] += (2.0 * self._D[mat_id][0] / self._dx) * \
									(1.0 / (1.0 + (4.0 *  self._D[mat_id][1] / self._dx)))

				# leakage into vacuum for cells at the rightmost edge of the geometry
				if (x == self._num_x_cells-1):
					M_diag[i] += (2.0 * self._D[mat_id][0] / self._dy) * \
									(1.0 / (1.0 + (4.0 *  self._D[mat_id][1] / self._dy)))


		# Construct sparse diagonal matrices
		self._M = dia_matrix(([M_diag, M_udiag, M_2udiag, M_ldiag, M_2ldiag, M_3ldiag], \
								[0, 1, self._num_x_cells, -1, -self._num_x_cells, -size/2]), \
								shape=(size, size))
		self._F = dia_matrix(([F_diag1, F_diag2], [0, size/2]), shape=(size, size))


def powerIteration(M, F, max_iters, tol):

	# Guess initial keff
	keff = 1.0

	# Guess initial flux and normalize it
	phi = np.ones((M.shape[1], 1))
	phi = phi / norm(phi)

	# Array for phi_res and keff+res
	phi_res = []
	keff_res = []

	for i in range(max_iters):

		# Update flux vector
		phi_new = spsolve(M.tocsr(), ((1. / keff) * F * phi))

		# Normalize new flux
		phi_new = phi_new / norm(phi_new)
	
		# Update keff
		source_new = F * phi_new
		source_old = F * phi

		keff_new = keff * np.vdot(source_new, source_new) / np.vdot(source_old, source_old)
			
		# Compute residuals
		phi_res.append(norm(phi_new - phi))
		keff_res.append(abs(keff_new - keff) / keff)

		print 'Power iteration: i = ' + str(i) + '  phi_res = ' + \
			            str(phi_res[i-1]) + '  keff_res = ' + str(keff_res[i-1]) \

		# Check convergence
		if phi_res[i] < tol and keff_res[i] < tol:
			break
		else:
			phi = phi_new
			if (i > 5):
				keff = keff_new

	print 'Converged to keff = ' + str(keff) + ' (tol = ' + str(tolerance) + \
								') in ' + str(i) + ' iterations.'

	# Plot the thermal and fast flux and convergence rate
	fig = figure()
	phi_g1 = np.reshape(phi[0:LRA._num_y_cells * LRA._num_x_cells], (-LRA._num_y_cells, LRA._num_y_cells), order='A')
	phi_g2 = np.reshape(phi[LRA._num_y_cells * LRA._num_x_cells:], (-LRA._num_y_cells, LRA._num_y_cells), order='A')
	phi_g1 = np.flipud(phi_g1)
	phi_g2 = np.flipud(phi_g2)
	
	fig.add_subplot(221)
	pcolor(linspace(0, 165, LRA._num_x_cells), linspace(0, 165, LRA._num_y_cells), phi_g1)
	axis([0, 165, 0, 165])
	title('Group 1 (Fast) Flux')
	
	fig.add_subplot(222)
	pcolor(linspace(0, 165, LRA._num_x_cells), linspace(0, 165, LRA._num_y_cells), phi_g2)
	axis([0, 165, 0, 165])
	title('Group 2 (Thermal) Flux')

	fig.add_subplot(223)
	plot(np.array(range(i+1)), np.array(phi_res))
	plot(np.array(range(i+1)), np.array(keff_res))
	legend(['flux', 'keff'])
	yscale('log')
	xlabel('iteration #')
	ylabel('residual')
	title('Residuals')

	show()

	return phi



def givens(H, c, s, g, n):

	# Apply all previous rotations to new column	
	for i in range(n):
	
		# Apply rotation to row i
		tmp1 = c[i] * H[i,n] - s[i] * H[i+1,n]
		tmp2 = s[i] * H[i,n] + c[i] * H[i+1,n]

		# Transfer computed values back to matrix
		H[i,n] = tmp1
		H[i+1,n] = tmp2

	# Compute cos and sin for new rotation
	r = sqrt(H[n,n]**2 + H[n+1,n]**2)
	c[n] = H[n,n] / r
	s[n] = -H[n+1,n] / r

	# Apply new rotation to new column
	tmp1 = c[n] * H[n,n] - s[n] * H[n+1,n]
	tmp2 = s[n] * H[n,n] + c[n] * H[n+1,n]

	# Transfer computed values back to matrix
	H[n,n] = tmp1
	H[n+1,n] = tmp2

	# Zero out sub-diagonal (to correct for roundoff)
	H[n+1,n] = 0.0

	# Apply new rotation to g
	tmp1 = c[n] * g[n] - s[n] * g[n+1]
	tmp2 = s[n] * g[n] + c[n] * g[n+1]

	g[n] = tmp1
	g[n+1] = tmp2

	return




def gmres(M, F, phi, keff, res, outer_iter, tol):
	
	# Find the size of b
	m = phi.size

	Q = np.zeros((m, res+1),dtype=np.float)
	H = np.zeros((res+1, res), dtype=np.float)
	err = np.ones((outer_iter*res, 1), dtype=np.float)
	c = np.zeros((res+1, 1), dtype=np.float)
	s = np.zeros((res+1, 1), dtype=np.float)

	for k in range(outer_iter):
		
		print 'phi.shape = ' + str(phi.shape)

		b = (1. / keff) * F * phi
		r = b - M * phi
		beta = np.linalg.norm(r)

		g = np.zeros((res+1, 1))
		g[0] = beta

		print 'beta = ' + str(beta)

		# Check for convergence
		if beta < tol:
			break

		# Compute first Q
		Q[:,0] = r.T / beta

		# GMRES	on the residual error
		for n in range(res):

			print 'n = ' + str(n)			

			# Arnoldi
			y = Q[:,n]
			v = M * y
			normv1 = np.linalg.norm(v)

			# Loop over previous vectors
			for j in range(n):
				# Orthogonal projection of A onto new Krylov subspace
				# H = Q'AQ
				H[j,n] = np.vdot(Q[:,j], v)
				
				# Equation 33.4 in Trefethen
				v = v - H[j,n] * Q[:,j]

			# Compute new H
			H[n+1,n] = norm(v)
			normv2 = H[n+1,n]

			# May need to reorthogonalize here

			# Compute next column of Q
			Q[:,n+1] = v.T / H[n+1,n]

			# Apply Givens rotation to new column of H
			givens(H, c, s, g, n)

			# Compute the error for this outer/inner iteration pair
			err[n+res*(k-1)] = abs(g[n+1])
			
			# Check for convergence
			if (err[n+res*(k-1)] < tol):
				print 'GMRES converged to ' + str(tol)
				break

		# Compute y
		y = np.linalg.solve(H[:n,:n], g[:n])

		print 'phi.shape = ' + str(phi.shape)

		# Compute new phi
		print 'type(Q) = ' + str(type(Q))
		print 'dot(Q[:,:n], y).shape = ' + str(dot(Q[:,:n], y).shape)
		phi = phi + dot(Q[:,:n], y)
		phi = phi / np.linalg.norm(phi)

		print 'phi.shape = ' + str(phi.shape)
		
		# Check convergence and break outer loop
		if err[k*n-1] < tol:
			print 'GMRES converged to ' + str(tol)
			break

		if k == outer_iter-1:
			print 'GMRES did not converge'

	return phi




if __name__ == '__main__':

    # Parse command line options
	opts, args = getopt.getopt(sys.argv[1:], "t:n:psn", \
									["num_mesh", "plot_mesh", "spy", "tolerance"])

	# Default arguments
	num_mesh = 5
	tolerance = 1E-3
	plot_mesh = False
	spyMF = False

	for o, a in opts:
		if o in ("-n", "--num_mesh"):
			num_mesh = min(int(a), 8)
#			num_mesh = int(a)
		elif o in ("-t", "--tolerance"):
			tolerance = float(a)
		elif o in ("-p", "--plot_mesh"):
			plot_mesh = True
		elif o in ("-s", "--spy"):
			spyMF = True
		else:
			assert False, "unhandled option"


	LRA = LRAProblem(num_mesh, num_mesh)
	LRA.setupMFMatrices()

	if plot_mesh:
		LRA.plotMesh()
	if spyMF:
		LRA.spyMF()

	phi = powerIteration(LRA._M, LRA._F, 200, tolerance)
#	phi = powerIteration(LRA._M, LRA._F, 3, tolerance)
#	phi = np.reshape(phi, (phi.size, 1))


	# Guess initial keff
	keff = 1.0

	# Guess initial flux and normalize it
#	phi = np.ones((LRA._M.shape[1], 1), dtype=np.float)
#	phi = phi / np.linalg.norm(phi)

'''
	res = 5
	outer_iter = 1000

	phi = gmres(LRA._M, LRA._F, phi, keff, res, outer_iter, tolerance)

	fig = figure()
	phi_g1 = np.reshape(phi[0:LRA._num_y_cells * LRA._num_x_cells], (-LRA._num_y_cells, LRA._num_y_cells), order='A')
	phi_g2 = np.reshape(phi[LRA._num_y_cells * LRA._num_x_cells:], (-LRA._num_y_cells, LRA._num_y_cells), order='A')
	phi_g1 = np.flipud(phi_g1)
	phi_g2 = np.flipud(phi_g2)
	
	fig.add_subplot(221)
	pcolor(linspace(0, 165, LRA._num_x_cells), linspace(0, 165, LRA._num_y_cells), phi_g1)
	axis([0, 165, 0, 165])
	title('Group 1 (Fast) Flux')
	
	fig.add_subplot(222)
	pcolor(linspace(0, 165, LRA._num_x_cells), linspace(0, 165, LRA._num_y_cells), phi_g2)
	axis([0, 165, 0, 165])
	title('Group 2 (Thermal) Flux')

	show()
'''

