from numpy import *
from pylab import *
from scipy.sparse import *


class LRAProblem(object):
	'''
	This class defines the 2D LRA neutron diffusion benchmark problem,
	including the material properties and geometry.
	'''

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
		self._dx = 15. / self._num_mesh_x
		self._dy = 15. / self._num_mesh_y

		# number of mesh cells in x,y dimension for entire LRA geometry
		self._num_x_cells = self._num_mesh_x * 11
		self._num_y_cells = self._num_mesh_y * 11

    	# Create a numpy array for materials ids
		self._material_ids = array([[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
									   [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
									   [3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5],
									   [3, 3, 3, 3, 3, 3, 3, 4, 5, 5, 5],
									   [2, 1, 1, 1, 1, 2, 2, 3, 3, 5, 5],
									   [2, 1, 1, 1, 1, 2, 2, 3, 3, 5, 5],
									   [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5],
									   [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5],
									   [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5],
									   [1, 1, 1, 1, 1, 1, 1, 3, 3, 5, 5],
									   [2, 1, 1, 1, 1, 2, 2, 3, 3, 5, 5]])

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

		# Geometric Buckling
		self._Bsquared = 1E-4

    	# Array with the material id for each fine mesh cell
		self._materials = zeros([self._num_x_cells, self._num_y_cells], float)
		for i in range(self._num_x_cells):
			for j in range(self._num_y_cells):
				self._materials[j,i] = self._material_ids[j  / \
										self._num_mesh_x][i / self._num_mesh_y]

		# Sparse destruction matrix used for solver (initialized empty)
		self.setupMFMatrices()


	def setupMFMatrices(self):
		'''
		Construct the destruction (M) and production (F) matrices
		in the neutron diffusion eigenvalue equation for the LRA 
		problem with a particular mesh size.
		'''

		# Create arrays for each of the diagonals of the 
		# production and destruction matrices
		size = 2 * self._num_x_cells * self._num_y_cells
		M_diag = zeros(size)
		M_udiag = zeros(size)
		M_2udiag = zeros(size)
		M_ldiag = zeros(size)
		M_2ldiag = zeros(size)
		M_3ldiag = zeros(size)
		F_diag1 = zeros(size)
		F_diag2 = zeros(size)
		
		# Loop over all cells in the mesh
		for i in range(size):

			# energy group 1
			if i < size / 2:
				x = i % self._num_x_cells
				y = i / self._num_y_cells
				mat_id = self._materials[y,x]

				# 2D lower - leakage from top cell
				if y > 0:
					M_2ldiag[i-self._num_x_cells] = -self._dx * \
								self.computeDCouple(self._D[mat_id][0], \
								self._D[self._materials[y-1,x]][0], self._dy)

				# lower - leakage from left cell
				if x > 0:
					M_ldiag[i-1] = -self._dy * \
								self.computeDCouple(self._D[mat_id][0], \
								self._D[self._materials[y,x-1]][0], self._dx)

				# 2D upper - leakage from bottom cell
				if y < self._num_y_cells-1:
					M_2udiag[i+self._num_x_cells] = -self._dx * \
								self.computeDCouple(self._D[mat_id][0], \
								self._D[self._materials[y+1,x]][0], self._dy)

				# upper - leakage from right cell
				if x < self._num_x_cells-1:
					M_udiag[i+1] = -self._dy * \
								self.computeDCouple(self._D[mat_id][0], \
								self._D[self._materials[y,x+1]][0], self._dx)

				# diagonal - absorption, downscattering, axial leakage
				M_diag[i] = (self._SigmaA[mat_id][0] + \
							self._D[mat_id][0] * self._Bsquared + \
                            self._SigmaS21[mat_id]) * self._dx * self._dy

				# leakage into cell above
				if y > 0:
					M_diag[i] += self._dx * \
								self.computeDCouple(self._D[mat_id][0], \
								self._D[self._materials[y-1,x]][0], self._dy)

				# leakage into cell below
				if y < self._num_y_cells-1:
					M_diag[i] += self._dx * \
								self.computeDCouple(self._D[mat_id][0], \
								self._D[self._materials[y+1,x]][0], self._dy)

				# leakage into cell to the left
				if x > 0:
					M_diag[i] += self._dy * \
								self.computeDCouple(self._D[mat_id][0], \
								self._D[self._materials[y,x-1]][0], self._dy)

				# leakage into cell to the right
				if x < self._num_x_cells-1:
					M_diag[i] += self._dy * \
								self.computeDCouple(self._D[mat_id][0], \
								self._D[self._materials[y,x+1]][0], self._dy)

				# leakage into vacuum for cells at top edge of the geometry
				if (y == 0):
					M_diag[i] += (2.0 * self._D[mat_id][0] / self._dx) * \
						(1.0 / (1.0 + (4.0 *  self._D[mat_id][0] / self._dx)))

				# leakage into vacuum for cells at right edge of the geometry
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
				M_3ldiag[i-size/2] = -self._SigmaS21[mat_id] * \
												self._dx * self._dy

				# 2self._D lower - leakage from top cell
				if y > 0:
					M_2ldiag[i-self._num_x_cells] = -self._dx * \
								self.computeDCouple(self._D[mat_id][1], \
								self._D[self._materials[y-1,x]][1], self._dy)

				# lower - leakage from left cell
				if x > 0:
					M_ldiag[i-1] = -self._dy * \
								self.computeDCouple(self._D[mat_id][1], \
								self._D[self._materials[y,x-1]][1], self._dx)

				# 2self._D upper - leakage from bottom cell
				if y < self._num_y_cells-1:
					M_2udiag[i+self._num_x_cells] = -self._dx * \
								self.computeDCouple(self._D[mat_id][1], \
								self._D[self._materials[y+1,x]][1], self._dy)

				# upper - leakage from right cell
				if x < self._num_x_cells-1:
					M_udiag[i+1] = -self._dy * \
								self.computeDCouple(self._D[mat_id][1], \
								self._D[self._materials[y,x+1]][1], self._dx)

				# diagonal - absorption, downscattering, axial leakage
				M_diag[i] = (self._SigmaA[mat_id][1]  + \
							self._D[mat_id][1] * self._Bsquared) * \
											self._dx * self._dy

				# leakage into cell above
				if y > 0:
					M_diag[i] += self._dx * \
								self.computeDCouple(self._D[mat_id][1], \
								self._D[self._materials[y-1,x]][1], self._dy)

				# leakage into cell below
				if y < self._num_y_cells-1:
					M_diag[i] += self._dx * \
								self.computeDCouple(self._D[mat_id][1], \
								self._D[self._materials[y+1,x]][1], self._dy)

				# leakage into cell to the left
				if x > 0:
					M_diag[i] += self._dy * \
								self.computeDCouple(self._D[mat_id][1], \
								self._D[self._materials[y,x-1]][1], self._dy)

				# leakage into cell to the right
				if x < self._num_x_cells-1:
					M_diag[i] += self._dy * \
								self.computeDCouple(self._D[mat_id][1], \
								self._D[self._materials[y,x+1]][1], self._dy)

				# leakage into vacuum for cells at top edge of the geometry
				if (y == 0):
					M_diag[i] += (2.0 * self._D[mat_id][0] / self._dx) * \
						(1.0 / (1.0 + (4.0 *  self._D[mat_id][1] / self._dx)))

				# leakage into vacuum for cells at right edge of the geometry
				if (x == self._num_x_cells-1):
					M_diag[i] += (2.0 * self._D[mat_id][0] / self._dy) * \
						(1.0 / (1.0 + (4.0 *  self._D[mat_id][1] / self._dy)))


		# Construct sparse diagonal matrices
		self._M = dia_matrix(([M_diag, M_udiag, M_2udiag, M_ldiag, M_2ldiag, 
							M_3ldiag], [0, 1, self._num_x_cells, -1, \
							-self._num_x_cells, -size/2]), shape=(size, size))
		self._F = dia_matrix(([F_diag1, F_diag2], [0, size/2]), \
												shape=(size, size))


	def plotMaterials(self):
		'''
		Plot a coarse 2D grid of the materials in the LRA problem
		'''	

		# Correct Python's array layout for display		
		materials = flipud(self._material_ids)

		figure()
		pcolor(linspace(self._xmin, self._xmax, 12), \
				linspace(self._ymin, self._ymax, 12), \
				materials, edgecolors='k', linewidths=1)
		axis([0, 165, 0, 165])
		title('2D LRA Benchmark Materials')
		show()



	def plotMesh(self):
		'''
		Plot the fine 2D mesh used to solve the LRA problem
		'''	

		materials = flipud(self._materials)
		figure()
		pcolor(linspace(0, 165, self._num_x_cells), \
				linspace(0, 165, self._num_y_cells), \
				materials, edgecolors='k', linewidths=0.25)
		axis([0, 165, 0, 165])
		title('2D LRA Benchmark Mesh')
		show()


	def spyMF(self):
		'''
		Display nonzeros for production (M) and destruction matrices (F)
		'''

		fig = figure()
		fig.add_subplot(121)		
		spy(self._M)
		fig.add_subplot(122)		
		fig = figure()
		spy(self._F)
		show()


	def computeDCouple(self, D1, D2, delta):
		'''
		Compute the diffusion coefficient coupling two adjacent cells
		'''
		return (2.0 * D1 * D1) / (delta * (D1 + D2))


	def computeF(self, x):
		'''
		Compute the residual vector for JFNK
		'''

		m = x.size
		phi = x[:m-1]
		lamb = x[m-1][0]

		# Allocate space for F
		F = ones((m, 1))

		print 'lamb = ' + str(lamb)
	
		# M - lambda * F * phi constraint
		F[:m-1] = self._M * phi - lamb * self._F * phi

		# Flux normalization constraint
		F[m-1] = -0.5 * vdot(phi, phi) + 0.5

		return F


	def plotPhi(self, phi):
		'''
		Generate a 2D color plot of the fast and thermal flux from a 1D ndarray
		'''

		# Plot the thermal and fast flux and convergence rate
		fig = figure()
		phi_g1 = reshape(phi[0:self._num_y_cells * self._num_x_cells], \
							(-self._num_y_cells, self._num_y_cells), order='A')
		phi_g2 = reshape(phi[self._num_y_cells * self._num_x_cells:], \
							(-self._num_y_cells, self._num_y_cells), order='A')
		phi_g1 = flipud(phi_g1)
		phi_g2 = flipud(phi_g2)
	
		fig.add_subplot(121)
		pcolor(linspace(0, 165, self._num_x_cells), \
								linspace(0, 165, self._num_y_cells), phi_g1)
		colorbar()	
		axis([0, 165, 0, 165])
		title('Group 1 (Fast) Flux')
	
		fig.add_subplot(122)
		pcolor(linspace(0, 165, self._num_x_cells), \
								linspace(0, 165, self._num_y_cells), phi_g2)
		colorbar()
		axis([0, 165, 0, 165])
		title('Group 2 (Thermal) Flux')

		show()


	def computeAnalyticJacobian(self, x):

		m = x.shape[0]
		phi = x[:m-1]
		lamb = x[m-1][0]

		J = lil_matrix((m,m))

		# Construct temporary blocks for Jacobian
		a = self._M - lamb * self._F
		b = -phi.T
		c = -self._F * phi
		c = vstack([c, zeros(1)])

		# Build Jacobian using scipy's sparse matrix stacking operators
		J = vstack([a, b])
		J = hstack([J, c])

		return J


	def computeAnalyticJacobVecProd(self, x, y):

		m = x.shape[0]
		phi = x[:m-1]
		lamb = x[m-1]

		print 'lamb = ' + str(lamb) + '	lamb[0] = ' + str(lamb[0])

#		J = lil_matrix((m,m))

		# Construct temporary blocks for Jacobian
		a = self._M - lamb[0] * self._F
		b = phi.T
		c = -self._F * phi
		c = vstack([c, zeros(1)])

		# Build Jacobian using scipy's sparse matrix stacking operators
		J = vstack([a, b])
		J = hstack([J, c])

		return J * y


	def computeFDJacobVecProd(self, x, y):

		phi = x[:x.size-2]
		lamb = x[x.size-1]		
		y = resize(y, [y.size, 1])

		b = 1E-8

		epsilon = b * sum(x) / (y.size * norm(y))
		
		# Approximate Jacobian matrix vector multiplication
		tmp1 = self.computeF(x + (epsilon * y))
#		print 'compute tmp1'
		tmp2 = self.computeF(x)
#		print 'compute tmp2'
		Jy = (tmp1 - tmp2) / epsilon

		return Jy
