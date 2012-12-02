import numpy as np
from pylab import *
from scipy.sparse import lil_matrix, dia_matrix, hstack, vstack
from scipy.sparse.linalg import gmres, spilu, LinearOperator
from os import *
from sys import *
import getopt

import LRAProblem as LRA
	

class Solver(object):

	def __init__(self):
		
		# Full weighting stencils for multigrid
		self._R_interior = np.array([[0.25, 0.5, 0.25], \
									 [0.5, 1.0, 0.5], \
									 [0.25, 0.5, 0.25]])
		self._R_interior *= (1 / 4.)

		self._R_tl = np.array([[1., 0.5], \
							   [0.5, 0.25]])
		self._R_tl *= (1. / 2.25)

		self._R_tr = np.array([[0.5, 1.], \
							   [0.25, .5]])
		self._R_tr *= (1. / 2.25)

		self._R_bl = np.array([[0.5, 0.25], \
							   [1., 0.5]])
		self._R_bl *= (1. / 2.25)

		self._R_br = np.array([[0.25, 0.5], \
							   [0.5, 1.]])
		self._R_br *= (1. / 2.25)

		self._R_top = np.array([[0.5, 0.25], \
								[1., 0.5], \
								[0.5, 0.25]])
		self._R_top *= (1. / 3.)		

		self._R_bottom = np.array([[0.5, 0.25], \
								   [0.5, 1.], \
								   [0.25, 0.5]])
		self._R_bottom *= (1. / 3.)

		

	def powerIteration(self, M, F, max_iters, tol, precond=False):
		'''
		Perform power iteration for the keff eigenvalue problem.
		Converges the source to a specified tolerance and plots the
		fast and thermal flux and the residual at each iteration
		'''

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

			# Preconditioned GMRES
			if precond:
				P = spilu(M.tocsc(), drop_tol=1e-5)
				D_x = lambda x: P.solve(x)
				D = LinearOperator((M.shape[0], M.shape[0]), D_x)
				phi_new = gmres(M.tocsr(), ((1. / keff) * F * phi), M=D)
			# Unpreconditioned GMRES
			else:
				phi_new = gmres(M.tocsr(), ((1. / keff) * F * phi))

			phi_new = phi_new[0][:]

			# Normalize new flux
			phi_new = phi_new / norm(phi_new)
	
			# Update keff
			source_new = F * phi_new
			source_old = F * phi

			tot_source_new = sum(source_new)
			tot_source_old = sum(source_old)

			keff_new = tot_source_new / tot_source_old

			# Compute residuals
			phi_res.append(norm(source_old - source_new))
			keff_res.append(abs(keff_new - keff) / keff)

			print 'Power iteration: i = %d	phi_res = %1E	keff_res = %1E' \
					% (i, phi_res[i-1], keff_res[i-1])

			# Check convergence
			if phi_res[i] < tol and keff_res[i] < tol:
				print 'Power iteration converged in %1d iters with keff = %1.5f' \
																		% (i, keff)
				break
			else:
				phi = phi_new
				if (i > 1):
					keff = keff_new

		# Plot the residuals of phi and keff at each iteration
		fig = figure()
		plot(np.array(range(i+1)), np.array(phi_res))
		plot(np.array(range(i+1)), np.array(keff_res))
		legend(['flux', 'keff'])
		yscale('log')
		xlabel('iteration #')
		ylabel('residual')
		title('Power Iteration Residuals')

		return phi


	def restrict(self, x):
		'''
		This restriction operator requires the number of mesh to be odd
		'''

		x = np.asarray(x)

		mesh_size = int(sqrt(x.size))
		new_mesh_size = int(ceil(mesh_size / 2.))

		print 'x.size = %d, mesh_size = %d, new_mesh_size = %d' % (x.size, mesh_size, new_mesh_size)
		
		# Reshape x into 2D arrays corresponding to the geometric mesh
		x = np.reshape(x, [mesh_size, mesh_size])

		# Allocate memory for the restricted x 
		x_new = np.zeros((new_mesh_size, new_mesh_size))

		# Restrict the x and b vectors
		for i in range(new_mesh_size):
			for j in range(new_mesh_size):
				print 'i = %d, j = %d' % (i,j)
				# top left corner
				if (i is 0 and j is 0):
					x_new[i,j] = sum(np.dot(self._R_tl, x[i*2:i*2+2,j*2:j*2+2]))
				# top right corner
				elif (i is 0 and j is new_mesh_size-1):
					x_new[i,j] = sum(np.dot(self._R_tr, x[i*2:i*2+2, j*2:j*2+1]))
				# top row but not a corner
				elif (i is 0):
					x_new[i,j] = sum(np.dot(self._R_top, x[i*2:i*2+2, j*2-1:j*2+2]))
				# bottom left corner
				elif (i is new_mesh_size-1 and j is 0):
					print 'R_bl.shape = ' + str(self._R_bl.shape) + 'x[i*2-1:i*2, j*2:j*2+2].shape = ' + str(x[i*2-1:i*2+1, j*2:j*2+2].shape)
					x_new[i,j] = sum(np.dot(self._R_bl, x[i*2-1:i*2+1, j*2:j*2+2]))
				# bottom right corner
				elif (i is new_mesh_size-1 and j is new_mesh_size-1):
					x_new[i,j] = sum(np.dot(self._R_br, x[i*2:i*2+1, j*2:j*2+1]))
				# bottom row but not a corner
				elif (i is new_mesh_size-1):
					x_new[i,j] = sum(np.dot(self._R_bottom, x[i*2:i*2+1, j*2-1:j*2+1]))
				# interior cell
				else:
					x_new[i,j] = sum(np.dot(self._R_interior, x[i*2-1:i*2+2, j*2-1:j*2+1]))


#				print 'i = %d, j = %d' % (i, j)
#				x_new[i,j] = sum(np.dot(self._R, x[i*2:i*2+3, j*2:j*2+3]))

		# Reshape x and b back into 1D arrays
		x_new = np.ravel(x_new)

		print 'x_new.size = %d, x_old.size = %d' % (x_new.size, x.size)

		return x_new



	def restrictAxb(self, A, x, b, m):
		'''

		'''

		#######################################################################
		#								Restrict M							  #
		#######################################################################

		# Extract the diagonals for M and F
		# Pad with zeros at front for superdiagonals / back for subdiagonals
		A = A.todense()
		size = A.shape[0]

		M_diag = [A[i,i] for i in range(size)]

		M_udiag = zeros(size)
		M_udiag[1:size] = [A[i,i+1] for i in range(size-1)]
		M_udiag = np.asarray(M_udiag)

		M_2udiag = zeros(size)
		M_2udiag[m:size] = [A[i,i+m] for i in range(size-m)]
		M_2udiag = np.asarray(M_2udiag)

		M_ldiag = zeros(size)
		M_ldiag[:size-1] = [A[i+1,i] for i in range(size-1)]
		M_ldiag = np.asarray(M_ldiag)

		M_2ldiag = zeros(size)
		M_2ldiag[:size-m] = [A[i+m,i] for i in range(size-m)]
		M_2ldiag = np.asarray(M_2ldiag)

		M_3ldiag = zeros(size)
		M_3ldiag[:size/2] = [A[i+size/2,i] for i in range(size/2)]
		M_3ldiag = np.asarray(M_3ldiag)


		# Restrict each energy group of each subdiagonal
		print 'size = %d' % (size)
		M_diag_egroup1_new = self.restrict(M_diag[:size/2])
		M_diag_egroup2_new = self.restrict(M_diag[size/2:])
		M_diag_new = np.append(M_diag_egroup1_new, M_diag_egroup2_new)

		M_udiag_egroup1_new = self.restrict(M_udiag[:size/2.])
		M_udiag_egroup2_new = self.restrict(M_udiag[size/2.:])
		M_udiag_new = np.append(M_udiag_egroup1_new, M_udiag_egroup2_new)

		M_2udiag_egroup1_new = self.restrict(M_2udiag[:size/2.])
		M_2udiag_egroup2_new = self.restrict(M_2udiag[size/2.:])
		M_2udiag_new = np.append(M_2udiag_egroup1_new, M_2udiag_egroup2_new)

		M_ldiag_egroup1_new = self.restrict(M_ldiag[:size/2.])
		M_ldiag_egroup2_new = self.restrict(M_ldiag[size/2.:])
		M_ldiag_new = np.append(M_ldiag_egroup1_new, M_ldiag_egroup2_new)

		M_2ldiag_egroup1_new = self.restrict(M_2ldiag[:size/2.])
		M_2ldiag_egroup2_new = self.restrict(M_2ldiag[size/2.:])
		M_2ldiag_new = np.append(M_2ldiag_egroup1_new, M_2ldiag_egroup2_new)

		M_3ldiag_egroup1_new = self.restrict(M_3ldiag[:size/2.])
		M_3ldiag_egroup2_new = self.restrict(M_3ldiag[size/2.:])
		M_3ldiag_new = np.append(M_3ldiag_egroup1_new, M_3ldiag_egroup2_new)


		# Construct the restricted A matrix
		A_new = dia_matrix(([M_diag_new, M_udiag_new, M_2udiag_new, 
							   M_ldiag_new, M_2ldiag_new, M_3ldiag_new], 
							   [0, 1, m/2., -1, -m/2., -size/4]), \
								shape=(size/4., size/4.))

				
		#######################################################################
		#							Restrict x and b						  #
		#######################################################################
		# Reshape x and b into 2D arrays corresponding to the geometric mesh
		# with separate arrays for each energy group
		print 'x.size = %d, x.size/2 = %d' % (x.size, x.size/2)
		print 'x[:(x.size/2)].shape = ' + str(x[:(x.size/2)].shape)
		x_egroup1 = np.asarray(x[:(x.size/2)])
		x_egroup2 = x[x.size/2:]
		b_egroup1 = b[:x.size/2]
		b_egroup2 = b[x.size/2:]

		# Restrict each energy group of x and b
		x_egroup1_new = self.restrict(x_egroup1)
		x_egroup2_new = self.restrict(x_egroup2)
		b_egroup1_new = self.restrict(b_egroup1)
		b_egroup2_new = self.restrict(b_egroup2)

		print 'x_egroup1.size = %d, x_egroup1_new.size = %d' % (x_egroup1.size, x_egroup1_new.size)
		print 'x_egroup2.size = %d, x_egroup2_new.size = %d' % (x_egroup2.size, x_egroup2_new.size)

		# Concatenate restricted x and b energy groups
		x_new = np.append(x_egroup1_new, x_egroup2_new)
		b_new = np.append(b_egroup1_new, b_egroup2_new)

		print 'x_new.size = ' + str(x_new.size)

		return A_new, x_new, b_new


	def prolongation(b, x):
		'''
		'''

		return

		# 
		m_old = sqrt(x.size)
		m_new = m_old

		# Reshape x and b into 2D arrays corresponding to the geometric mesh
		x = np.reshape(x, [sqrt(m_old), sqrt(m_old)])
		b = np.reshape(b, [sqrt(m_old), sqrt(m_old)])

		# Allocate memory for the restricted x
		x_new = np.array((sqrt(m_new), sqrt(m_new)))
		b_new = np.array((sqrt(m_new), sqrt(m_new)))

		# Restrict the x and b vectors
		for i in range(sqrt(m_new)):
			for j in range(sqrt(m_new)):
				x_new[i,j] = self._R * x[i*2:i*2+2, j*2:j*2+2]
				b_new[i,j] = self._R * b[i*2:i*2+2, j*2:j*2+2]

		return x_new, b_new
		


#	def newtonScipyGMRES(self, newton_iter, gmres_iter, tol):

		# Initial guess
#		x = ones((self._M.shape[1]+1, 1))
#		dx = 0.01 * ones((self._M.shape[1]+1, 1))
#		source_old = self._F * x[:x.size-1]

		# Outer Newton iteration
#		for i in range(newton_iter):

			# Compute keff
#			phi = x[:x.size-1]
#			source_new = self._F * x[:x.size-1]
#			keff = sum(source_new) / sum(source_old)
#			source_old = source_new

			# Compute residual vector
#			F = self.computeF(x)
#			res = norm(F)
#			print 'Newton-GMRES: i = %d	res = %1E	keff = %1.5f' % (i, res, keff)

			# Check for convergence
#			if res < tol:
#				print 'Newton-GMRES converged to %1d iterations for ' \
#						+ 'tol = %1E' % (i, res)
#				break

			# Use GMRES to solve for delta in J*delta = F equation
#			J = self.computeAnalyticJacobian(x)

			# ILU Preconditioned GMRES
#			P = spilu(J.tocsc(), drop_tol=1e-5)
#			D_x = lambda x: P.solve(x)
#			D = LinearOperator((J.shape[0], J.shape[0]), D_x)
#			dx = gmres(J.tocsr(), -F, tol=1E-3, maxiter=gmres_iter, M=D)


#			# Unpreconditioned GMRES
#			dx = gmres(J, -F, tol=1E-3, maxiter=gmres_iter)
#			dx = array(dx[0])
#			dx = reshape(dx, [dx.size, 1])

			# Update x and renormalize the flux
#			print 'x = ' + str(x)
#			x = x + dx

#			x[:x.size-1] = x[:x.size-1] / norm(x[:x.size-1])
			

#		print 'Newton-GMRES did not converge in %d iterations to tol = %1E' \
#														 % (newton_iter, tol)

#		phi = x[:x.size-1]
#		self.plotPhi(phi)


if __name__ == '__main__':

    # Parse command line options
	opts, args = getopt.getopt(sys.argv[1:], "t:n:psn", \
							["num_mesh", "plot_mesh", "spy", "tolerance"])

	# Default arguments
	num_mesh = 5
	tolerance = 1E-5
	plot_mesh = False
	spyMF = False

	for o, a in opts:
		if o in ("-n", "--num_mesh"):
#			num_mesh = min(int(a), 8)
			num_mesh = int(a)
		elif o in ("-t", "--tolerance"):
			tolerance = float(a)
		elif o in ("-p", "--plot_mesh"):
			plot_mesh = True
		elif o in ("-s", "--spy"):
			spyMF = True
		else:
			assert False, "unhandled option"


	prob = LRA.LRAProblem(num_mesh, num_mesh)
	prob.setupMFMatrices()
	solver = Solver()

	if plot_mesh:
		prob.plotMesh()
	if spyMF:
		prob.spyMF()

	x = np.ones((prob._num_x_cells*prob._num_y_cells*2))
	x = ravel(x)
	b = np.ones((prob._num_x_cells*prob._num_y_cells*2))
	b = ravel(b)
	print 'x.size = ' + str(x.size)
	[A, x, b] = solver.restrictAxb(prob._M, x, b, prob._num_x_cells)
	print 'x.size = ' + str(x.size)

	fig = figure()
	spy(A)
	show()

	A, x, b = solver.restrictAxb(A, x, b, prob._num_x_cells/2+1)

	fig = figure()
	spy(A)
	show()

#	phi = solver.powerIteration(prob._M, prob._F, 200, tolerance)
#	prob.plotPhi(phi)

#	LRA.newtonScipyGMRES(1000, 1000, tolerance)
