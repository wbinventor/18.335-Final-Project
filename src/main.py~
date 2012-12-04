import numpy as np
from pylab import *
from scipy.sparse import *
from scipy.sparse.linalg import *
from os import *
from sys import *
from time import time
import getopt

import LRAProblem as LRA
	

class Solver(object):

	def __init__(self, lra):

		self._LRA = lra


	def linearSolve(self, A, b, tolerance=1E-5, restart_iter=None, 
					outer_iter=None, method='gmres', preconditioned=False):
		'''
		Solves the linear system Ax=b for x using one of SciPy's 
		iterative solvers
		'''

		# If ILU Preconditioned
		if preconditioned:

			P = spilu(A.tocsc(), drop_tol=1e-5)
			D_x = lambda x: P.solve(x)
			D = LinearOperator((A.shape[0], A.shape[0]), D_x)

			if method in ('bicgstab'):
				x = bicgstab(A, b, tol=tolerance, maxiter=outer_iter, M=D)
			elif method in ('bicg'):
				x = bicg(A, b, tol=tolerance,	maxiter=outer_iter, M=D)
			elif method in ('gmres'):
				x = gmres(A, b, tol=tolerance, restart=restart_iter, \
													maxiter=outer_iter, M=D)
			elif method in ('cg'):
				x = cg(A, b, tol=tolerance, maxiter=outer_iter, M=D)
			elif method in ('cgs'):
				x = cgs(A, b, tol=tolerance, maxiter=outer_iter, M=D)
			elif method in ('qmr'):
				x = qmr(A, b, tol=tolerance, maxiter=outer_iter, M=D)

		# Unpreconditioned
		else:
			if method in ('bicgstab'):
				x = bicgstab(A, b, tol=tolerance, maxiter=outer_iter)
			elif method in ('bicg'):
				x = bicg(A, b, tol=tolerance, maxiter=outer_iter)
			elif method in ('gmres'):
				x = gmres(A, b, tol=tolerance, restart=restart_iter, \
														maxiter=outer_iter)
			elif method in ('cg'):
				x = cg(A, b, tol=tolerance, maxiter=outer_iter)
			elif method in ('cgs'):
				x = cgs(A, b, tol=tolerance, maxiter=outer_iter)
			elif method in ('qmr'):
				x = qmr(A, b, tol=tolerance, maxiter=outer_iter)

		x = array(x[0])
		x = reshape(x, [x.size, 1])

		return x



	def powerIteration(self, M, F, max_iters, tol, method='gmres', guess=None, \
									preconditioned=False, plot_residual=False):
		'''
		Perform power iteration for the keff eigenvalue problem.
		Converges the source to a specified tolerance and plots the
		fast and thermal flux and the residual at each iteration
		'''

		# Start the timer
		start_time = time()

		# Guess initial keff
		keff = 1.0

		# Guess initial flux and normalize it if user did not provide it
		if guess is None:
			phi = np.ones((M.shape[1], 1))
		else:
			phi = guess

		phi = phi / norm(phi)

		# Array for phi_res and keff_res
		res = []
		keff_res = []

		for i in range(max_iters):

			# Update flux vector
			phi_new = self.linearSolve(M, ((1. / keff) * F * phi), \
						method=method, preconditioned=preconditioned)

			# Normalize new flux
			phi_new = phi_new / norm(phi_new)
	
			# Update keff
			source_new = F * phi_new
			source_old = F * phi

			tot_source_new = sum(source_new)
			tot_source_old = sum(source_old)

			keff_new = tot_source_new / tot_source_old

			# Compute residuals
			res.append(norm(source_old - source_new))
			keff_res.append(abs(keff_new - keff) / keff)

			print ("Power iteration: i = %d	res = %1.5E	keff = %1.5f"
											% (i, res[i], keff))

			# Check convergence
			if res[i] < tol:
				elapsed_time = time() - start_time
				print ("Power iteration converged in %1d iters and %1.3f sec"
						" with res = %1E" % (i, elapsed_time, res[i]))
				break
			else:
				phi = phi_new
				if (i > 1):
					keff = keff_new


		# Resize phi as a 2D array
		phi = np.reshape(phi, (phi.size, 1))

		# Plot the residuals of phi and keff at each iteration
		if plot_residual:
			fig = figure()
			plot(np.array(range(i+1)), np.array(res))
			plot(np.array(range(i+1)), np.array(keff_res))
			legend(['flux', 'keff'])
			yscale('log')
			xlabel('iteration #')
			ylabel('residual')
			title('Power Iteration Residuals')

		return phi, keff




	def newtonKrylov(self, M, F, FD=True, newton_iter=20, inner_iter=25, \
					tol=1E-5, guess=None, method='gmres', plot_residual=False):
		'''
		Solves for the maximal eigenvector of a linear system using an 
		Inexact Newton outer iteration with a Krylov or other iterative 
		linear solver for an inner iteration.
		'''

		# Start the timer
		start_time = time()

		# Initial guess for phi
		if guess is None:
			self._x = ones((M.shape[1]+1, 1))
		else:
			self._x = np.concatenate((guess, np.ones((1,1))))

		# Get initial phi from x and normalize
		phi = self._x[:self._x.size-1]
		phi = phi / norm(phi)

		# Guess initial keff
		keff = 1.0

		# Initial guess for the Newton update
		dx = 0.01 * ones((M.shape[1]+1, 1))

		# Compute the initial source
		source_old = F * self._x[:self._x.size-1]

		# Array for source and keff residuals
		res = []
		keff_res = []

		# Outer Newton iteration
		for i in range(newton_iter):

			# Compute residual vector
			Fx = self.computeF(self._x)

			# Create a linear operator to compute the Jacobian vector product
			if FD:
				J = LinearOperator( (M.shape[0]+1,M.shape[1]+1), dtype=float, \
									matvec=self.computeFDJacobVecProd )
			else:
				J = LinearOperator( (M.shape[0]+1,M.shape[1]+1), dtype=float, \
						matvec=self.computeAnalyticJacobVecProd )

			# Find the Newton update
			dx = self.linearSolve(J, -Fx, outer_iter=inner_iter, \
									method=method, tolerance=1E-4)

			# Update x
			self._x += dx			

			# Normalize the flux
			phi_new = self._x[:self._x.size-1]
			phi_new = phi_new / norm(phi_new)
	
			# Update keff
			source_new = F * phi_new
			source_old = F * phi
			tot_source_new = sum(source_new)
			tot_source_old = sum(source_old)
			keff_new = tot_source_new / tot_source_old

			# Compute residuals
			res.append(norm(source_old - source_new))
			keff_res.append(abs(keff_new - keff) / keff)

			print ("JFNK: i = %d	res = %1.5E"	
					"	keff = %1.5f" % (i, res[i], keff))

			# Check for convergence
			if res[i] < tol:
				elapsed_time = time() - start_time
				print ("JFNK converged in %1d iters and %1.3f sec"
						" with res = %1E" % (i, elapsed_time, res[i]))
				break
			else:
				phi = phi_new
				if (i > 1):
					keff = keff_new

		# Plot the residuals of phi and keff at each iteration
		if plot_residual:
			fig = figure()
			plot(np.array(range(i+1)), np.array(res))
			plot(np.array(range(i+1)), np.array(keff_res))
			legend(['flux', 'keff'])
			yscale('log')
			xlabel('iteration #')
			ylabel('residual')
			title('Power Iteration Residuals')

		return phi_new



	def computeF(self, x):
		'''
		Compute the residual vector for JFNK
		'''

		m = x.size
		phi = x[:m-1]
		lamb = x[m-1][0]

		# Allocate space for F
		F = ones((m, 1))
	
		# M - lambda * F * phi constraint
		F[:m-1] = self._LRA._M * phi - lamb * self._LRA._F * phi

		# Flux normalization constraint
		F[m-1] = -0.5 * vdot(phi, phi) + 0.5

		return F



	def computeAnalyticJacobVecProd(self, y):
		'''
		'''

		m = self._x.shape[0]
		phi = self._x[:m-1]
		lamb = self._x[m-1]

		# Construct temporary blocks for Jacobian
		a = self._LRA._M - lamb[0] * self._LRA._F
		b = phi.T
		c = -self._LRA._F * phi
		c = vstack([c, zeros(1)])

		# Build Jacobian using scipy's sparse matrix stacking operators
		J = vstack([a, b])
		J = hstack([J, c])

		return J * y




	def computeFDJacobVecProd(self, y):
		'''
		'''

		phi = self._x[:self._x.size-2]
		lamb = self._x[self._x.size-1]		
		y = resize(y, [y.size, 1])

		b = 1E-8

		epsilon = b * sum(self._x) / (y.size * norm(y))
		
		# Approximate Jacobian matrix vector multiplication
		tmp1 = self.computeF(self._x + (epsilon * y))
		tmp2 = self.computeF(self._x)
		Jy = (tmp1 - tmp2) / epsilon

		return Jy


	def computeAnalyticJacobian(self):
		'''
		'''

		m = self._x.shape[0]
		phi = self._x[:m-1]
		lamb = self._x[m-1][0]

#		J = lil_matrix((m,m))

		# Construct temporary blocks for Jacobian
		a = self._LRA._M - lamb * self._LRA._F
		b = -phi.T
		c = -self._LRA._F * phi
		c = vstack([c, zeros(1)])

		# Build Jacobian using scipy's sparse matrix stacking operators
		J = vstack([a, b])
		J = hstack([J, c])

		return J




	def constructRestrictionOperator(self, mesh_old, method='fullweighting'):
		'''
		Requires mesh to be a factor of 2 or 3
		'''

		# Compute new mesh size
		if mesh_old % 2 is 0:
			mesh_new = mesh_old / 2
		elif mesh_old % 3 is 0:
			mesh_new = mesh_old / 3
		else:
			print 'Unable to restrict a mesh size of %d.'\
				+ 'Adjust mesh size to be a factor of 2 or 3.' % (mesh_old)
			sys.exit(0)

		print 'mesh_old = %d, mesh_new = %d' % (mesh_old, mesh_new)

		# Create restriction operator for using different stencil types
		if method is 'injection':

			# Construct base stencil vector
			stencil = np.zeros((1,mesh_old**2))
			stencil[0,0] = 1.


		elif method is 'fullweighting':

			# Old odd mesh
			if mesh_old % 2 is 1:
				# Construct base stencil vector
				stencil = np.zeros((1,mesh_old**2))
				stencil[0,0:3] = [0.25, 0.5, 0.25]
				stencil[0,mesh_old:mesh_old+3] = [0.5, 1., 0.5]
				stencil[0,2*mesh_old:2*mesh_old+3] = [0.25, 0.5, 0.25]
				stencil *= 0.25

			# Old even mesh
			else:
				# Construct base stencil vector
				stencil = np.zeros((1,mesh_old**2))
				stencil[0,0:2] = [0.5, 0.5]
				stencil[0,mesh_old:mesh_old+2] = [0.5, 0.5]
				stencil *= 0.5

		else:
			print str(method) + ' not yet implemented for restriction operator'
			sys.exit(0)

		# Construct restriction operator from the stencil
		stencil = np.reshape(stencil, (1,stencil.size))
		R = np.zeros((mesh_new**2,mesh_old**2))
		for i in range(mesh_new**2):
			if mesh_old % 2 is 0:
				R[i,:] = np.roll(stencil, (i*2 + (i/mesh_new) * mesh_old))
			elif mesh_old % 3 is 0:
				R[i,:] = np.roll(stencil, (i*3 + (i/mesh_new) * mesh_old))

		# convert to sparse matrix
		sparseR = csr_matrix(R)

		return sparseR



	def constructProlongationOperator(self, mesh_old, method='fullweighting'):
		'''
		'''
		
		R = self.constructRestrictionOperator(mesh_old, method)

		# Create prolongation operator for using different stencil types
		if method is 'injection':
			P = csr_matrix.transpose(R)

		elif method is 'fullweighting':

			# Old odd mesh to new even mesh
			if mesh_old % 2 is 1:
				P = 4. * csr_matrix.transpose(R)
			
			# Old even mesh to new odd mesh
			else:
				P = 2. * csr_matrix.transpose(R)

		else:
			print str(method) + ' not yet implemented for prolongation operator'
			sys.exit(0)

		sparseP = csr_matrix(P)

		return P



	def restrictResidual(self, r, method='fullweighting'):
		'''
		'''

		# Compute the current mesh size
		m_old = int(sqrt(r.size/2))
		
		# Apply the restriction operator to the residual
		R = self.constructRestrictionOperator(m_old, method)

		r_new1 = R * r[:r.size/2]
		r_new2 = R * r[r.size/2:]
		r_new = np.append(r_new1, r_new2)

		return r_new



	def restrictCoeffMatrix(self, A, method='fullweighting'):
		'''
		This does not seem to work except for even mesh sizes
		'''

		# Compute new mesh size
		m_old = int(sqrt(A.shape[0]/2))
		
		# Apply the restriction/prolongation operators to coeff matrix
		# to restrict it according to the Galerkin Condition
		R = self.constructRestrictionOperator(m_old, method)
		P = self.constructProlongationOperator(m_old, method)
		A = lil_matrix(A)
		A_new11 = csr_matrix(csr_matrix(R * A[:m_old**2, :m_old**2]) * P)
		A_new12 = csr_matrix(csr_matrix(R * A[:m_old**2, m_old**2:]) * P)
		A_new21 = csr_matrix(csr_matrix(R * A[m_old**2:, :m_old**2]) * P)
		A_new22 = csr_matrix(csr_matrix(R * A[m_old**2:, m_old**2:]) * P)
		A_new = bmat([[A_new11,A_new12],[A_new21,A_new22]])

		return A_new



	def prolongResidual(self, r, m_old, method='fullweighting'):
		'''
		'''
		
		# Apply the restriction operator to the residual
		m_old /= 2
		P = self.constructProlongationOperator(m_old, method)

		r_new1 = P * r[:r.size/2]
		r_new2 = P * r[r.size/2:]
		r_new = np.append(r_new1, r_new2)

		return r_new



	def multigridVCycle(self, M, F, num_cycles, tol, precond=False):
		
		# Pre-relaxation using power iteration to get initial guess
		phi, keff = self.powerIteration(M, F, 2, 1E-5, method='bicgstab')

		# Compute residual
		Au = (1. / keff) * F * phi
		Av = M * phi
		r = Au - Av

		# Restrict the residual and coefficient matrix
		r_2h = solver.restrictResidual(r, 'fullweighting')
#		M_2h = solver.restrictCoeffMatrix(M, 'fullweighting')

		num_fine_mesh = int(sqrt(r.size/2))
		num_coarse_mesh = int(sqrt(r_2h.size/2))

		# Construct the M, F matrices for the coarse problem
		coarse_mesh = int(sqrt(r_2h.size/2))
		coarse_prob = LRA.LRAProblem(coarse_mesh/11, coarse_mesh/11)
		M_2h = coarse_prob._M

#		fine_prob = LRA.LRAProblem(coarse_mesh*2/11, coarse_mesh*2/11)
#		fine_prob.plotPhi(phi)
#		fine_prob.plotPhi(r)

#		phi_2h = solver.restrictResidual(phi, 'fullweighting')
#		coarse_prob.plotPhi(phi_2h)
#		phi_h = solver.prolongResidual(phi_2h, num_fine_mesh*2)
#		fine_prob.plotPhi(phi_h)


		# Solve the residual equation Ae=r exactly at this second level
		if method is 'bicgstab':
			e_2h = bicgstab(M_2h, r_2h)
		elif method is 'bicg':
			e_2h = bicg(M_2h, r_2h)
		elif method is 'gmres':
			e_2h = gmres(M_2h, r_2h)
		elif method is 'cg':
			e_2h = cg(M_2h, r_2h)
		elif method is 'cgs':
			e_2h = cgs(M_2h, r_2h)
		elif method is 'qmr':
			e_2h = qmr(M_2h, r_2h)

		e_2h = e_2h[0]
		e_2h = np.reshape(e_2h, (e_2h.size, 1))

		# Prolong / Interpolate the residual
		e_h = solver.prolongResidual(e_2h, num_fine_mesh*2)

		# Correct residual with approximate error
		e_h = np.reshape(e_h, (e_h.size, 1))
		phi = phi + e_h

#		coarse_prob.plotPhi(e_2h)
#		fine_prob.plotPhi(e_h)
#		fine_prob.plotPhi(phi)

		phi, keff = self.powerIteration(M, F, 10, 1E-5, method='bicgstab', \
																guess=phi)


if __name__ == '__main__':

    # Parse command line options
	opts, args = getopt.getopt(sys.argv[1:], "t:n:i:o:psn", \
					["num_mesh", "plot_mesh", "spy", "tolerance", \
									"inner-method", "outer-method"])

	# Default arguments
	num_mesh = 5
	tolerance = 1E-8
	plot_mesh = False
	spyMF = False
	inner_method = 'bicgstab'
	outer_method = 'jfnk-fd'

	for o, a in opts:
		if o in ("-n", "--num_mesh"):
			num_mesh = int(a)
		elif o in ("-t", "--tolerance"):
			tolerance = float(a)
		elif o in ("-p", "--plot_mesh"):
			plot_mesh = True
		elif o in ("-s", "--spy"):
			spyMF = True
		elif o in ("-i", "--inner-method"):
			if a in ('bicgstab'):
				inner_method = 'bicgstab'
			elif a in ('bicg'):
				inner_method = 'bicg'
			elif a in ('gmres'):
				inner_method = 'gmres'
			elif a in ('cg'):
				inner_method = 'cg'
			elif a in ('cgs'):
				inner_method = 'cgs'
			elif a in ('qmr'):
				inner_method = 'qmr'
			else:
				print '%s is not a recognized linear solver' % (a)
		elif o in ("-o", "--outer-method"):
			if a in ('power'):
				outer_method = 'power'
			elif a in ('jfnk-fd'):
				outer_method = 'jfnk-fd'
			elif a in ('jfnk-analytic'):
				outer_method = 'jfnk-analytic'
			else:
				print '%s is not recognized solver type' % (a)
		else:
			assert False, 'unhandled option'

	prob = LRA.LRAProblem(num_mesh, num_mesh)
	prob.setupMFMatrices()
	solver = Solver(prob)

	# Plot any items requested by the user
	if plot_mesh:
		prob.plotMesh()
		prob.plotMaterials()
	if spyMF:
		prob.spyMF()


	if outer_method in ('power'):
		phi, keff = solver.powerIteration(prob._M, prob._F, 1000, \
								method=inner_method, tol=tolerance)
		prob.plotPhi(phi)

	elif outer_method in ('jfnk-fd'):
		phi = solver.newtonKrylov(prob._M, prob._F, FD=True, newton_iter=100, 
									inner_iter=1000, method=inner_method, \
									tol=tolerance, plot_residual=True)
		prob.plotPhi(phi)

	elif outer_method in ('jfnk-analytic'):
		phi = solver.newtonKrylov(prob._M, prob._F, FD=False, newton_iter=100, 
									inner_iter=1000, method=inner_method, \
									tol=tolerance, plot_residual=True)
		prob.plotPhi(phi)

#	phi, keff = solver.multigridVCycle(prob._M, prob._F, 2, method=method, tol=tolerance)


#	x = np.ones((prob._num_x_cells*prob._num_y_cells*2))
#	x = ravel(x)
#	method = 'fullweighting'

#	x = solver.restrictResidual(x, method)
#	A = solver.restrictCoeffMatrix(prob._M, method)

#	fig = figure()
#	spy(A)
#	show()

#	x = solver.restrictResidual(x, method)
#	A = solver.restrictCoeffMatrix(A, method)

#	fig = figure()
#	spy(A)
#	show()

