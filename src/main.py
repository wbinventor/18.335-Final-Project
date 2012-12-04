import numpy as np
from pylab import *
from scipy.sparse import *
from scipy.sparse.linalg import *
from os import *
from sys import *
import getopt

import LRAProblem as LRA
	

class Solver(object):

	def __init__(self, lra):

		self._LRA = lra


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
		phi, keff = self.powerIteration(M, F, 'bicgstab', 2, 1E-5)

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

		phi, keff = self.powerIteration(M, F, 'bicgstab', 10, 1E-5, guess=phi)



	def powerIteration(self, M, F, method, max_iters, tol, guess=None, precond=False):
		'''
		Perform power iteration for the keff eigenvalue problem.
		Converges the source to a specified tolerance and plots the
		fast and thermal flux and the residual at each iteration
		'''

		# Guess initial keff
		keff = 1.0

		# Guess initial flux and normalize it if user did not provide it
		if guess is None:
			phi = np.ones((M.shape[1], 1))
			phi = phi / norm(phi)
		else:
			phi = guess
			phi = phi / norm(phi)

		# Array for phi_res and keff+res
		phi_res = []
		keff_res = []

		for i in range(max_iters):

			# Update flux vector

			# If Preconditioned
			if precond:
				P = spilu(M.tocsc(), drop_tol=1e-5)
				D_x = lambda x: P.solve(x)
				D = LinearOperator((M.shape[0], M.shape[0]), D_x)

				if method is 'bicgstab':
					phi_new = bicgstab(M.tocsr(), ((1. / keff) * F * phi), M=D)
				elif method is 'bicg':
					phi_new = bicg(M.tocsr(), ((1. / keff) * F * phi), M=D)
				elif method is 'gmres':
					phi_new = gmres(M.tocsr(), ((1. / keff) * F * phi), M=D)
				elif method is 'cg':
					phi_new = cg(M.tocsr(), ((1. / keff) * F * phi), M=D)
				elif method is 'cgs':
					phi_new = cgs(M.tocsr(), ((1. / keff) * F * phi), M=D)
				elif method is 'qmr':
					phi_new = qmr(M.tocsr(), ((1. / keff) * F * phi), M=D)

			# Unpreconditioned
			else:
				if method is 'bicgstab':
					phi_new = bicgstab(M.tocsr(), ((1. / keff) * F * phi))
				elif method is 'bicg':
					phi_new = bicg(M.tocsr(), ((1. / keff) * F * phi))
				elif method is 'gmres':
					phi_new = gmres(M.tocsr(), ((1. / keff) * F * phi))
				elif method is 'cg':
					phi_new = cg(M.tocsr(), ((1. / keff) * F * phi))
				elif method is 'cgs':
					phi_new = cgs(M.tocsr(), ((1. / keff) * F * phi))
				elif method is 'qmr':
					phi_new = qmr(M.tocsr(), ((1. / keff) * F * phi))

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
					% (i, phi_res[i], keff_res[i])

			# Check convergence
			if phi_res[i] < tol and keff_res[i] < tol:
				print 'Power iteration converged in %1d iters with keff = %1.5f' \
																		% (i, keff)
				break
			else:
				phi = phi_new
				if (i > 1):
					keff = keff_new


		# Resize phi as a 2D array
		phi = np.reshape(phi, (phi.size, 1))

		# Plot the residuals of phi and keff at each iteration
		fig = figure()
		plot(np.array(range(i+1)), np.array(phi_res))
		plot(np.array(range(i+1)), np.array(keff_res))
		legend(['flux', 'keff'])
		yscale('log')
		xlabel('iteration #')
		ylabel('residual')
		title('Power Iteration Residuals')

		return phi, keff



	def newtonScipyGMRES(self, M, F, newton_iter, gmres_iter, tol, guess=None):

		# Initial guess
		if guess is None:
			x = ones((M.shape[1]+1, 1))
		else:
			x = guess

		dx = 0.01 * ones((M.shape[1]+1, 1))
		source_old = F * x[:x.size-1]

		# Outer Newton iteration
		for i in range(newton_iter):

			# Compute keff
			phi = x[:x.size-1]
			source_new = F * x[:x.size-1]
			keff = sum(source_new) / sum(source_old)
			res = norm(source_new - source_old)
			source_old = source_new

			# Compute residual vector
			Fx = self._LRA.computeF(x)
			print 'res = ' + str(res)
#			res = norm(Fx)
			print 'type(res) = ' + str(type(res))
			print 'Newton-GMRES: i = %d	res = %1E	keff = %1.5f' % (i, res, keff)

			# Check for convergence
			if res < tol and i > 1:
				print 'Newton-GMRES converged to %1d iterations for tol = %1E' % (i, res)
				break

			# Use GMRES to solve for delta in J*delta = F equation
			J = self._LRA.computeAnalyticJacobian(x)

			# ILU Preconditioned GMRES
			P = spilu(J.tocsc(), drop_tol=1e-5)
			D_x = lambda x: P.solve(x)
			D = LinearOperator((J.shape[0], J.shape[0]), D_x)
			dx = gmres(J.tocsr(), -Fx, tol=1E-3, maxiter=gmres_iter, M=D)


			# Unpreconditioned GMRES
			dx = gmres(J, -Fx, tol=1E-3, maxiter=gmres_iter)
			dx = array(dx[0])
			dx = reshape(dx, [dx.size, 1])

			# Update x and renormalize the flux
#			print 'x = ' + str(x[:5])
			x = x + dx

			x[:x.size-1] = x[:x.size-1] / norm(x[:x.size-1])
			

		print 'Newton-GMRES did not converge in %d iterations to tol = %1E' \
														 % (newton_iter, tol)

		phi = x[:x.size-1]

		return phi



if __name__ == '__main__':

    # Parse command line options
	opts, args = getopt.getopt(sys.argv[1:], "t:n:m:psn", \
					["num_mesh", "plot_mesh", "spy", "tolerance", "method"])

	# Default arguments
	num_mesh = 5
	tolerance = 1E-5
	plot_mesh = False
	spyMF = False
	method = 'bicgstab'

	for o, a in opts:
		if o in ("-n", "--num_mesh"):
			num_mesh = int(a)
		elif o in ("-t", "--tolerance"):
			tolerance = float(a)
		elif o in ("-p", "--plot_mesh"):
			plot_mesh = True
		elif o in ("-s", "--spy"):
			spyMF = True
		elif o in ("-m", "--method"):
			if a is 'bicgstab':
				method = 'bicgstab'
			elif a is 'bicg':
				method = 'bicg'
			elif a is 'gmres':
				method = 'gmres'
			elif a is 'cg':
				method = 'cg'
			elif a is 'cgs':
				method = 'cgs'
			elif a is 'qmr':
				method = 'qmr'
			else:
				print str(a) + ' is not a recognized linear solver'
		else:
			assert False, "unhandled option"


	prob = LRA.LRAProblem(num_mesh, num_mesh)
	prob.setupMFMatrices()

	if plot_mesh:
		prob.plotMesh()
		prob.plotMaterials()
	if spyMF:
		prob.spyMF()

	solver = Solver(prob)


#	phi, keff = solver.powerIteration(prob._M, prob._F, method, 200, tolerance)

#	phi, keff = solver.multigridVCycle(prob._M, prob._F, 2, tolerance, precond=False)

	phi, keff = solver.powerIteration(prob._M, prob._F, method, 5, tolerance)
	phi = solver.newtonScipyGMRES(prob._M, prob._F, 100, 10, tolerance, guess=phi)

	prob.plotPhi(phi)

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

