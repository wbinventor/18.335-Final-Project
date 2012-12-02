import numpy as np
from pylab import *
from scipy.sparse import *
from scipy.sparse.linalg import *
from os import *
from sys import *
import getopt

import LRAProblem as LRA
	

class Solver(object):


	def constructRestrictionOperator(self, mesh_old, method='fullweighting'):
		'''
		'''

		# Compute new mesh size
		mesh_new = int(floor(mesh_old / 2.))	# Round down for odd mesh size

		print 'mesh_old = %d, mesh_new = %d' % (mesh_old, mesh_new)

		# Create relaxation operator for using different stencil types
		if method is 'injection':
			print 'Injection not yet implemented for restriction operator'
			sys.exit(0)

		elif method is 'fullweighting':

			# Old odd mesh to new even mesh
			if mesh_old % 2 is 1:
				# Construct base stencil vector
				stencil = np.zeros((1,mesh_old**2))
				stencil[0,0:3] = [0.25, 0.5, 0.25]
				stencil[0,mesh_old:mesh_old+3] = [0.5, 1., 0.5]
				stencil[0,2*mesh_old:2*mesh_old+3] = [0.25, 0.5, 0.25]
				stencil *= 0.25

			# Old even mesh to new odd mesh
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
			R[i,:] = np.roll(stencil, (i*2 + (i/mesh_new) * mesh_old))

		# convert to sparse matrix
		sparseR = csr_matrix(R)

		return sparseR



	def constructProlongationOperator(self, mesh_old, method='fullweighting'):
		'''
		'''
		
		R = self.constructRestrictionOperator(mesh_old, method)

		# Create prolongation operator for using different stencil types
		if method is 'injection':
			print 'Injection not yet implemented for prolongation operator'
			sys.exit(0)

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
		


	def powerIteration(self, M, F, method, max_iters, tol, precond=False):
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
	if spyMF:
		prob.spyMF()

	solver = Solver()


	phi = solver.powerIteration(prob._M, prob._F, method, 200, tolerance)
	prob.plotPhi(phi)

#	LRA.newtonScipyGMRES(1000, 1000, tolerance)

	if False:
		x = np.ones((prob._num_x_cells*prob._num_y_cells*2))
		x = ravel(x)
		b = np.ones((prob._num_x_cells*prob._num_y_cells*2))
		b = ravel(b)

		x = solver.restrictResidual(x, 'fullweighting')
		A = solver.restrictCoeffMatrix(prob._M, 'fullweighting')

		fig = figure()
		spy(A)
		show()

		x = solver.restrictResidual(x, 'fullweighting')
		A = solver.restrictCoeffMatrix(A, 'fullweighting')

		fig = figure()
		spy(A)
		show()

