	def newtonGMRES(self, newton_iter, gmres_iter, tol):

		# Initial guess
		x = ones((self._M.shape[1]+1, 1))
		dx = 0.01 * ones((self._M.shape[1]+1, 1))
		x[:x.size-1] = x[:x.size-1] / norm(x[:x.size-1])

		# Outer Newton iteration
		for i in range(newton_iter):

			# Compute residual vector
			F = self.computeF(x)
			res = norm(F)
			print 'Newton-GMRES: i = %d	res = %1E	keff = %1.5f' % \
												(i,res, x[x.size-1])

			# Check for convergence
			if res < tol:
				print 'Newton-GMRES converged to %1d iterations for ' \
						+ 'tol = %1E' % (i, res)
				break

			# Use GMRES to solve for delta in J*delta = F equation
			dx, err = self.gmres_jfnk(x, dx, -F, gmres_iter, 100, 1E-3)

			print 'GMRES err = %1E' % (err)

			# Update x
			x = x + dx
			x[:x.size-1] = x[:x.size-1] / norm(x[:x.size-1])

		print 'Newton-GMRES did not converge in %d iterations to tol = %1E' \
														 % (newton_iter, tol)

		phi = x[:x.size-2]
		self.plotPhi(phi)



	def gmres_jfnk(self, x_old, x_new, b, res, outer_iter, tol):
	
		matvec=''

		# Pre-allocate arrays
		Q = np.zeros((b.size, res+1), dtype=np.float)
		H = np.zeros((res+1, res), dtype=np.float)
		err = np.zeros((outer_iter*res, 1), dtype=np.float)
		c = np.zeros((res+1, 1), dtype=np.float)
		s = np.zeros((res+1, 1), dtype=np.float)

		for k in range(outer_iter):
		
			if matvec is 'FD':
#				print 'x_old.size = ' + str(x_old.size)
				Ax = self.computeFDJacobVecProd(x_old,x_new)
			else:
				Ax = self.computeAnalyticJacobVecProd(x_old,x_new)
				

#			b = (1. / keff) * F * phi
#			r = b - M * phi
			r = Ax - b
			beta = np.linalg.norm(r)

			g = np.zeros((res+1, 1))
			g[0] = beta

			# Check for convergence
			if beta < tol:
				break

			# Compute first Q
			Q[:,0] = r.T / beta

			# GMRES	on the residual error
			for n in range(res):

				# Arnoldi
#				y = np.array(Q[:,n])[:][0]
				y = np.array(Q[:,n])[:]
#				v = M * y
				if matvec is 'FD':
#					print 'x_old.shape = ' + str(x_old.shape) + '	y.shape = ' + str(y.shape)
					Ay = self.computeFDJacobVecProd(x_old,y)
				else:
					Ay = self.computeAnalyticJacobVecProd(x_old,y)

				v = Ay
				normv1 = norm(v)

				# Loop over previous vectors
				for j in range(n):
					# Orthogonal projection of A onto new Krylov subspace
					# H = Q'AQ
					H[j,n] = np.vdot(Q[:,j], v)
				
					# Equation 33.4 in Trefethen
					v = v - (H[j,n] * Q[:,j])[:][0]

#					print 'GMRES: k = %d, n = %d, j = %d' % (k,n,j)

				# Compute new H
				H[n+1,n] = norm(v)
				normv2 = H[n+1,n]

				# May need to reorthogonalize here

				# Compute next column of Q
				Q[:,n+1] = (v / H[n+1,n]).T

				# Apply Givens rotation to new column of H
				givens(H, c, s, g, n)

				# Compute the error for this outer/inner iteration pair
				err[n+res*(k-1)] = abs(g[n+1])
			
				# Check for convergence
				if (err[n+res*(k-1)] < tol):
					break

			# Compute y
			y = np.linalg.solve(H[:n,:n], g[:n])

			# Compute new phi
			x_new = x_new + dot(Q[:,:n], y)
#			phi = phi + dot(Q[:,:n], y)
#			phi = phi / np.linalg.norm(phi)
		
			# Check convergence and break outer loop
			if err[k*n-1] < tol:
				print 'GMRES converged to %1E in %d iterations' % \
													(err[k*n-1], n)
				break

			if k == outer_iter-1:
				print 'GMRES did not converge'

		return x_new, err[k*n-1]




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

