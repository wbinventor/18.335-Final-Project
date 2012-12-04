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
		
