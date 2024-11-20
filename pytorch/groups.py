import numpy as np
# import tensorflow as tf
import torch
from abc import ABC, abstractmethod


class GroupBase(ABC): 
	"""
	Abstract Base Class.

	All @abstractmethod must be implemented in subclasses.

	The setup of 3D space:

			dim2-axis
				^
				|
				|
				|
				|
				|
				O--------------->dim1-axis
			   /
			  /
			 /
			/
		   v  
		dim0-axis

	Denote a rotation by (i,j,k),
	i: rotate dim0-axis to dim2-axis by 90*i degree,
	j: rotate dim0-axis to dim1-axis by 90*j degree,
	k: rotate dim1-axis to dim2-axis by 90*k degree,
	IMPORTANT: rotation order matters! We first perform "i" rotation then "j" then "k" rotation. A vector r=[x,y,z]^T after the rotation should be Rk*Rj*Ri*r.
	In this sense, the full 24 rotations are {(i,0,0), (i,1,0), (i,2,0), (i,3,0), (i,0,1), (i,0,3)} for i = 0,1,2,3.
	"""

	def __init__(self):
		matrix_group = self.get_matrix_group() # abstract method to be implemented in subclasses
		self.cayleytable = GroupBase.matrix_group_cayleytable(matrix_group)
		self.group_elements = [GroupBase.matrix_to_ijk(matrix) for matrix in matrix_group] # e.g. [(0,0,0), (0,2,0), (2,0,0), (0,0,2)]
		self.group_dim = len(self.group_elements)
		self.inverse_map = [self.inverse(i) for i in range(self.group_dim)]
		print(self.group_elements)
		print(self.cayleytable)


	@abstractmethod
	def get_matrix_group(self):
		"""
			Find the isomorphic matrix group, i.e., all rotation matrices corresponding to the group elements.
			About the construction of group elements and the order, see get_Grotations.
			This is used to construct the Cayley table.
		"""
		raise NotImplementedError("Subclasses should implement this method.")


	@staticmethod
	def ijk_to_matrix(i,j,k):
		"""
			Return a rotation matrix in 3D vector space.
		"""
		c = [1.,0.,-1.,0.]
		s = [0.,1.,0.,-1]
		Ri = np.asarray([[c[i],     0.,     -s[i]],
						[0.,       1.,     0.],
						[s[i],    0.,     c[i]]])
		Rj = np.asarray([[c[j],     -s[j],  0.],
						[s[j],     c[j],   0.],
						[0.,       0.,     1.]])
		Rk = np.asarray([[1.,       0.,     0.],
						[0.,       c[k],   -s[k]],
						[0.,       s[k],   c[k]]])
		return Rk @ Rj @ Ri
	
	
	@staticmethod
	def matrix_to_ijk(R : np.array):
		"""
			Return a (i,j,k) index given a rotation matrix in 3D vector space.
		"""
		for i in range(4):
			for j in range(4):
				for k in range(4):
					if np.allclose(R, GroupBase.ijk_to_matrix(i,j,k)):
						return (i,j,k)
		return None
	

	@staticmethod
	def matrix_group_cayleytable(Z):
		"""
			Find the Cayley table of the group, by simulating group operations in the isomorphic matrix group.
			The result Cayley[i, j] is the index of the group element gj * gi.
		"""
		group_size = len(Z)
		cayley = []
		for y in Z:
			for z in Z:
				r = z @ y
				for i, el in enumerate(Z):
					if np.allclose(r, el):
						cayley.append(i)
						break
		assert len(cayley) == group_size*group_size
		cayley = torch.tensor(cayley, dtype=torch.int32)
		cayley = cayley.reshape(group_size, group_size)
		return cayley


	def rotate_tensor(self, input, element, start_dim):
		"""
			Input (3D filter) shape [n_channels,N0,N1,N2].
			Return a tensor rotated by (i,j,k).			
		"""
		i,j,k = self.group_elements[element]
		if i:
			input = torch.rot90(input, k=i, dims=(start_dim+0,start_dim+2))
		if j:
			input = torch.rot90(input, k=j, dims=(start_dim+0,start_dim+1))
		if k:
			input = torch.rot90(input, k=k, dims=(start_dim+1,start_dim+2))
		return input
	

	def permute_tensor(self, input, element, dim):
		"""
			Input:
			[out_channels,in_channels,in_group,N0,N1,N2]
			The group dimension of input will be permuted according to the element.
		"""
		# permute within the group_dim according to Cayley table's gi-th column, which means to perform e = gi * e for each element
		# The following is to move the gi*e input to the e output
		# input = tf.gather(input, self.cayleytable[:, element], axis=-1)
		# But we want the e input to the gi*e position in output
		# So we need to inverse the permutation
		
		# input = input[perm] means input[i] = input[perm[i]]
		# we need input[perm[i]] to be input[i],
		# so input = input[inv_perm]
		inv_perm = self.cayleytable[:, self.inverse_map[element]]
		# idx = [slice(None)] * len(input.shape)  # Keeps other dimensions unchanged
		# idx[dim] = inv_perm
		# return input[idx]
		# if dim == 2:
			
		# input = input[:, :, inv_perm, :, :, :]
		# print(inv_perm)
		input = torch.index_select(input, dim=dim, index=inv_perm)
		return input
	

	def inverse(self, gi):
		"""Return the inverse of gi"""
		for i in range(self.group_dim):
			if self.cayleytable[i,gi] == 0:
				return i


	def get_Grotations(self, x):
		"""Rotate the tensor x with all rotations in group
		Args:
			x (3D filter): [out_channels,in_group,in_channels,N0,N1,N2],
			Here we DON'T use ambiguous and confusing h,w,d to discribe dimensions.
			Just use dim0,dim1,dim2 or x,y,z.
			N0,N1,N2 are lengths along dim0,dim1,dim2.
		Returns:
			list of full rotations of x [[out_channels,in_group,in_channels,N0,N1,N2],....]
		"""
		return [
			self.rotate_tensor(x, element=g, start_dim=3) 
			for g in range(self.group_dim)
		]

	def get_Grotations_permutations(self, x):
		"""Rotate the tensor x with all rotations in group
		Args:
			x (3D filter): [out_channels,in_group,in_channels,N0,N1,N2],
			Here we DON'T use ambiguous and confusing h,w,d to discribe dimensions.
			Just use dim0,dim1,dim2 or x,y,z.
			N0,N1,N2 are lengths along dim0,dim1,dim2.
		Returns:
			list of full rotations of x [out_channels,in_group,in_channels,N0,N1,N2],....]
		"""
		return [
			self.permute_tensor(self.rotate_tensor(x, element=g, start_dim=3), element=g, dim=1) 
		  	for g in range(self.group_dim)
		]
	
	# def get_Gpermutations(self, W, kernel_shape):
	# 	"""Permute the outputs of the group convolution
	# 	Args: 
	# 		W: [N0,N1,N2,in_channel,group_dim,out_channel*group_dim]
	# 		W[:,:,:,:,:,:,i] is the i-th rotated copy of filter.

	# 	Returns:
	# 		list of the 4 rotated and permuted (in the group_dim) copies of filter
	# 	"""
	# 	return [
	# 		torch.reshape(W[gi], kernel_shape)[:, :,:,:, :, self.cayleytable[:, self.inverse_map[gi]], :]
	# 		for gi in range(self.group_dim)
	# 	]
	

	# def get_permutation_matrix(self, gi):
	# 	"""
	# 		convert the i-th column [gi*g0, gi*g1, gi*g2, ..., gi*g23] in cayley table into a permutation matrix P.
	# 		The permutation on W (filter) is to move the gj-th column to gi*gj-th column.
	# 		So, consider the gi*gj-th column of P, only the gj-th element is 1 (to select the gj-th column of W), and the other elements are 0.
	# 	"""
	# 	mat = np.zeros((self.group_dim, self.group_dim))
	# 	for j in range(self.group_dim):
	# 		mat[j, self.cayleytable[j, gi]] = 1
	# 	return mat
	

class S4_group(GroupBase):
	"""
	[[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23]
	[ 1  2  3  0  7 19 11 17 23  5  6 20 15 12 13 14  9 21 22 16 10  4  8 18]
	[ 2  3  0  1 17 16 20 21 18 19 11 10 14 15 12 13  5  4  8  9  6  7 23 22]
	[ 3  0  1  2 21  9 10  4 22 16 20  6 13 14 15 12 19  7 23  5 11 17 18  8]
	[ 4 22 16 10 14  0  7  8  9  6 13  1 17 23  5 11 12  2 21 18 19 20 15  3]
	[ 5 11 17 23  0 14  9  6  7  8  3 15 16 10  4 22  2 12 19 20 21 18  1 13]
	[ 6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23  0  1  2  3  4  5]
	[ 7  8  9  6 13  1 17 23  5 11 12  2 21 18 19 20 15  3  4 22 16 10 14  0]
	[ 8  9  6  7 23 22  2  3  0  1 17 16 20 21 18 19 11 10 14 15 12 13  5  4]
	[ 9  6  7  8  3 15 16 10  4 22  2 12 19 20 21 18  1 13  5 11 17 23  0 14]
	[10  4 22 16 20  6 13 14 15 12 19  7 23  5 11 17 18  8  3  0  1  2 21  9]
	[11 17 23  5  6 20 15 12 13 14  9 21 22 16 10  4  8 18  1  2  3  0  7 19]
	[12 13 14 15 16 17 18 19 20 21 22 23  0  1  2  3  4  5  6  7  8  9 10 11]
	[13 14 15 12 19  7 23  5 11 17 18  8  3  0  1  2 21  9 10  4 22 16 20  6]
	[14 15 12 13  5  4  8  9  6  7 23 22  2  3  0  1 17 16 20 21 18 19 11 10]
	[15 12 13 14  9 21 22 16 10  4  8 18  1  2  3  0  7 19 11 17 23  5  6 20]
	[16 10  4 22  2 12 19 20 21 18  1 13  5 11 17 23  0 14  9  6  7  8  3 15]
	[17 23  5 11 12  2 21 18 19 20 15  3  4 22 16 10 14  0  7  8  9  6 13  1]
	[18 19 20 21 22 23  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17]
	[19 20 21 18  1 13  5 11 17 23  0 14  9  6  7  8  3 15 16 10  4 22  2 12]
	[20 21 18 19 11 10 14 15 12 13  5  4  8  9  6  7 23 22  2  3  0  1 17 16]
	[21 18 19 20 15  3  4 22 16 10 14  0  7  8  9  6 13  1 17 23  5 11 12  2]
	[22 16 10  4  8 18  1  2  3  0  7 19 11 17 23  5  6 20 15 12 13 14  9 21]
	[23  5 11 17 18  8  3  0  1  2 21  9 10  4 22 16 20  6 13 14 15 12 19  7]]
	"""
	
	def get_matrix_group(self):
		"""
			Find all rotation matrices corresponding to the group elements.
			About the construction of group elements and their ordering, see get_Grotations.
		""" 
		R = []
		for i in range(4):
			for j in range(4):
				R.append(GroupBase.ijk_to_matrix(i,j,0))
			R.append(GroupBase.ijk_to_matrix(i,0,1))
			R.append(GroupBase.ijk_to_matrix(i,0,3))
		return R



class V_group(GroupBase):
	"""
	[[0,1,2,3],
	[1,0,3,2],
	[2,3,0,1],
	[3,2,1,0]]
	"""


	def get_matrix_group(self):
		"""
			Find all rotation matrices corresponding to the group elements.
			About the construction of group elements and their ordering, see get_Grotations.
		""" 
		R = [
			GroupBase.ijk_to_matrix(0,0,0),
			GroupBase.ijk_to_matrix(0,2,0),
			GroupBase.ijk_to_matrix(2,0,0),
			GroupBase.ijk_to_matrix(0,0,2)
		]
		return R
	


class T4_group(GroupBase):
	"""
	[[ 0  1  2  3  4  5  6  7  8  9 10 11]
	[ 1  2  0 11  9 10  5  3  4  8  6  7]
	[ 2  0  1  7  8  6 10 11  9  4  5  3]
	[ 3  4  5  6  7  8  0  1  2 10 11  9]
	[ 4  5  3  9 10 11  8  6  7  2  0  1]
	[ 5  3  4  1  2  0 11  9 10  7  8  6]
	[ 6  7  8  0  1  2  3  4  5 11  9 10]
	[ 7  8  6 10 11  9  2  0  1  5  3  4]
	[ 8  6  7  4  5  3  9 10 11  1  2  0]
	[ 9 10 11  8  6  7  4  5  3  0  1  2]
	[10 11  9  2  0  1  7  8  6  3  4  5]
	[11  9 10  5  3  4  1  2  0  6  7  8]]
	""" 


	def get_matrix_group(self):
		"""
			Find all rotation matrices corresponding to the group elements.
			About the construction of group elements and their ordering, see get_Grotations.
		""" 
		g0 = GroupBase.ijk_to_matrix(0,0,0)
		g1 = GroupBase.ijk_to_matrix(0,1,1)
		g3 = GroupBase.ijk_to_matrix(0,1,3)
		g9 = GroupBase.ijk_to_matrix(0,2,0)
		R = [g0, g1@g0, g1@g1@g0,
	    	g3, g1@g3, g1@g1@g3,
			g3@g3, g1@g3@g3, g1@g1@g3@g3,
			g9, g1@g9, g1@g1@g9]
		return R
	

class Z4_group(GroupBase):
	"""
	[[0,1,2,3],
	[1,0,3,2],
	[2,3,0,1],
	[3,2,1,0]]
	"""

	def get_matrix_group(self):
		"""
			Find all rotation matrices corresponding to the group elements.
			About the construction of group elements and their ordering, see get_Grotations.
		""" 
		R = [
			GroupBase.ijk_to_matrix(0,0,0),
			GroupBase.ijk_to_matrix(0,1,0),
			GroupBase.ijk_to_matrix(0,2,0),
			GroupBase.ijk_to_matrix(0,3,0)
		]
		return R
	

class D3_group(GroupBase):
	"""
	https://proofwiki.org/wiki/Definition:Dihedral_Group_D3
	Our Cayley table is a transposed version, because we consider C[i,j] as gj * gi.
	[[0 1 2 3 4 5]
	[1 2 0 5 3 4]
	[2 0 1 4 5 3]
	[3 4 5 0 1 2]
	[4 5 3 2 0 1]
	[5 3 4 1 2 0]]
	"""

	def get_matrix_group(self):
		"""
			Find all rotation matrices corresponding to the group elements.
			About the construction of group elements and their ordering, see get_Grotations.
		""" 
		a = GroupBase.ijk_to_matrix(3,0,1)
		b = GroupBase.ijk_to_matrix(2,1,0)
		R = [
			GroupBase.ijk_to_matrix(0,0,0),a,a@a,
			b,a@b,a@a@b
		]
		return R