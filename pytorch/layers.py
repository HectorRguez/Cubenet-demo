import sys
import torch
from groups import GroupBase

class GConv3D(torch.nn.Module):
	def __init__(self, group : GroupBase, in_group_dim, in_channels, out_channels, kernel_size, stride=1, padding=0):
		"""
		Group Equivariant Convolution Layer with 3D convolution.

		Args:
			in_channels (int): Number of input channels.
			out_channels (int): Number of output channels.
			kernel_size (int or tuple): Size of the 3D convolutional kernel.
			group_elements (list of tuples): List of group transformations (e.g., rotation angles).
			stride (int or tuple): Convolution stride.
			padding (int or tuple): Padding size.
		"""
		super(GConv3D, self).__init__()
		self.group = group
		# if group == "V":
		# 	from groups import V_group
		# 	self.group = V_group()
		# elif group == "S4":
		# 	from groups import S4_group
		# 	self.group = S4_group()
		# elif group == "T4":
		# 	from groups import T4_group
		# 	self.group = T4_group()
		# elif group == "Z4":
		# 	from groups import Z4_group
		# 	self.group = Z4_group()
		# elif group == "D3":
		# 	from groups import D3_group
		# 	self.group = D3_group()
		# else:
		# 	print("Group is not recognized")
		# 	sys.exit(-1)
		self.group_dim = self.group.group_dim
		self.cayley = self.group.cayleytable
	
		self.in_channels = in_channels
		self.in_group_dim = in_group_dim
		assert in_group_dim == 1 or in_group_dim == self.group_dim, "in_group_dim must be 1 or group_dim"

		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding

		def _to_tuple(value):
			if isinstance(value, int):
				return (value, value, value)
			return value
		
		# Define a learnable kernel for one group transformation
		self.kernel = torch.nn.Parameter(
			torch.empty(out_channels, in_group_dim, in_channels, *_to_tuple(kernel_size), requires_grad=True)
		)
		# initialize
		torch.nn.init.xavier_normal_(self.kernel)


	def prepare_filters(self):
		"""
		Apply the group transformation to the filter.
		kernel (torch.Tensor): shape [out_channels,in_group_dim,in_channels,N0,N1,N2].
		Returns:
			torch.Tensor: Transformed kernel. shape [out_channels*out_group,in_channels*in_group_dim,N0,N1,N2].
		"""
		if self.in_group_dim == 1:
			WN = self.group.get_Grotations(self.kernel)
		elif self.in_group_dim == self.group_dim:
			WN = self.group.get_Grotations_permutations(self.kernel)
		# a list of [out_channels,in_group_dim,in_channels,N0,N1,N2]
		WN = torch.cat(WN, dim=0)
		WN = torch.flatten(WN, start_dim=1, end_dim=2)
		return WN


	def forward(self, x) -> torch.Tensor:
		"""
		Forward pass of the GConv layer.

		Args:
			x (torch.Tensor): Input tensor of shape [batch_size, in_group_dim,in_channels, N0,N1,N2].

		Returns:
			torch.Tensor: Output tensor after group equivariant convolution.
			[batch_size, out_group,out_channels, N0,N1,N2].
		"""
		batch_size = x.size(0)
		xN = torch.flatten(x, start_dim=1, end_dim=2)
		WN = self.prepare_filters()
		yN = torch.nn.functional.conv3d(xN, WN, stride=self.stride, padding=self.padding)
		y = yN.reshape(batch_size, self.group_dim, self.out_channels, *yN.size()[2:])
		return y