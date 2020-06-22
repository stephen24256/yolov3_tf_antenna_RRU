# import torch
# import tensorflow as tf
#
# torch.manual_seed(0)
# # a = torch.randint(1,10,(2,3,4)).type(torch.float32)
# # print(a)
# # b = torch.mean(a**2/2,dim=[1,2],keepdim=True)
# # c = torch.mean(torch.sum(a**2)/2)
# # d = torch.mean(a[0]**2/2)
# # print(b,c,d)
# # a1 = torch.tensor([9],dtype=torch.float32)
# # print(torch.rsqrt(a1))
#
# def FRNlayer(x, tau=1.0, beta =0.02, gamma=1, eps=0.001):
#     nu2 = torch.mean(x**2,dim=[2,3],keepdim=True)
#     print(nu2.shape)
#     tau = torch.tensor(tau)
#     beta = torch.tensor(beta)
#     gamma = torch.tensor(gamma)
#     eps = torch.tensor(eps)
#     x = x * torch.rsqrt(nu2 + torch.abs(eps))
#
#     return torch.max(gamma*x+beta,tau)
#
# def FRNLayer_tf(x, tau=1.0, beta =0.02, gamma=1, eps=0.001):
#
#     nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2],keepdims=True)
#
#     nu2 = tf.cast(nu2,tf.float32)
#     eps = tf.cast(eps,tf.float32)
#     x = tf.cast(x, tf.float32)
#     # print(nu2)
#     # print("----")
#     # print(eps)
#     x = x * tf.rsqrt(nu2 + tf.abs(eps))
#
#     return tf.maximum(gamma * x + beta, tau)
#
# x =  torch.randint(1,10,(1,2,3,3)).type(torch.float32)
# # tau = torch.tensor([[[[0.02]]]])
# # beta = torch.tensor([[[[0.02)
# # gamma = torch.tensor([[[[1]]]])
# # eps = torch.tensor([[[[0.001]]]])
#
# x1 = x.permute(0,2,3,1)
# eps1 = tf.constant([0.001])
# y1 = FRNLayer_tf(x1)
#
# with tf.Session() as sess:
#     print("tf",sess.run(y1))
#     print("sun",sess.run(tf.reduce_sum(y1)))
#
# y = FRNlayer(x)
# print("pytorch",y)
# print(torch.sum(y))


import torch
import torch.nn as nn

# a = (1, 2) + (1,) * (4- 2)
# print("a",a)
# a1 = (2,3)
# a2 = (3,4)
# print(a1+a2)

class FilterResponseNormNd(nn.Module):

    def __init__(self, ndim, num_features, eps=1e-6,
                 learnable_eps=False):
        """
        Input Variables:
        ----------------
            ndim: An integer indicating the number of dimensions of the expected input tensor.
            num_features: An integer indicating the number of input feature dimensions.
            eps: A scalar constant or learnable variable.
            learnable_eps: A bool value indicating whether the eps is learnable.
        """
        assert ndim in [3, 4, 5], \
            'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1,) * (ndim - 2)
        # print("shape",shape)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()

    def forward(self, x):
        avg_dims = tuple(range(2, x.dim())) # (2, 3)
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


# x =  torch.randint(1,10,(1,2,3,3)).type(torch.float32)
# frp = FilterResponseNormNd(4,2)
# out = frp(x)
# print(out,out.shape)
