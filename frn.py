import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn import ReLU, LeakyReLU


class TLU(nn.Module):
    def __init__(self, num_features):
        """max(y, tau) = max(y - tau, 0) + tau = ReLU(y - tau) + tau"""
        super(TLU, self).__init__()
        self.num_features = num_features
        self.tau = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.tau)

    def extra_repr(self):
        return 'num_features={num_features}'.format(**self.__dict__)

    def forward(self, x):
        return torch.max(x, self.tau)


class FRN(nn.Module):
    def __init__(self, num_features, eps=1e-6, is_eps_leanable=False, momentum=0, efficient=True):
        """
        weight = gamma, bias = beta
        beta, gamma:
            Variables of shape [1, 1, 1, C]. if TensorFlow
            Variables of shape [1, C, 1, 1]. if PyTorch
        eps: A scalar constant or learnable variable.
        momentum: not used
        """
        super(FRN, self).__init__()

        self.num_features = num_features
        self.init_eps = eps
        self.is_eps_leanable = is_eps_leanable

        self.weight = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        self.bias = nn.parameter.Parameter(torch.Tensor(1, num_features, 1, 1), requires_grad=True)
        if is_eps_leanable:
            self.eps = nn.parameter.Parameter(torch.Tensor(1), requires_grad=True)
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()
        self.efficient = efficient

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)
        if self.is_eps_leanable:
            nn.init.constant_(self.eps, self.init_eps)

    def extra_repr(self):
        return 'num_features={num_features}, eps={init_eps}'.format(**self.__dict__)

    def forward(self, x):
        """
        0, 1, 2, 3 -> (B, H, W, C) in TensorFlow
        0, 1, 2, 3 -> (B, C, H, W) in PyTorch
        TensorFlow code
            nu2 = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
            x = x * tf.rsqrt(nu2 + tf.abs(eps))
            # This Code include TLU function max(y, tau)
            return tf.maximum(gamma * x + beta, tau)
        """

        if self.efficient and self.training:
            return FRNImplementation.apply(x, self.weight, self.bias)

        # Compute the mean norm of activations per channel.
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x_hat = x * torch.rsqrt(nu2 + self.eps.abs())

        # Scale and Bias
        f = self.weight * x_hat + self.bias
        return f


class FRNImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)

        # Perform FRN.
        x_hat = x * torch.rsqrt(nu2 + 1e-6)        # TODO : eps? abs?
        ctx.save_for_backward(x, weight, bias)

        # Scale and Bias
        f = weight * x_hat + bias
        return f

    @staticmethod
    def backward(ctx, grad_output):
        x, w, b = ctx.saved_tensors

        # x_hat
        nu2 = x.pow(2).mean(dim=[2, 3], keepdim=True)
        rsqrt = torch.rsqrt(nu2 + 1e-6)
        x_hat = x * rsqrt

        # grad_w, grad_b
        grad_b = torch.sum(grad_output, dim=(0, 2, 3), keepdim=True)
        grad_w = torch.sum(grad_output * x_hat, dim=(0, 2, 3), keepdim=True)
        grad_f = torch.sum(w * grad_output, dim=(0, ), keepdim=True)

        # grad_in
        # I = torch.eye(x.shape[2]).unsqueeze(0).unsqueeze(0).to(x.device)
        I = torch.ones((x.shape[2], x.shape[3])).unsqueeze(0).unsqueeze(0).to(x.device)
        B, C, W, H = x_hat.shape
        N = W * H
        # bmm = torch.bmm(x_hat.view(B*C, W, H), torch.transpose(x_hat, 2, 3).view(B*C, H, W)) / N
        # bmm = x_hat * torch.transpose(x_hat, 2, 3) / N

        x_flatten = x_hat.view(B*C, 1, N)
        bmm = torch.bmm(x_flatten, torch.transpose(x_flatten, 1, 2)) / N
        bmm = bmm.view(B, C, 1, 1)

        mat = (I - bmm)
        grad_in = rsqrt * mat * grad_f
        return grad_in, grad_w, grad_b


if __name__ == '__main__':
    x = torch.rand((1, 10, 3, 3), requires_grad=True)
    frn = FRN(10, efficient=False)

    y = frn(x)
    y_sum = y.sum()
    y_sum.backward()

    print(frn.weight.grad[0, 0], frn.weight.grad.shape)
    print(frn.bias.grad[0, 0], frn.bias.grad.shape)
    print(x.grad[0, 0], x.grad.shape)

    frn = FRN(10, efficient=True)

    y = frn(x)
    y_sum = y.sum()
    y_sum.backward()

    print(frn.weight.grad[0, 0], frn.weight.grad.shape)
    print(frn.bias.grad[0, 0], frn.bias.grad.shape)
    print(x.grad[0, 0], x.grad.shape)
    pass
