import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['forward_hook', 'Clone', 'Add', 'Cat', 'ReLU', 'GELU', 'Dropout', 'BatchNorm2d', 'Linear', 'MaxPool2d',
           'AdaptiveAvgPool2d', 'AvgPool2d', 'Conv2d', 'Sequential', 'safe_divide', 'einsum', 'Softmax', 'IndexSelect',
           'LayerNorm', 'AddEye', 'Identity',
           'PatchEmbed', 'ClassifierHead', 'Mlp']


def safe_divide(a, b):
    den = b.clamp(min=1e-9) + b.clamp(max=1e-9)
    den = den + den.eq(0).type(den.type()) * 1e-9
    return a / den * b.ne(0).type(b.type())


def forward_hook(self, input, output):
    if type(input[0]) in (list, tuple):
        self.X = []
        for i in input[0]:
            x = i.detach()
            x.requires_grad = True
            self.X.append(x)
    else:
        self.X = input[0].detach()
        self.X.requires_grad = True

    self.Y = output


def backward_hook(self, grad_input, grad_output):
    self.grad_input = grad_input
    self.grad_output = grad_output


class RelProp(nn.Module):
    def __init__(self):
        super(RelProp, self).__init__()
        # if not self.training:
        self.register_forward_hook(forward_hook)

    def gradprop(self, Z, X, S):
        C = torch.autograd.grad(Z, X, S, retain_graph=True)
        return C

    def relprop(self, R, alpha):
        return R

class RelPropSimple(RelProp):
    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs

class AddEye(RelPropSimple):
    # input of shape B, C, seq_len, seq_len
    def forward(self, input):
        return input + torch.eye(input.shape[2]).expand_as(input).to(input.device)

class ReLU(nn.ReLU, RelProp):
    pass

class GELU(nn.GELU, RelProp):
    pass

class Identity(nn.Identity, RelProp):
    pass

class Softmax(nn.Softmax, RelProp):
    pass

class LayerNorm(nn.LayerNorm, RelProp):
    pass

class Dropout(nn.Dropout, RelProp):
    pass

class Identity(nn.Identity, RelProp):
    pass


class MaxPool2d(nn.MaxPool2d, RelPropSimple):
    pass


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, RelPropSimple):
    pass


class AvgPool2d(nn.AvgPool2d, RelPropSimple):
    pass


class Add(RelPropSimple):
    def forward(self, inputs):
        return torch.add(*inputs)

    def relprop(self, R, alpha):
        Z = self.forward(self.X)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        a = self.X[0] * C[0]
        b = self.X[1] * C[1]

        a_sum = a.sum()
        b_sum = b.sum()

        a_fact = safe_divide(a_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()
        b_fact = safe_divide(b_sum.abs(), a_sum.abs() + b_sum.abs()) * R.sum()

        a = a * safe_divide(a_fact, a.sum())
        b = b * safe_divide(b_fact, b.sum())

        outputs = [a, b]

        return outputs

class einsum(RelPropSimple):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation
    def forward(self, *operands):
        return torch.einsum(self.equation, *operands)

class IndexSelect(RelProp):
    def forward(self, inputs, dim, indices):
        self.__setattr__('dim', dim)
        self.__setattr__('indices', indices)

        return torch.index_select(inputs, dim, indices)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim, self.indices)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        if torch.is_tensor(self.X) == False:
            outputs = []
            outputs.append(self.X[0] * C[0])
            outputs.append(self.X[1] * C[1])
        else:
            outputs = self.X * (C[0])
        return outputs



class Clone(RelProp):
    def forward(self, input, num):
        self.__setattr__('num', num)
        outputs = []
        for _ in range(num):
            outputs.append(input)

        return outputs

    def relprop(self, R, alpha):
        Z = []
        for _ in range(self.num):
            Z.append(self.X)
        S = [safe_divide(r, z) for r, z in zip(R, Z)]
        C = self.gradprop(Z, self.X, S)[0]

        R = self.X * C

        return R

class Cat(RelProp):
    def forward(self, inputs, dim):
        self.__setattr__('dim', dim)
        return torch.cat(inputs, dim)

    def relprop(self, R, alpha):
        Z = self.forward(self.X, self.dim)
        S = safe_divide(R, Z)
        C = self.gradprop(Z, self.X, S)

        outputs = []
        for x, c in zip(self.X, C):
            outputs.append(x * c)

        return outputs


class Flatten(nn.Flatten, RelProp):
    def relprop(self, cam, **kwargs):
        # Implement the relevance propagation for Flatten operation
        # The CAM needs to be reshaped back to the pre-flattened shape
        return cam.reshape(self.X.shape)
    

class Sequential(nn.Sequential):
    def relprop(self, R, alpha):
        for m in reversed(self._modules.values()):
            R = m.relprop(R, alpha)
        return R

class BatchNorm2d(nn.BatchNorm2d, RelProp):
    def relprop(self, R, alpha):
        X = self.X
        beta = 1 - alpha
        weight = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3) / (
            (self.running_var.unsqueeze(0).unsqueeze(2).unsqueeze(3).pow(2) + self.eps).pow(0.5))
        Z = X * weight + 1e-9
        S = R / Z
        Ca = S * weight
        R = self.X * (Ca)
        return R


class Linear(nn.Linear, RelProp):
    def relprop(self, R, alpha):
        beta = alpha - 1
        pw = torch.clamp(self.weight, min=0)
        nw = torch.clamp(self.weight, max=0)
        px = torch.clamp(self.X, min=0)
        nx = torch.clamp(self.X, max=0)

        def f(w1, w2, x1, x2):
            Z1 = F.linear(x1, w1)
            Z2 = F.linear(x2, w2)
            S1 = safe_divide(R, Z1 + Z2)
            S2 = safe_divide(R, Z1 + Z2)
            C1 = x1 * torch.autograd.grad(Z1, x1, S1)[0]
            C2 = x2 * torch.autograd.grad(Z2, x2, S2)[0]

            return C1 + C2

        activator_relevances = f(pw, nw, px, nx)
        inhibitor_relevances = f(nw, pw, px, nx)

        R = alpha * activator_relevances - beta * inhibitor_relevances

        return R


class Conv2d(nn.Conv2d, RelProp):
    def gradprop2(self, DY, weight):
        Z = self.forward(self.X)

        output_padding = self.X.size()[2] - (
                (Z.size()[2] - 1) * self.stride[0] - 2 * self.padding[0] + self.kernel_size[0])

        return F.conv_transpose2d(DY, weight, stride=self.stride, padding=self.padding, output_padding=output_padding)

    def relprop(self, R, alpha):
        if self.X.shape[1] == 3:
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            X = self.X
            L = self.X * 0 + \
                torch.min(torch.min(torch.min(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            H = self.X * 0 + \
                torch.max(torch.max(torch.max(self.X, dim=1, keepdim=True)[0], dim=2, keepdim=True)[0], dim=3,
                          keepdim=True)[0]
            Za = torch.conv2d(X, self.weight, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(L, pw, bias=None, stride=self.stride, padding=self.padding) - \
                 torch.conv2d(H, nw, bias=None, stride=self.stride, padding=self.padding) + 1e-9

            S = R / Za
            C = X * self.gradprop2(S, self.weight) - L * self.gradprop2(S, pw) - H * self.gradprop2(S, nw)
            R = C
        else:
            beta = alpha - 1
            pw = torch.clamp(self.weight, min=0)
            nw = torch.clamp(self.weight, max=0)
            px = torch.clamp(self.X, min=0)
            nx = torch.clamp(self.X, max=0)

            def f(w1, w2, x1, x2):
                Z1 = F.conv2d(x1, w1, bias=None, stride=self.stride, padding=self.padding)
                Z2 = F.conv2d(x2, w2, bias=None, stride=self.stride, padding=self.padding)
                S1 = safe_divide(R, Z1)
                S2 = safe_divide(R, Z2)
                C1 = x1 * self.gradprop(Z1, x1, S1)[0]
                C2 = x2 * self.gradprop(Z2, x2, S2)[0]
                return C1 + C2

            activator_relevances = f(pw, nw, px, nx)
            inhibitor_relevances = f(nw, pw, px, nx)

            R = alpha * activator_relevances - beta * inhibitor_relevances
        return R

from typing import Callable, List, Optional, Tuple, Union
from timm.layers.format import Format, nchw_to
from timm.layers.helpers import to_2tuple
from timm.layers.trace_utils import _assert

class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW

        self.proj = Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
            _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x
    
    def relprop(self, cam, **kwargs):
        cam = self.norm.relprop(cam, **kwargs)
        if self.flatten:
            cam = cam.transpose(1, 2)
            cam = cam.reshape(cam.shape[0], cam.shape[1],
                        (self.img_size[0] // self.patch_size[0]), (self.img_size[1] // self.patch_size[1]))
        else:
            raise Exception("There is a shape conversion problem in PatchEmbed Class")
            
        return self.proj.relprop(cam, **kwargs)




class ClassifierHead(nn.Module):
    """Classifier head w/ configurable global pooling and dropout."""

    def __init__(
            self,
            in_features: int,
            num_classes: int,
            pool_type: str = 'avg',
            drop_rate: float = 0.,
            use_conv: bool = False,
            input_fmt: str = 'NCHW',
    ):
        """
        Args:
            in_features: The number of input features.
            num_classes:  The number of classes for the final classifier layer (output).
            pool_type: Global pooling type, pooling disabled if empty string ('').
            drop_rate: Pre-classifier dropout rate.
        """
        super(ClassifierHead, self).__init__()
        self.in_features = in_features
        self.use_conv = use_conv
        self.input_fmt = input_fmt

        global_pool, fc = create_classifier(
            in_features,
            num_classes,
            pool_type,
            use_conv=use_conv,
            input_fmt=input_fmt,
        )
        self.global_pool = global_pool
        self.drop = Dropout(drop_rate)
        self.fc = fc
        self.flatten = Flatten(1) if use_conv and pool_type else Identity()

    def reset(self, num_classes, pool_type=None):
        if pool_type is not None and pool_type != self.global_pool.pool_type:
            self.global_pool, self.fc = create_classifier(
                self.in_features,
                num_classes,
                pool_type=pool_type,
                use_conv=self.use_conv,
                input_fmt=self.input_fmt,
            )
            self.flatten = Flatten(1) if self.use_conv and pool_type else Identity()
        else:
            num_pooled_features = self.in_features * self.global_pool.feat_mult()
            self.fc = _create_fc(
                num_pooled_features,
                num_classes,
                use_conv=self.use_conv,
            )

    def forward(self, x, pre_logits: bool = False):
        self.pre_logits = pre_logits
        x = self.global_pool(x)
        x = self.drop(x)
        if pre_logits:
            return self.flatten(x)
        x = self.fc(x)
        return self.flatten(x)
    
    def relprop(self, cam, **kwargs):
        cam = self.flatten.relprop(cam, **kwargs)
        if not self.pre_logits:
            cam = self.fc.relprop(cam, **kwargs)
        cam = self.drop.relprop(cam, **kwargs)
        cam = self.global_pool.relprop(cam, **kwargs)
    
        return cam


def create_classifier(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: str = 'NCHW',
        drop_rate: Optional[float] = None,
):
    global_pool, num_pooled_features = _create_pool(
        num_features,
        num_classes,
        pool_type,
        use_conv=use_conv,
        input_fmt=input_fmt,
    )
    fc = _create_fc(
        num_pooled_features,
        num_classes,
        use_conv=use_conv,
    )
    if drop_rate is not None:
        dropout = Dropout(drop_rate)
        return global_pool, dropout, fc
    return global_pool, fc

def _create_pool(
        num_features: int,
        num_classes: int,
        pool_type: str = 'avg',
        use_conv: bool = False,
        input_fmt: Optional[str] = None,
):
    flatten_in_pool = not use_conv  # flatten when we use a Linear layer after pooling
    if not pool_type:
        assert num_classes == 0 or use_conv,\
            'Pooling can only be disabled if classifier is also removed or conv classifier is used'
        flatten_in_pool = False  # disable flattening if pooling is pass-through (no pooling)
    global_pool = SelectAdaptivePool2d(
        pool_type=pool_type,
        flatten=flatten_in_pool,
        input_fmt=input_fmt,
    )
    num_pooled_features = num_features * global_pool.feat_mult()
    return global_pool, num_pooled_features




def _create_fc(num_features, num_classes, use_conv=False):
    if num_classes <= 0:
        fc = Identity()  # pass-through (no classifier)
    elif use_conv:
        fc = Conv2d(num_features, num_classes, 1, bias=True)
    else:
        fc = Linear(num_features, num_classes, bias=True)
    return fc



_int_tuple_2_t = Union[int, Tuple[int, int]]
class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(
            self,
            output_size: _int_tuple_2_t = 1,
            pool_type: str = 'fast',
            flatten: bool = False,
            input_fmt: str = 'NCHW',
    ):
        super(SelectAdaptivePool2d, self).__init__()
        assert input_fmt in ('NCHW', 'NHWC')
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        if not pool_type:
            self.pool = Identity()  # pass through
            self.flatten = nn.Flatten(1) if flatten else Identity()
        elif pool_type.startswith('fast') or input_fmt != 'NCHW':
            assert output_size == 1, 'Fast pooling and non NCHW input formats require output_size == 1.'
            self.pool = FastAdaptiveAvgPool(flatten, input_fmt=input_fmt)
            self.flatten = Identity()
        else:
            assert input_fmt == 'NCHW'
            self.pool = AdaptiveAvgPool2d(output_size)
            self.flatten = nn.Flatten(1) if flatten else Identity()

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def feat_mult(self):
        return adaptive_pool_feat_mult(self.pool_type)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'
    
    def relprop(self, cam, **kwargs):
        cam = self.flatten.relprop(cam, **kwargs)
        cam = self.pool.relprop(cam, **kwargs)
    
        return cam

from timm.layers.format import get_spatial_dim

class FastAdaptiveAvgPool_ori(nn.Module):
    def __init__(self, flatten: bool = False, input_fmt: F = 'NCHW'):
        super(FastAdaptiveAvgPool_ori, self).__init__()
        self.flatten = flatten
        self.dim = get_spatial_dim(input_fmt)

    def forward(self, x):
        return x.mean(self.dim, keepdim=not self.flatten) 


class FastAdaptiveAvgPool(FastAdaptiveAvgPool_ori, RelPropSimple):
    pass



def adaptive_pool_feat_mult(pool_type='avg'):
    if pool_type.endswith('catavgmax'):
        return 2
    else:
        return 1
    


from functools import partial

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(Conv2d, kernel_size=1) if use_conv else Linear

        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def relprop(self, cam, **kwargs):
        cam = self.drop2.relprop(cam, **kwargs)
        cam = self.fc2.relprop(cam, **kwargs)
        cam = self.drop1.relprop(cam, **kwargs)
        cam = self.act.relprop(cam, **kwargs)
        cam = self.fc1.relprop(cam, **kwargs)
    
        return cam