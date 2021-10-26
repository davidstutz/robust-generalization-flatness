# https://github.com/tomgoldstein/loss-landscape/blob/64ef4d57f8dabe79b57a637819c44e48eda98f33/hess_vec_prod.py
import torch
import time
import numpy as np
from torch.autograd import Variable
from scipy.sparse.linalg import LinearOperator, eigsh
import common.torch


################################################################################
#                              Supporting Functions
################################################################################
def npvec_to_tensorlist(vec, params):
    """ Convert a numpy vector to a list of tensor with the same dimensions as params

        Args:
            vec: a 1D numpy vector
            params: a list of parameters from net

        Returns:
            rval: a list of tensors with the same shape as params
    """
    loc = 0
    rval = []
    for p in params:
        numel = p.data.numel()
        rval.append(torch.from_numpy(vec[loc:loc+numel]).view(p.data.shape).float())
        loc += numel
    assert loc == vec.size, 'The vector has more elements than the net has parameters'
    return rval


def gradtensor_to_npvec(net, include_bn=False):
    """ Extract gradients from net, and return a concatenated numpy vector.

        Args:
            net: trained model
            include_bn: If include_bn, then gradients w.r.t. BN parameters and bias
            values are also included. Otherwise only gradients with dim > 1 are considered.

        Returns:
            a concatenated numpy vector containing all gradients
    """
    filter = lambda p: include_bn or len(p.data.size()) > 1
    return np.concatenate([p.grad.data.cpu().numpy().ravel() for p in net.parameters() if filter(p)])


################################################################################
#                  For computing Hessian-vector products
################################################################################
def eval_hess_vec_prod(vec, params, net, criterion, dataloader, use_cuda=False):
    """
    Evaluate product of the Hessian of the loss function with a direction vector "vec".
    The product result is saved in the grad of net.

    Args:
        vec: a list of tensor with the same dimensions as "params".
        params: the parameter list of the net (ignoring biases and BN parameters).
        net: model with trained parameters.
        criterion: loss function.
        dataloader: dataloader for the dataset.
        use_cuda: use GPU.
    """

    if use_cuda:
        net.cuda()
        vec = [v.cuda() for v in vec]

    net.eval()
    net.zero_grad() # clears grad for every parameter in the net

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = Variable(inputs), Variable(targets)
        inputs = inputs.permute(0, 3, 1, 2)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        grad_f = torch.autograd.grad(loss, inputs=params, create_graph=True)

        # Compute inner product of gradient with the direction vector
        prod = Variable(torch.zeros(1)).type(type(grad_f[0].data))
        for (g, v) in zip(grad_f, vec):
            prod = prod + (g * v).cpu().sum()

        # Compute the Hessian-vector product, H*v
        # prod.backward() computes dprod/dparams for every parameter in params and
        # accumulate the gradients into the params.grad attributes
        prod.backward()


def min_max_k_hessian_eigs(net, dataloader, criterion, k=10, rank=0, use_cuda=False, verbose=False):
    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.

        Args:
            net: the trained model.
            dataloader: dataloader for the dataset, may use a subset of it.
            criterion: loss function.
            rank: rank of the working node.
            use_cuda: use GPU

        Returns:
            maxeig: max eigenvalue
            mineig: min eigenvalue
            hess_vec_prod.count: number of iterations for calculating max and min eigenvalues
    """
    assert k > 1

    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, dataloader, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0: print("   Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(net)

    hess_vec_prod.count = 0
    A = LinearOperator((N, N), matvec=hess_vec_prod)
    eigvals = eigsh(A, k=k, tol=1e-2, return_eigenvectors=False, which='BE')
    return eigvals, hess_vec_prod.count


def min_max_k_hessian_eigsv(net, dataloader, criterion, k=10, rank=0, use_cuda=False, verbose=False):
    """
        Compute the largest and the smallest eigenvalues of the Hessian marix.

        Args:
            net: the trained model.
            dataloader: dataloader for the dataset, may use a subset of it.
            criterion: loss function.
            rank: rank of the working node.
            use_cuda: use GPU

        Returns:
            maxeig: max eigenvalue
            mineig: min eigenvalue
            hess_vec_prod.count: number of iterations for calculating max and min eigenvalues
    """
    assert k > 1

    params = [p for p in net.parameters() if len(p.size()) > 1]
    N = sum(p.numel() for p in params)

    def hess_vec_prod(vec):
        hess_vec_prod.count += 1  # simulates a static variable
        vec = npvec_to_tensorlist(vec, params)
        start_time = time.time()
        eval_hess_vec_prod(vec, params, net, criterion, dataloader, use_cuda)
        prod_time = time.time() - start_time
        if verbose and rank == 0: print("   Iter: %d  time: %f" % (hess_vec_prod.count, prod_time))
        return gradtensor_to_npvec(net)

    hess_vec_prod.count = 0
    A = LinearOperator((N, N), matvec=hess_vec_prod)
    eigvals, eigvecs = eigsh(A, k=k, tol=1e-2, which='BE')

    eigen_models = []
    for i in range(eigvecs.shape[1]):
        eigen_params = npvec_to_tensorlist(eigvecs[:, i], params)
        eigen_model = common.torch.clone(net)
        eigen_model_params = [p for p in list(eigen_model.parameters()) if len(p.size()) > 1]

        for j in range(len(eigen_params)):
            eigen_model_params[j].data = eigen_params[j].data

        eigen_models.append(eigen_model)

    return eigvals, eigen_models, hess_vec_prod.count