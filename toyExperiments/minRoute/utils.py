import torch

def routeL1 (coordinates):
    """Calculate the L2-norm of a list of routes in n-dimension
    
    Keyword arguments:
    coordinates -- PyTorch Tensor. Dimension is s-by-m-by-n. s is the number routes, m is the number of points per route.
     
    """
    # Check the number of dimensions
    numDim = coordinates.dim();
    assert (numDim == 3), "Number of input dimension is not 3!"

    numPoints = coordinates.size()[1];
    l2Norm = (coordinates.split(dim=1, split_size=[1,numPoints-1])[-1] - coordinates.split(dim=1, split_size=[numPoints-1,1])[0]).pow(2).sum(dim=(2)).pow(.5).sum(dim=(1))
    return l2Norm
    
