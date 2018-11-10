import torch
import numpy as np

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
    
def parameterGenerator(parameters):
    '''
    Produce an iteratable list of parameters
    
    Keyword arguments:
    parameters: list of parameters
    '''
    for parameter in parameters:
        yield parameter

def fitnessShaping(fitnessValues):
    '''
    Performs fitness shaping according to Wierstra et al 2014, page 959
    :param fitnessValues: Pytorch tensor of fitness values
    :return: utilityValues: Pytorch tesnr of utility values
    '''

    assert (fitnessValues.dim()==1), "fitnessShaping: Number of dimensions of the fitness list is not 1!"
    size = fitnessValues.shape[0]
    sorted, indices = torch.sort(fitnessValues, descending=True)
    utilityValues = torch.zeros_like(fitnessValues, dtype=torch.float)
    denominator = 0
    for idx, value in enumerate(indices):
        local = torch.max(torch.tensor(0, dtype=torch.float), torch.tensor(np.log(float(size)/2 + 1)) - torch.tensor(np.log(float(idx+1))))
        utilityValues[value] = local
        denominator += local
    utilityValues = utilityValues / denominator - (1/float(size))

    return utilityValues

