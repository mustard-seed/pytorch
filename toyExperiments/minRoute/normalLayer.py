import utils
import torch


def normalLayer(coordinates, endPoints):


    '''
    Calculate the distance based on current distances

    Keyword arguments:
    coordinates-- torch.Tensor. m-by-n.
             m current variational means in the n-dimensional space
    endPoints -- torch.Tensor.
                 2-by-n. End points of the problem
    '''
    # Sanity check and obtain dimensinoality as well as number of movable points on a route
    m = coordinates.size()[0]  #Number of movable points
    n = coordinates.size()[1]  #Dimensions
    assert (endPoints.size()[0] == 2), "Number of end points does not equal to 2!"
    assert (endPoints.size()[1] == n), "Number of dimensions of the end points do not equal to that of the endpoints"

    route = torch.zeros(1,m+2,n)
    route[0][1:-1] = coordinates
    route[0][0] = endPoints[0]
    route[0][-1] = endPoints[1]
    routeDistance = utils.routeL1(route)
    return routeDistance
