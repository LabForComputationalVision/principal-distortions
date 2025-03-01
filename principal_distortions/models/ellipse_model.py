import torch as ch
import numpy as np

class EllipseModel(ch.nn.Module):
    """
    Pytorch module that simulates 2D models. Used for demos. 
    """
    def __init__(self, theta, eig_1, eig_2,
                 device='cuda', model_name=None,
                 ):
        """
        Inputs:
            theta (float): parameterizes the angle of the FI matrix
            eig_1 (float) : parameterizes the first eigenvector of the FI matrix
            eig_2 (float): parameterizes the second eigenvector o the FI matrix
            device (cuda or cpu): location to place the model # TODO make everything buffers 
            model_name (str): Name for the model, set based on inputs if not specified.
        """
        super().__init__()
        self.theta = theta
        self.eig_1 = eig_1
        self.eig_2 = eig_2
        if model_name is None:
            self.model_name = f'Theta:{self.theta:0.2f}|Eig1:{self.eig_1:0.2f}|Eig2:{self.eig_2:0.2f}'
        else:
            self.model_name = model_name
        self.FI = sim_fisher_mat(self.theta,
                                 self.eig_1,
                                 self.eig_2)
        self.A = ch.tensor(sim_A(self.theta,
                                 self.eig_1,
                                 self.eig_2,
                                ), dtype=ch.float32,
                           device=device)
    
    def forward(self, x):
        return ch.matmul(self.A, x)

    def make_sensitivity_ellipse(self, num_steps=200):
        """
        Makes an ellipse of value sqrt(x.T * FI * x) * x. This computes the 
        sensitivity in the direction x, so that we can interpret the norm along
        each direction as the sensitivity. 
        """
        t = np.linspace(0,2*np.pi,200)
        x = np.vstack([np.cos(t), np.sin(t)])
        ellipse = [1/np.sqrt(np.matmul(x_tmp.T, # TODO: Vectorize this. 
                                       np.matmul(np.matmul(self.A.T.detach().cpu().numpy(),
                                                           self.A.detach().cpu().numpy()),
                                                 x_tmp))) * x_tmp for x_tmp in x.T]
        return np.array(ellipse), x


def sim_fisher_mat(theta, eig_1, eig_2):
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    lamb = np.diag([eig_1, eig_2])
    return np.matmul(np.matmul(U, lamb), U.T)

def sim_A(theta, eig_1, eig_2):
    U = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    d_lamb = np.diag([np.sqrt(eig_1), np.sqrt(eig_2)])
    return np.matmul(d_lamb, U.T)

