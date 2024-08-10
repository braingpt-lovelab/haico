import torch
# import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

class BayesianCombinationModel:
    def __init__(self):
        self.params = {} # Store parameters
        self.trace = {}  # Store trace
        
    def _get_params(self):
        # Define priors and constants
        self.params['muA1']   = pyro.sample('muA1', dist.Normal(0, 10))
        self.params['muA0']   = pyro.sample('muA0', dist.Normal(0, 10))
        self.params['sigmaA'] = pyro.sample('sigmaA', dist.Uniform(0, 15))
        self.params['muB1']   = pyro.sample('muB1', dist.Normal(0, 10))
        self.params['muB0']   = pyro.sample('muB0', dist.Normal(0, 10))
        self.params['sigmaB'] = pyro.sample('sigmaB', dist.Uniform(0, 15))
        self.params['rho']    = pyro.sample('rho', dist.Uniform(-1, 1))

    def _model(self, probscoresA, probscoresB, truelabel=None, params=None):
        N, L = probscoresA.shape
        
        # Initialize parameters if not provided
        if not params: 
            self._get_params()
            params = self.params
                
        # Extract parameters from the provided dictionary
        muA1   = params['muA1']
        muA0   = params['muA0']
        sigmaA = params['sigmaA']
        muB1   = params['muB1']
        muB0   = params['muB0']
        sigmaB = params['sigmaB']
        rho    = params['rho']
        
        # Define true labels if observed or latent
        if truelabel is not None:  
            truelabel_i = pyro.sample('truelabel', dist.Categorical(probs=torch.ones(L)).expand([N]), obs=truelabel)
        else:
            # Learn label probabilities using a Dirichlet distribution with uniform prior
            labelprob = pyro.sample('labelprob', dist.Dirichlet(torch.ones(N,L)))
            truelabel_i = torch.argmax(labelprob, dim=-1).squeeze()

        # Set the means based on true label
        muA_i = muA0.unsqueeze(0).expand(N, L).clone()
        muB_i = muB0.unsqueeze(0).expand(N, L).clone()
        muA_i[torch.arange(N), truelabel_i] = muA1.type(muA0.type()) # Make sure muA1 is the same type as muA0
        muB_i[torch.arange(N), truelabel_i] = muB1.type(muB0.type()) # Make sure muB1 is the same type as muB0

        # Generate correlated probability scores for each label using a bivariate normal distribution
        probscoresA_i = pyro.sample('probscoresA', dist.Normal(muA_i, sigmaA), obs=probscoresA)
        pyro.sample('probscoresB', dist.Normal(muB_i + rho * sigmaB * ((probscoresA_i - muA_i) / sigmaA), \
            (1 - rho**2)**0.5 * sigmaB), obs=probscoresB)
        
    def infer(self, probscoresA, probscoresB, truelabel=None, params=None, num_samples=10, warmup_steps=1000, num_chains=2, disable_progbar=False, group_by_chain=False):
        # Run inference
        pyro.clear_param_store()
        kernel = NUTS(self._model)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, disable_progbar=disable_progbar)
        mcmc.run(probscoresA, probscoresB, truelabel=truelabel, params=params)
        return mcmc.get_samples(group_by_chain=group_by_chain)