import torch
import torch.nn.functional as F

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
        self.params['muB0']   = torch.tensor(0.0) # Fixed for the purpose of identifiability
        self.params['sigmaB'] = torch.tensor(1.0) # Fixed for the purpose of identifiability
        self.params['rho']    = pyro.sample('rho', dist.Uniform(-1, 1))
        self.params['tau']    = torch.tensor(0.05) # Fixed for best convergence results
        self.params['delta']  = pyro.sample('delta', dist.Uniform(0, 100))
        self.params['cutp']   = torch.sort(pyro.sample('cutp', dist.Uniform(0, 1).expand([2])))[0]

    def _model(self, probscoresA, classificationB, confidenceB, truelabel=None, params=None):
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
        tau    = params['tau']
        delta  = params['delta']
        cutp   = params['cutp']
        
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
        probscoresB_i = pyro.sample('probscoresB', dist.Normal( \
            muB_i + rho * sigmaB * ((probscoresA_i - muA_i) / sigmaA), (1 - rho**2)**0.5 * sigmaB))
        
        # Compute softmax scores with temperature parameter tau
        softmaxscores_i = F.softmax(probscoresB_i / tau, dim=-1)

        # Generate classification for classifier B
        classificationB_i = pyro.sample('classificationB', dist.Categorical(softmaxscores_i), obs=classificationB)

        # Generate confidence rating for classifier B from the ordered probit model
        eta = torch.gather(probscoresB_i, probscoresB_i.dim()-1, classificationB_i.unsqueeze(1)) * delta
        pyro.sample('confidenceB', dist.OrderedLogistic(eta.squeeze(), cutp * delta), obs=confidenceB)

    def infer(self, probscoresA, classificationB, confidenceB, truelabel=None, params=None, num_samples=10, warmup_steps=1000, num_chains=2, disable_progbar=False, group_by_chain=False):
        # Run inference
        pyro.clear_param_store()
        kernel = NUTS(self._model)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, disable_progbar=disable_progbar)
        mcmc.run(probscoresA, classificationB, confidenceB, truelabel=truelabel, params=params)
        return mcmc.get_samples(group_by_chain=group_by_chain)