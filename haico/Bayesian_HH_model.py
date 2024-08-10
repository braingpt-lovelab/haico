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
        self.params['muB1']   = pyro.sample('muB1', dist.Uniform(0, 1))
        self.params['muB0']   = torch.tensor(0.0)  # Fixed for the purpose of identifiability
        self.params['sigmaB'] = torch.tensor(1.0)  # Fixed for the purpose of identifiability
        self.params['rho']    = pyro.sample('rho', dist.Uniform(-1, 1))
        self.params['tau']    = torch.tensor(0.05)  # Fixed for best convergence results
        self.params['delta']  = pyro.sample('delta', dist.Uniform(0, 100))
        self.params['cutp']   = torch.sort(pyro.sample('cutp', dist.Uniform(0, 1).expand([2])))[0]

    def _model(self, classificationA, confidenceA, classificationB, confidenceB, truelabel=None, params=None):
        N, L = len(classificationA), 2
        
        # Initialize parameters if not provided
        if not params: 
            self._get_params()
            params = self.params
                
        # Extract parameters from the provided dictionary
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
        muB_i = muB0.unsqueeze(0).expand(N, L).clone()
        muB_i[torch.arange(N), truelabel_i] = muB1.type(muB0.type()) # Make sure muB1 is the same type as muB0

        # Generate correlated probability scores for each label using a bivariate normal distribution
        probscoresA_i = pyro.sample('probscoresA', dist.Normal(muB_i, sigmaB))
        probscoresB_i = pyro.sample('probscoresB', dist.Normal( \
            muB_i + rho * sigmaB * ((probscoresA_i - muB_i) / sigmaB), (1 - rho**2)**0.5 * sigmaB))

        # Compute softmax scores with temperature parameter tau
        softmaxscoresA_i = F.softmax(probscoresA_i / tau, dim=-1)
        softmaxscoresB_i = F.softmax(probscoresB_i / tau, dim=-1)

        # Generate classification for classifiers A and B
        classificationA_i = pyro.sample('classificationA', dist.Categorical(softmaxscoresA_i), obs=classificationA)
        classificationB_i = pyro.sample('classificationB', dist.Categorical(softmaxscoresB_i), obs=classificationB)

        # Generate confidence rating for classifier B from the ordered probit model
        etaA = torch.gather(probscoresA_i, probscoresA_i.dim()-1, classificationA_i.unsqueeze(1)) * delta
        pyro.sample('confidenceA', dist.OrderedLogistic(etaA.squeeze(), cutp * delta), obs=confidenceA)
        etaB = torch.gather(probscoresB_i, probscoresB_i.dim()-1, classificationB_i.unsqueeze(1)) * delta
        pyro.sample('confidenceB', dist.OrderedLogistic(etaB.squeeze(), cutp * delta), obs=confidenceB)

    def infer(self, classificationA, confidenceA, classificationB, confidenceB, truelabel=None, params=None, num_samples=10, warmup_steps=1000, num_chains=2, disable_progbar=False, group_by_chain=False):
        # Run inference
        pyro.clear_param_store()
        kernel = NUTS(self._model)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, disable_progbar=disable_progbar)
        mcmc.run(classificationA, confidenceA, classificationB, confidenceB, truelabel=truelabel, params=params)
        return mcmc.get_samples(group_by_chain=group_by_chain)