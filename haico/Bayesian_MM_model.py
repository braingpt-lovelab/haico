import torch
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

class BayesianCombinationModel:
    def __init__(self):
        self.params = {} # Store parameters
        
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
        
        # Define true labels
        truelabel_i = pyro.sample("truelabel", dist.Categorical(probs=torch.full((N, L), 1.0 / L)), obs=truelabel)

        # Set the means based on true label
        muA_i = muA0.unsqueeze(0).expand(N, L).clone()
        muB_i = muB0.unsqueeze(0).expand(N, L).clone()
        muA_i[torch.arange(N), truelabel_i] = muA1.type(muA0.type()) # Make sure muA1 is the same type as muA0
        muB_i[torch.arange(N), truelabel_i] = muB1.type(muB0.type()) # Make sure muB1 is the same type as muB0

        # Generate correlated probability scores for each label using a bivariate normal distribution
        probscoresA_i = pyro.sample('probscoresA', dist.Normal(muA_i, sigmaA), obs=probscoresA)
        pyro.sample('probscoresB', dist.Normal(muB_i + rho * sigmaB * ((probscoresA_i - muA_i) / sigmaA), \
            (1 - rho**2)**0.5 * sigmaB), obs=probscoresB)
    
    # Using the mp_context parameter to run the mode using multiprocessing with CUDA
    # REF01: https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
    # REF02: https://docs.pyro.ai/en/1.4.0/mcmc.html
    def infer(self, probscoresA, probscoresB, truelabel=None, params=None, num_samples=50, warmup_steps=1000, 
              num_chains=8, mp_context='spawn', disable_progbar=False, group_by_chain=False):
        # Run inference
        pyro.clear_param_store()
        kernel = NUTS(self._model)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, 
                    disable_progbar=disable_progbar, mp_context=mp_context)
        mcmc.run(probscoresA, probscoresB, truelabel=truelabel, params=params)
        self.posterior_samples = mcmc.get_samples(group_by_chain=group_by_chain)
        return self.posterior_samples
    
    def posterior_predict(self, probscoresA, probscoresB):
        if self.posterior_samples is None:
            raise RuntimeError("Call infer() first on training data.")

        N, L = probscoresA.shape
        samples = self.posterior_samples
        S = next(iter(samples.values())).shape[0]

        ll_accum = torch.zeros(N, L)

        # Aggregate over posterior draws
        for s in range(S):
            muA1   = samples['muA1'][s]
            muA0   = samples['muA0'][s]
            sigmaA = samples['sigmaA'][s]
            muB1   = samples['muB1'][s]
            muB0   = samples['muB0'][s]
            sigmaB = samples['sigmaB'][s]
            rho    = samples['rho'][s]

            for k in range(L):
                muA_i = muA0.expand(N,L).clone()
                muB_i = muB0.expand(N,L).clone()
                muA_i[:,k] = muA1
                muB_i[:,k] = muB1

                # Log-likelihoods
                logpA = dist.Normal(muA_i, sigmaA).log_prob(probscoresA).sum(dim=1)
                mean_B = muB_i + rho*sigmaB*((probscoresA - muA_i)/sigmaA)
                logpB = dist.Normal(mean_B, ((1-rho**2)**0.5)*sigmaB).log_prob(probscoresB).sum(dim=1)
                ll_accum[:,k] += logpA + logpB

        # Normalize
        probs = F.softmax(ll_accum, dim=1)
        y_pred = probs.argmax(dim=1)
        return y_pred