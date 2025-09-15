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
        
        # Define true labels 
        truelabel_i = pyro.sample("truelabel", dist.Categorical(probs=torch.full((N, L), 1.0 / L)), obs=truelabel)

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

    # Using the mp_context parameter to run the mode using multiprocessing with CUDA
    # REF01: https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
    # REF02: https://docs.pyro.ai/en/1.4.0/mcmc.html
    def infer(self, probscoresA, classificationB, confidenceB, truelabel=None, params=None, 
              num_samples=50, warmup_steps=1000, num_chains=8, mp_context='spawn', 
              disable_progbar=False, group_by_chain=False):
        # Run inference
        pyro.clear_param_store()
        kernel = NUTS(self._model)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, 
                    disable_progbar=disable_progbar, mp_context=mp_context)
        mcmc.run(probscoresA, classificationB, confidenceB, truelabel=truelabel, params=params)
        self.posterior_samples = mcmc.get_samples(group_by_chain=group_by_chain)
        return self.posterior_samples
    
    def posterior_predict(self, probscoresA, classificationB, confidenceB):
        if self.posterior_samples is None:
            raise RuntimeError("Train first with infer()")

        N, L = probscoresA.shape
        samples = self.posterior_samples
        S = next(iter(samples.values())).shape[0]

        logpA_accum = torch.zeros(N, L)
        logpB_class_accum = torch.zeros(N, L)
        logpB_conf_accum = torch.zeros(N, L)

        # Aggregate over posterior draws
        for s in range(S):
            muA1   = samples['muA1'][s]
            muA0   = samples['muA0'][s]
            sigmaA = samples['sigmaA'][s]
            muB1   = samples['muB1'][s]
            muB0   = self.params['muB0']
            sigmaB = self.params['sigmaB']
            rho    = samples['rho'][s]
            tau    = self.params['tau']
            delta  = samples['delta'][s]
            cutp   = torch.sort(samples['cutp'][s], dim=-1)[0]

            for k in range(L):
                muA_i = muA0.expand(N,L).clone()
                muB_i = muB0.expand(N,L).clone()
                muA_i[:,k] = muA1
                muB_i[:,k] = muB1

                # Log-likelihoods
                logpA = dist.Normal(muA_i, sigmaA).log_prob(probscoresA).sum(dim=1)
                logpA_accum[:,k]  = logpA
                mean_B = muB_i + rho*sigmaB*((probscoresA - muA_i)/sigmaA)
                softmax_B = F.softmax(mean_B / tau, dim=1)
                logpB_class = torch.log(softmax_B[torch.arange(N), classificationB]+1e-12)
                logpB_class_accum[:,k]  = logpB_class
                chosen_score = mean_B[torch.arange(N), classificationB]*delta
                logpB_conf = dist.OrderedLogistic(chosen_score, cutp*delta).log_prob(confidenceB)
                logpB_conf_accum[:,k] += logpB_conf

        # Normalize
        pA = F.softmax(logpA_accum, dim=1)
        pB_class = F.softmax(logpB_class_accum, dim=1)
        pB_conf = F.softmax(logpB_conf_accum, dim=1)
        probs = pA + pB_class + pB_conf
        y_pred = probs.argmax(dim=1)
        return y_pred
    