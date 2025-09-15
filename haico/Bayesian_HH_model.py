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
        self.params['mu1']   = pyro.sample('mu1', dist.Uniform(0, 1))
        self.params['mu0']   = torch.tensor(0.0)  # Fixed for the purpose of identifiability
        self.params['sigma'] = torch.tensor(1.0)  # Fixed for the purpose of identifiability
        self.params['rho']    = pyro.sample('rho', dist.Uniform(-1, 1))
        self.params['tau']    = torch.tensor(0.05)  # Fixed for best convergence results
        self.params['delta']  = pyro.sample('delta', dist.Uniform(0, 100))
        self.params['cutp']   = torch.sort(pyro.sample('cutp', dist.Uniform(0, 1).expand([2])))[0]

    def _model(self, classificationA, confidenceA, classificationB, confidenceB, truelabel=None, params=None):
        N, L = len(classificationA), 16
        
        # Initialize parameters if not provided
        if not params: 
            self._get_params()
            params = self.params
                
        # Extract parameters from the provided dictionary
        mu1   = params['mu1']
        mu0   = params['mu0']
        sigma = params['sigma']
        rho   = params['rho']
        tau   = params['tau']
        delta = params['delta']
        cutp  = params['cutp']
        
        # Define true labels
        truelabel_i = pyro.sample("truelabel", dist.Categorical(probs=torch.full((N, L), 1.0 / L)), obs=truelabel)

        # Set the means based on true label
        mu_i = mu0.unsqueeze(0).expand(N, L).clone()
        mu_i[torch.arange(N), truelabel_i] = mu1.type(mu0.type()) # Make sure muB1 is the same type as muB0

        # Generate correlated probability scores for each label using a bivariate normal distribution
        probscoresA_i = pyro.sample('probscoresA', dist.Normal(mu_i, sigma))
        probscoresB_i = pyro.sample('probscoresB', dist.Normal( \
            mu_i + rho * sigma * ((probscoresA_i - mu_i) / sigma), (1 - rho**2)**0.5 * sigma))

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

    # Using the mp_context parameter to run the mode using multiprocessing with CUDA
    # REF01: https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
    # REF02: https://docs.pyro.ai/en/1.4.0/mcmc.html
    def infer(self, classificationA, confidenceA, classificationB, confidenceB, truelabel=None, 
              params=None, num_samples=50, warmup_steps=1000, num_chains=8, mp_context="spawn", 
              disable_progbar=False, group_by_chain=False):
        # Run inference
        pyro.clear_param_store()
        kernel = NUTS(self._model)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, 
                    disable_progbar=disable_progbar, mp_context=mp_context)
        mcmc.run(classificationA, confidenceA, classificationB, confidenceB, truelabel=truelabel, params=params)
        self.posterior_samples = mcmc.get_samples(group_by_chain=group_by_chain)
        return self.posterior_samples
    
    def posterior_predict(self, classificationA, confidenceA, classificationB, confidenceB):
        if self.posterior_samples is None:
            raise RuntimeError("Call infer() first on training data.")

        N, L = len(classificationA), 16
        samples = self.posterior_samples
        S = next(iter(self.posterior_samples.values())).shape[0]

        ll_accum = torch.zeros(N, L)

        # Aggregate over posterior draws
        for s in range(S):
            mu1   = samples['mu1'][s]
            mu0   = self.params['mu0']
            sigma = self.params['sigma']
            rho   = samples['rho'][s]
            tau   = self.params['tau']
            delta = samples['delta'][s]
            cutp  = torch.sort(samples['cutp'][s], dim=-1)[0]

            for k in range(L):  
                mu_i = mu0.expand(N, L).clone()
                mu_i[:, k] = mu1  

                # Log-likelihoods
                softmaxA = F.softmax(mu_i / tau, dim=1)
                logpA_class = torch.log(softmaxA[torch.arange(N), classificationA] + 1e-12)
                mean_B = mu_i + rho * ((mu_i - mu_i)/sigma) * sigma 
                softmaxB = F.softmax(mean_B / tau, dim=1)
                logpB_class = torch.log(softmaxB[torch.arange(N), classificationB] + 1e-12)
                chosenA = mu_i[torch.arange(N), classificationA] * delta
                logpA_conf = dist.OrderedLogistic(chosenA, cutp * delta).log_prob(confidenceA)
                chosenB = mean_B[torch.arange(N), classificationB] * delta
                logpB_conf = dist.OrderedLogistic(chosenB, cutp * delta).log_prob(confidenceB)
                ll_accum[:, k] += (logpA_class + logpB_class + logpA_conf + logpB_conf)

        # Normalize
        probs = F.softmax(ll_accum, dim=1)
        y_pred = probs.argmax(dim=1)
        return y_pred