import torch
import torch.nn.functional as F

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS

class BayesianCombinationModel:
    def __init__(self):
        self.params = {} # Store parameters
        self.trace = {}  # Store trace

        # Find if there is a GPU available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.device = torch.device("cpu") # Force to use CPU

        # Set the default tensor type to single precision
        torch.set_default_dtype(torch.float32)
        
    def _get_params(self):

        # Define priors and constants
        self.params['muA1']   = pyro.sample('muA1', dist.Normal(0, torch.tensor(10., device=self.device)))
        self.params['muA0']   = pyro.sample('muA0', dist.Normal(0, torch.tensor(10., device=self.device)))
        self.params['sigmaA'] = pyro.sample('sigmaA', dist.Uniform(0, torch.tensor(15., device=self.device)))
        self.params['muB1']   = pyro.sample('muB1', dist.Normal(0, torch.tensor(10., device=self.device)))
        self.params['muB0']   = torch.tensor(0.0, device=self.device) # Fixed for the purpose of identifiability
        self.params['sigmaB'] = torch.tensor(1.0, device=self.device) # Fixed for the purpose of identifiability
        self.params['rho']    = pyro.sample('rho', dist.Uniform(torch.tensor(-1., device=self.device), torch.tensor(1., device=self.device)))
        self.params['tau']    = torch.tensor(0.05, device=self.device) # Fixed for best convergence results
        self.params['delta']  = pyro.sample('delta', dist.Uniform(0, torch.tensor(100., device=self.device)))
        self.params['cutp']   = torch.sort(pyro.sample('cutp', dist.Uniform(0, torch.tensor(1.,device=self.device)).expand([2])))[0]

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
            truelabel_i = pyro.sample('truelabel', dist.Categorical(probs=torch.ones(L, device=self.device)).expand([N]), obs=truelabel)
        else:
            # Learn label probabilities using a Dirichlet distribution with uniform prior
            labelprob = pyro.sample('labelprob', dist.Dirichlet(torch.ones(N,L, device=self.device)))
            truelabel_i = torch.argmax(labelprob, dim=-1).squeeze()

        # DEBUG: Print the shape and device of the true label
        #print(f"truelabel_i: {truelabel_i} ({truelabel_i.shape}) type: {truelabel_i.dtype} device: {truelabel_i.device}")

        # Set the means based on true label
        muA_i = muA0.unsqueeze(0).expand(N, L).clone()
        muB_i = muB0.unsqueeze(0).expand(N, L).clone()
        # NOTE: In PyTorch tensors used as indices must be long, int, byte or bool tensors
        muA_i[torch.arange(N, device=self.device, dtype=torch.int), truelabel_i.int()] = muA1.type(muA0.type()) # Make sure muA1 is the same type as muA0
        muB_i[torch.arange(N, device=self.device, dtype=torch.int), truelabel_i.int()] = muB1.type(muB0.type()) # Make sure muB1 is the same type as muB0

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

    def infer(self, probscoresA, classificationB, confidenceB, truelabel=None, params=None, num_samples=10, warmup_steps=1000, num_chains=2, 
              disable_progbar=False, group_by_chain=False, mp_context='spawn', debug=False):

        # Testing if infer arguments are on the same device
        if probscoresA.device != self.device:
            if debug:
                print(f"Moving probscoresA to {self.device}")
            probscoresA = probscoresA.to(self.device)

        if classificationB.device != self.device:
            if debug:
                print(f"Moving classificationB to {self.device}")
            classificationB = classificationB.to(self.device)

        if confidenceB.device != self.device:
            if debug:
                print(f"Moving confidenceB to {self.device}")
            confidenceB = confidenceB.to(self.device)

        if truelabel is not None:
            if truelabel.device != self.device:
                if debug:
                    print(f"Moving truelabel to {self.device}")
                truelabel = truelabel.to(self.device)

        if params:
            for key in params.keys():
                if params[key].device != self.device:
                    if debug:
                        print(f"Moving {key} to {self.device}")
                    params[key] = params[key].to(self.device)

        # Run inference
        pyro.clear_param_store()
        # JIT compile of the model fails with the error:
        # RuntimeError: Index put requires the source and destination dtypes match, got Double for the destination and Float for the source.
        kernel = NUTS(self._model, jit_compile=True, ignore_jit_warnings=True, max_tree_depth=10, target_accept_prob=0.8)
        mcmc = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=num_chains, 
                    disable_progbar=disable_progbar, mp_context=mp_context)
        mcmc.run(probscoresA, classificationB, confidenceB, truelabel=truelabel, params=params)
        return mcmc.get_samples(group_by_chain=group_by_chain)