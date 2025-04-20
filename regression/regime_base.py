import numpy as np
import pandas as pd
from contracts.projection_pb2 import RegimeBoundary
from itertools import groupby
from dataclasses import dataclass
from collections import defaultdict
from dataclasses import replace
from scipy.optimize import minimize

@dataclass
class OptVariableRegimes:
    """
    Class to represent an optimization variable.
    """
    name:str    
    boundaries:list[float]    
   
@dataclass
class OptVariable:
    """
    Class to represent an optimization variable.
    """
    name:str
    value:float
    min_value:float
    max_value:float
    order:int
    
    def __post_init__(self):
        if self.value < self.min_value or self.value > self.max_value:
            raise ValueError(f"Value {self.value} is out of bounds [{self.min_value}, {self.max_value}] for variable {self.name}")
    def __repr__(self):
        return f"OptVariable(name={self.name}, value={self.value}, min_value={self.min_value}, max_value={self.max_value})"
    def __str__(self):
        return f"OptVariable(name={self.name}, value={self.value}, min_value={self.min_value}, max_value={self.max_value})"
   
class RegimeBaseProjectionModel:
    
    def __init__(self, optVariableRegimes:list[OptVariableRegimes]):                        
        self.OpVariablesRegimes = optVariableRegimes
       
    def _get_beta_factor_name(name:str,i:int)->str:
        """
        Generate a unique name for the beta variable based on the factor name and index.
        """        
        return f"b_{name}_{i}"
    
    def _get_intercep_factor_name(name:str,i:int)->str:
        """
        Generate a unique name for the intercept variable based on the factor name and index.
        """        
        return f"I_{name}_{i}"
   
    def extract_factor_name(self, name:str)->str:
        """
        Extract the factor name from the variable name.
        """
        return name.split("_")[1]
    
    def isBeta(self, name:str)->bool:
        return name.startswith("b_")
    
    def isIntercept(self, name:str)->bool:
        return name.startswith("I_")
        
    def generate_variables(self, x:pd.Series)->np.array[OptVariable]:
        """converts an input pandas optimization seris into the complete list of
           optimization variables to be uses with regimes.
           Betas and Intercepts are created for each regime this essentially will 
           generate multiple betas for each factor and regime. Intercepts are created also
           in a similar fashion, but they are subject to continuity constraints.
           """
        I = {} 
        betas = {}
        
        
        for factor in self.OpVariablesRegimes:
            name = factor.name
            fbetaName = lambda i : self._get_beta_factor_name(name,i)
            finterceptName = lambda i : self._get_intercep_factor_name(name,i)
            
            
            if name not in x: #if a factor is not in the optimization variables, skip it
                continue
            
            boundaries = factor.boundaries
            # Create a variable for each factor      
            v = x[factor] 
                                     
            I.append(OptVariable(finterceptName(0),0.0,float("-inf"),float("-inf"),0))
            if len(boundaries) == 0:
                betas.append(OptVariable(fbetaName(0),0.0,float("-inf"),float("inf"),0))
               
            
            if len(boundaries) == 1:
                betas.append(OptVariable(fbetaName(0),0.0,float("-inf"),boundaries[0],0))
                betas.append(OptVariable(fbetaName(1),0.0,boundaries[0],float("+inf"),1))
               
            else: 
                startRegime = float("-inf")
                endRegime = boundaries[0]
                N = len(boundaries)
                for i in range(N-1):                                  
                    betas.append(OptVariable(fbetaName(i),0.0,startRegime,endRegime,i))                   
                    startRegime = boundaries[i]
                    endRegime = boundaries[i+1]
                     
                betas.append(OptVariable(fbetaName(N-1),0.0,boundaries[-1],float("+inf"),N-1))                
                
        return betas, I
                             
        
    
      
    def sortFactors(self, w:list[OptVariable])->list[OptVariable]:
        """
        Sort the optimization variables by their factor names.
        """
        return sorted(w, key=lambda x: self.extract_factor_name(x.name))
    
    def convert_opt_variables_to_array(self, w:list[OptVariable])->np.array:
        """
        Convert the optimization variables to an array.
        """        
        return np.array([x.value for x in sortFactors(w)])
    
    def compute_intercepts(self, w:list[OptVariable])->list[OptVariable]:
        extended_w = w.copy()
        orderbetas =  sorted([x for x in w if self.isBeta(x.name)], lambda a: a.order)
        I = [x for x in w if self.isIntercept(x.name)]
        lastIntercept = I[0]
        intercepts = []
        for i in range(1,len(orderbetas)):
            limit= orderbetas[i].min_value
            b_1 = orderbetas[i-1].value
            b_2 = orderbetas[i].value
            I_2 = lastIntercept + b_1*limit - b_2*limit
            intercepts.append(OptVariable(self._get_intercep_factor_name(i), I_2, float("-inf"),float("+inf"), i))

        extended_w.extend(intercepts)
        return extended_w
    
    def compute_prediction(self, w:list[OptVariable], X:pd.DataFrame)->np.array:
        w = self.compute_intercepts(w)        
        factors =  list(dict.fromkeys([self.extract_factor_name(x.name) for x in w]))
        betas = {factor_name:self.sortFactors(list(group)) for factor_name, group in groupby(w, lambda x: self.extract_factor_name(x.name)) if self.isBeta(factor_name)}        
        intercepts = {name: self.sortFactors(list(g)) for name,g in groupby(w,lambda x: self.extract_factor_name(x.name)) if self.isIntercept(name)}        
                
        acc = None
        for factor in factors: 
            betas_for_factor = betas[factor]
            intercepts_for_factor = intercepts[factor]
            for n,beta in enumerate(betas_for_factor):
                min_value, max_value = beta.min_value, beta.max_value  
                beta:float = beta.value   
                I:float =  intercepts_for_factor[n].value     
                
                factorSeries = X[factor].copy()
                factorSeries[factorSeries < min_value] = 0.0
                factorSeries[factorSeries > max_value] = 0.0
                if acc is None:
                    acc = I + beta * factorSeries
                else:
                    acc += I + beta * factorSeries
                
        return acc
                
        
    def FromArrayToOptVariables(self, w:np.array)->list[OptVariable]:
        """
        Convert the optimization variables from an array to a list of OptVariable objects.
        """
        return [replace(x, value=w[i]) for i,x in enumerate(w)]
    
    def FromOptVariablesArray(self, w:list[OptVariable])->np.array:
        """
        Convert the optimization variables from an array to a list of OptVariable objects.
        """
        return np.array([x.value for x in w])
    
      
    def fitness(self, w:list[OptVariable],X:pd.DataFrame,Y:pd.Series)->float:
        """
        Calculate the fitness of the model based on the regimes.
        """
       
        pred = self.compute_prediction(w, X)
        err = pred - Y
        rerr = np.sqrt(np.mean(err**2))
        return rerr
    
    
    def calibrate(self, optVariables:list[OptVariable], X: pd.DataFrame, Y: pd.Series) -> list[OptVariable]:
        """
        Calibrate the model using L-BFGS-B optimization.
        """
        # Convert the initial optimization variables to an array        
        X = X.loc[:,[x.name for x in optVariables]]
        initial_w =  self.FromOptVariablesArray(optVariables)
        bounds = [(x.min_value, x.max_value) for x in optVariables]
        
        # Define the objective function (fitness)
        def objective(w):
            optVariables = self.FromArrayToOptVariables(w)
            return self.fitness(optVariables, X, Y)

        # Perform optimization using L-BFGS-B
        result = minimize(
            objective,
            initial_w,
            method="L-BFGS-B",
            bounds=bounds,
            options={"disp": True}  # Display optimization progress
        )

        # Check if optimization was successful
        if not result.success:
            raise ValueError(f"Optimization failed: {result.message}")

        # Convert the optimized array back to OptVariable objects        

        return result.x
    

import kagglehub


print("Path to dataset files:", path)
if __name__ == "__main__":
    # Download latest version
    path = kagglehub.dataset_download("belayethossainds/global-inflation-dataset-212-country-19702022")
    optRegimes = [OptVariableRegimes("In", [0.1, 0.5])
    pass