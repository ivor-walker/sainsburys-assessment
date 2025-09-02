# Too many unresolved data issues to go ahead with linear approach
"""
Generalised estimated equations (GEE)
Deals with the structural problems of the data (i.e. the product and catalogue IID violations)
"""
from src.models.interface import InterfaceModel

# Cancel imports for non-used models to avoid unnecessary dependencies
# import statsmodels.api as sm
# import statsmodels.formula.api as smf

class GEE(InterfaceModel):
    def __init__(self):
        # Binomial link function as we're predicting a binary outcome
        self.__family = sm.families.Binomial()

        # Exchangeable covariance structure: e
        self.__cov_struct = sm.cov_struct.Exchangeable()
