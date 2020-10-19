# TODO: add support for non-OLS regression

"""
Forward stepwise model selection

Author : Meaghan Flagg

Start with null model (Y ~ 1)
Fit p OLS regression models, each with one of the X variables and the intercept
Select the best out of these models, and fix this X term in the model going forward. (lowest residual sum of squares, highest R squared, lowest P-value)
Search through remaining p-1 variables and determine which variable should be added to the current model to best improve the residual sum of squares.
Continue until some stopping rule is satisfied (additional terms have p-value > 0.05, minimum AIC is reached)

"""

import pandas as pd
import numpy as np
import statsmodels.multivariate.manova as manova
import statsmodels.multivariate.multivariate_ols as smv

idx=pd.IndexSlice

def forward_stepwise(X, Y, data, param_select="ssr", statistic="pillai", verbose=False):
    """
    Performs forward stepwise model selection for a set of X and Y data.
    
    Parameters
    -------
    X : array-like
        Dataframe or array containing independent/exog variable data. Must be of
        shape (m,n) where m is variables and n is observations.
        If data is not None, must be list-like, corresponding to column names in data.
    Y : array-like
        Dataframe or array containing dependent/endog variable data. Must be of
        shape (m,n) where m is outcome(s) and n is observations.
        If data is not None, must be list-like, corresponding to column names in data.
    data : array-like
        Dataframe containing both independent/exog and dependent/endog variables. Must
        be of shape (m,n), where m is variables/outcomes and n is observations.
    param_select : str, optional
        Can be one of ['ssr', 'pval']. Metric used to select the parameter that
        best improves the model at each iteration. Default is 'ssr'.
    statistic : str, optional
        Can be one of ['pillai','wilks','hotelling-lawly','roys']. Test statistic used by
        statsmodels.multivariate.manova to calculate statistical significance of added parameters.
    verbose : boolean, optional
        If True, return an additional dataframe containing information about paramters that
        were not added to model due to co-linearity.
    
    Returns
    -------
    results : dataframe
        Dataframe of model parameters at each iteration of selection
    dropped_params : dataframe, optional
        Only returned if verbose=True.
    
    """
    
    if data is not None:
        endog=data[Y]
        endog_names=Y
        exog=data[X]
        exog_names=X
    else:
        raise NotImplementedError("use with 'data' arg for now.")
        endog=Y
        endog_names=Y.columns
        exog=X
        exog_names=X.columns
        # combine data
        if isinstance(X, pd.DataFrame):
            data = pd.concat([X,Y], axis=1, ignore_index=True)
        if isinstance(X, np.ndarray): # TODO: fix.
            raise NotImplementedError("X and Y should be dataframes for now.")
    
    # check if all data is numeric
    numeric = data.apply(lambda s: pd.to_numeric(s, errors='coerce').isnull().all())
    if numeric.sum() > 0:
        fail = numeric.index[numeric==True].values
        msg = "Data in the following column(s) is not numeric! {0}".format(str(fail))
        raise ValueError(msg)
    numeric.index[numeric==True].values
    
    # parse "statistics" arg:
    stats_dict = dict(zip(['pillai','wilks','hotelling-lawly','roys'],
                         ["Pillai's trace","Wilks' lambda","Hotelling-Lawley trace","Roy's greatest root"]))
    manv_statistic = stats_dict[statistic]
    
    
    # initialize empty dataframes to store results from each round of parameter addition, and dropped params
    cols=["formula","ssr","log-likelihood","F statistic","Pr>F","AIC"]
    resDF = pd.DataFrame(np.zeros((1,len(cols))), index=["NullModel"], columns=cols)
    
    dropDF = pd.DataFrame(columns=["colinear_var","spearman_r"])
    
    # initialize null model:
    nullformula = "{0} ~ 1".format(" + ".join(endog_names))
    nullmod = smv._MultivariateOLS.from_formula(
    formula=nullformula, data=data).fit()
    # add to results df
    resDF.loc["NullModel"] = [nullmod.formula, nullmod.ssr(), nullmod.loglike, np.nan, np.nan, nullmod.aic]
    
    #import pdb; pdb.set_trace() #####
    
    params=exog_names
    bestmodel=None
    while len(params) > 0: # e.g. as long as there are parameters remaining
        
        # initialize empty df to store parameter data
        cols=["paramter","ssr","statistic","F-value","Pr>F"]
        paramDF = pd.DataFrame(columns=cols, index=params)
        
        for param in params:
            if bestmodel: # defined at end of first iteration
                refmodel=bestmodel
            else:
                refmodel=nullmod
            
            # build + fit model with parameter:
            formula = refmodel.formula + " + {0}".format(str(param))
            testmod = smv._MultivariateOLS.from_formula(
                formula=formula, data=data).fit()
            
            # run MANOVA to calculate p val for parameters
            manv = manova.MANOVA.from_formula(testmod.formula, data=cars).mv_test()
            Fval = manv.summary_frame.loc[idx[param, manv_statistic],"F Value"]
            PrF = manv.summary_frame.loc[idx[param, manv_statistic],"Pr > F"]
            
            # add data to paramDF
            paramDF.loc[param] = [param,testmod.ssr(),manv_statistic,Fval,PrF]
            ########################## end of parameter addition ###########################
        
        # Choose best parameter:
        if param_select == "ssr":
            bestParam = paramDF.sort_values(by="ssr", ascending=True).index[0]
        elif param_select == "pval":
            bestParam = paramDF.sort_values(by="Pr>F", ascending=True).index[0]
        else:
            raise ValueError("'param_select' must be either 'ssr' or 'pval'. Got {0}".format(str(param_select)))
        
        # initialize model with best parameter
        formula = refmodel.formula + " + {0}".format(str(bestParam))
        bestmodel = statsmodels.multivariate.multivariate_ols._MultivariateOLS.from_formula(
                formula=formula, data=data).fit()
            
        # run F test versus refmodel:
        # arg order: refmod, bestmod 
        fTest = smv.F_test_multivariate(refmodel, bestmodel)
        
        # add data to results df
        resDF.loc[bestParam] = [bestmodel.formula, bestmodel.ssr(), bestmodel.loglike,
                                fTest.F_statistic, fTest.prF, bestmodel.aic]
        
        # remove bestParam and any highly co-linear variables from remaining params:
        corr = data[params].corr(method='spearman')
        drop = list(corr[abs(corr[bestParam]) > 0.8].index) # this includes the term itself since r=1
        [params.remove(d) for d in drop]
        # note in resDF that parameter was dropped due to co-linearity
        drop.remove(bestParam)
        
        for d in drop:
            dropDF.loc[d,"colinear_var"] = bestParam
            dropDF.loc[d,"spearman_r"] = corr.loc[d,bestParam]
            #resDF.loc[d,"formula"] = "dropped due to colinearity with {0}".format(bestParam)
    if verbose == True:
        return resDF.sort_values(by="AIC", ascending=True), dropDF
    else:
        return resDF.sort_values(by="AIC", ascending=True)
