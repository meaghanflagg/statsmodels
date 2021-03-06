# TODO: add support for non-OLS regression

"""
Functions to perform model selection using multivariate OLS regression.

Currently includes:
 -------
forward_stepwise : performs forward stepwise model selection

Author : Meaghan Flagg
"""

import pandas as pd
import numpy as np
import statsmodels.multivariate.manova as manova
import statsmodels.multivariate.multivariate_ols as smv
import warnings

idx=pd.IndexSlice

def forward_stepwise(X, Y, data, ref_model=None, param_select="ssr", statistic="pillai", pval_thresh=None, return_dropped=False, verbose=False):
    """
    Performs forward stepwise model selection for a set of X and Y data.

    1. Start with null model (Y ~ 1)
    2. Fit p OLS regression models, each with one of the X variables and the intercept,
       where p is the number of X terms in the dataset.
    3. Select the best out of these models, and fix this X term in the model going forward.
         a. lowest residual sum of squares (param_select='ssr')
         b. lowest p-value from MANOVA test (not yet implemented)
    4. Compare this "test" model to previous reference model using F test to determine
       if reduction in residual sum of squares is significant.
    5. Remove any terms that are highly co-linear with selected terms.
    6. Search through remaining p-1 terms and determine which term should be added to the
       current model to best improve the residual sum of squares.
    7. Continue until some stopping rule is satisfied or you have run out of terms
         a. additional terms have p-value > threshold in MANOVA


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
    ref_model : str, optional
        Alternative to null model to initiate model selection. Should be a string
        containing an R-style formula, as in `statsmodels.formula.api`.
        Example: 'Y1 + Y2 ~ X1 + X2'
    param_select : str, optional
        Can be one of ['ssr', 'pval']. Metric used to select the parameter that
        best improves the model at each iteration. Default is 'ssr'.
    statistic : str, optional
        Can be one of ['pillai','wilks','hotelling-lawly','roys']. Test statistic used by
        statsmodels.multivariate.manova to calculate statistical significance of added parameters.
    pval_thresh : float, optional
        Stop parameter addition when MANOVA p-value for all remaining terms is greater than threshold.
    return_dropped : boolean, optional
        If True, return an additional dataframe containing information about paramters that
        were not added to model due to co-linearity.
    verbose : boolean, optional
        If True, print status updates.

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

    # define Parameters
    params=list(exog_names)

    # check if all data is numeric
    numeric = data.apply(lambda s: pd.to_numeric(s, errors='coerce').isnull().all())
    if numeric.sum() > 0:
        fail = numeric.index[numeric==True].values
        msg = "Data in the following column(s) is not numeric! {0}".format(str(fail))
        raise ValueError(msg)
    #numeric.index[numeric==True].values # why is this here?

    # parse "statistics" arg:
    stats_dict = dict(zip(['pillai','wilks','hotelling-lawly','roys'],
                         ["Pillai's trace","Wilks' lambda","Hotelling-Lawley trace","Roy's greatest root"]))
    manv_statistic = stats_dict[statistic]


    # initialize empty dataframes to store results from each round of parameter addition, and dropped params
    cols=["formula","param_Pr>F","ssr","log-likelihood","df_model","F statistic","Pr>F","AIC"]
    resDF = pd.DataFrame(np.zeros((1,len(cols))), index=["NullModel"], columns=cols)

    dropDF = pd.DataFrame(columns=["colinear_var","spearman_r"])

    # initialize null model:
    if ref_model is None:
        nullformula = "{0} ~ 1".format(" + ".join(endog_names))
        nullmod = smv._MultivariateOLS.from_formula(formula=nullformula, data=data).fit()
    else:
        # TODO check formula syntax here?
        nullmod = smv._MultivariateOLS.from_formula(formula=ref_model, data=data).fit()
        # remove null mod X parameters from exog_names so we don't add them to model:
        null_params = ref_model.split(" ~ ")[1].split(" + ")
        #null_params = nullmod.exog_names[:1] # first term is intercept THIS FAILS IF nullmod CONTAINS CATEGORICAL VARIABLES
        [params.remove(p) for p in null_params]


    # add to results df
    resDF.loc["NullModel"] = [nullmod.formula.split('~')[1], np.nan, nullmod.ssr(), nullmod.loglike, nullmod.df_model,np.nan, np.nan, nullmod.aic]

    # begin iteration
    itercount=0
    bestmodel=None
    while len(params) > 0: # e.g. as long as there are parameters remaining

        itercount+=1

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
            try:
                testmod = smv._MultivariateOLS.from_formula(
                    formula=formula, data=data).fit()
            except ValueError as e:
                if str(e) == "Covariance of x singular!":
                    warnings.warn("{0}\nRemoving parameter: {1}".format(str(e), str(param)))
                    params.remove(param)
                    continue
                else:
                    raise

            # run MANOVA to calculate p val for parameters
            try:
                manv = manova.MANOVA.from_formula(testmod.formula, data=data).mv_test()
                Fval = manv.summary_frame.loc[idx[param, manv_statistic],"F Value"]
                PrF = manv.summary_frame.loc[idx[param, manv_statistic],"Pr > F"]

                # add data to paramDF
                paramDF.loc[param] = [param,testmod.ssr(),manv_statistic,Fval,PrF]

            except np.linalg.LinAlgError as e: # in some scenarios, running MANOVA generates LinAlgError: singular matrix. Not sure why, could this be due to minimal variation?
                warnings.warn("{0}\nSkipping parameter: {1}".format(str(e), str(param)))
                continue # go to next parameter

            ########################## end of parameter addition ###########################

        if pval_thresh:
            # nan pvalues cause problems here. replace them with 1.
            paramDF["Pr>F"] = paramDF["Pr>F"].replace(np.nan, 1)
            if paramDF["Pr>F"].min() > pval_thresh: # all parameters are above pval_thresh
                if verbose==True: print("No remaining params with Pr>F < {}, exiting".format(pval_thresh))
                break # exit while loop, should go to line 207

            else: # filter paramDF to only include terms with Pr>F < pval_thresh:
                paramDF = paramDF[ paramDF["Pr>F"] < pval_thresh ]



        # Choose best parameter:
        if param_select == "ssr":
            bestParam = paramDF.sort_values(by="ssr", ascending=True).index[0]
        elif param_select == "pval":
            bestParam = paramDF.sort_values(by="Pr>F", ascending=True).index[0]
        else:
            raise ValueError("'param_select' must be either 'ssr' or 'pval'. Got {0}".format(str(param_select)))

        # initialize model with best parameter
        formula = refmodel.formula + " + {0}".format(str(bestParam))
        bestmodel = smv._MultivariateOLS.from_formula(
                formula=formula, data=data).fit()

        # run F test versus refmodel:
        # arg order: refmod, bestmod
        fTest = smv.F_test_multivariate(refmodel, bestmodel)

        # add data to results df
        resDF.loc[bestParam] = [bestmodel.formula.split('~')[1], paramDF.loc[bestParam,"Pr>F"],
                                bestmodel.ssr(), bestmodel.loglike, bestmodel.df_model,
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
            print('Iteration #{0}'.format(str(itercount)))
            print('adding {0} to model'.format(str(bestParam)))
            print('dropped: {0}'.format(', '.join(drop)))
            print('-' * 20, '\n')

    if return_dropped == True:
        return resDF.sort_values(by="AIC", ascending=True), dropDF
    else:
        return resDF.sort_values(by="AIC", ascending=True)
