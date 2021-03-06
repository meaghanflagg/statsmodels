# -*- coding: utf-8 -*-

# TODO: add summary method for _MultivariateOLSresults

"""General linear model

author: Yichuan Liu, Meaghan Flagg 
"""
import numpy as np
from numpy.linalg import eigvals, inv, solve, matrix_rank, pinv, svd
from scipy import stats
import pandas as pd
from patsy import DesignInfo

from statsmodels.compat.pandas import Substitution
from statsmodels.base.model import Model
from statsmodels.iolib import summary2
from statsmodels.tools.decorators import (cache_readonly,
                                          cache_writable)
__docformat__ = 'restructuredtext en'

_hypotheses_doc = \
"""hypotheses : list[tuple]
    Hypothesis `L*B*M = C` to be tested where B is the parameters in
    regression Y = X*B. Each element is a tuple of length 2, 3, or 4:

      * (name, contrast_L)
      * (name, contrast_L, transform_M)
      * (name, contrast_L, transform_M, constant_C)

    containing a string `name`, the contrast matrix L, the transform
    matrix M (for transforming dependent variables), and right-hand side
    constant matrix constant_C, respectively.

    contrast_L : 2D array or an array of strings
        Left-hand side contrast matrix for hypotheses testing.
        If 2D array, each row is an hypotheses and each column is an
        independent variable. At least 1 row
        (1 by k_exog, the number of independent variables) is required.
        If an array of strings, it will be passed to
        patsy.DesignInfo().linear_constraint.

    transform_M : 2D array or an array of strings or None, optional
        Left hand side transform matrix.
        If `None` or left out, it is set to a k_endog by k_endog
        identity matrix (i.e. do not transform y matrix).
        If an array of strings, it will be passed to
        patsy.DesignInfo().linear_constraint.

    constant_C : 2D array or None, optional
        Right-hand side constant matrix.
        if `None` or left out it is set to a matrix of zeros
        Must has the same number of rows as contrast_L and the same
        number of columns as transform_M

    If `hypotheses` is None: 1) the effect of each independent variable
    on the dependent variables will be tested. Or 2) if model is created
    using a formula,  `hypotheses` will be created according to
    `design_info`. 1) and 2) is equivalent if no additional variables
    are created by the formula (e.g. dummy variables for categorical
    variables and interaction terms)
"""


def _multivariate_ols_fit(endog, exog, method='svd', tolerance=1e-8):
    """
    Solve multivariate linear model y = x * params
    where y is dependent variables, x is independent variables

    Parameters
    ----------
    endog : array_like
        each column is a dependent variable
    exog : array_like
        each column is a independent variable
    method : str
        'svd' - Singular value decomposition
        'pinv' - Moore-Penrose pseudoinverse
    tolerance : float, a small positive number
        Tolerance for eigenvalue. Values smaller than tolerance is considered
        zero.
    Returns
    -------
    a tuple of matrices or values necessary for hypotheses testing

    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    Notes
    -----
    Status: experimental and incomplete
    """
    y = endog
    x = exog
    nobs, k_endog = y.shape
    nobs1, k_exog= x.shape
    if nobs != nobs1:
        raise ValueError('x(n=%d) and y(n=%d) should have the same number of '
                         'rows!' % (nobs1, nobs))

    # Calculate the matrices necessary for hypotheses testing
    df_resid = nobs - k_exog
    if method == 'pinv':
        # Regression coefficients matrix
        pinv_x = pinv(x)
        params = pinv_x.dot(y)

        # inverse of x'x
        inv_cov = pinv_x.dot(pinv_x.T)
        if matrix_rank(inv_cov,tol=tolerance) < k_exog:
            raise ValueError('Covariance of x singular!')

        # Sums of squares and cross-products of residuals
        # Y'Y - (X * params)'B * params
        t = x.dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    elif method == 'svd':
        u, s, v = svd(x, 0)
        if (s > tolerance).sum() < len(s):
            raise ValueError('Covariance of x singular!')
        invs = 1. / s

        params = v.T.dot(np.diag(invs)).dot(u.T).dot(y)
        inv_cov = v.T.dot(np.diag(np.power(invs, 2))).dot(v)
        t = np.diag(s).dot(v).dot(params)
        sscpr = np.subtract(y.T.dot(y), t.T.dot(t))
        return (params, df_resid, inv_cov, sscpr)
    else:
        raise ValueError('%s is not a supported method!' % method)


def multivariate_stats(eigenvals,
                       r_err_sscp,
                       r_contrast, df_resid, tolerance=1e-8):
    """
    For multivariate linear model Y = X * B
    Testing hypotheses
        L*B*M = 0
    where L is contrast matrix, B is the parameters of the
    multivariate linear model and M is dependent variable transform matrix.
        T = L*inv(X'X)*L'
        H = M'B'L'*inv(T)*LBM
        E =  M'(Y'Y - B'X'XB)M

    Parameters
    ----------
    eigenvals : ndarray
        The eigenvalues of inv(E + H)*H
    r_err_sscp : int
        Rank of E + H
    r_contrast : int
        Rank of T matrix
    df_resid : int
        Residual degree of freedom (n_samples minus n_variables of X)
    tolerance : float
        smaller than which eigenvalue is considered 0

    Returns
    -------
    A DataFrame

    References
    ----------
    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
    """
    v = df_resid
    p = r_err_sscp
    q = r_contrast
    s = np.min([p, q])
    ind = eigenvals > tolerance
    n_e = ind.sum()
    eigv2 = eigenvals[ind]
    eigv1 = np.array([i / (1 - i) for i in eigv2])
    m = (np.abs(p - q) - 1) / 2
    n = (v - p - 1) / 2

    cols = ['Value', 'Num DF', 'Den DF', 'F Value', 'Pr > F']
    index = ["Wilks' lambda", "Pillai's trace",
             "Hotelling-Lawley trace", "Roy's greatest root"]
    results = pd.DataFrame(columns=cols,
                           index=index)

    def fn(x):
        return np.real([x])[0]

    results.loc["Wilks' lambda", 'Value'] = fn(np.prod(1 - eigv2))

    results.loc["Pillai's trace", 'Value'] = fn(eigv2.sum())

    results.loc["Hotelling-Lawley trace", 'Value'] = fn(eigv1.sum())

    results.loc["Roy's greatest root", 'Value'] = fn(eigv1.max())

    r = v - (p - q + 1)/2
    u = (p*q - 2) / 4
    df1 = p * q
    if p*p + q*q - 5 > 0:
        t = np.sqrt((p*p*q*q - 4) / (p*p + q*q - 5))
    else:
        t = 1
    df2 = r*t - 2*u
    lmd = results.loc["Wilks' lambda", 'Value']
    lmd = np.power(lmd, 1 / t)
    F = (1 - lmd) / lmd * df2 / df1
    results.loc["Wilks' lambda", 'Num DF'] = df1
    results.loc["Wilks' lambda", 'Den DF'] = df2
    results.loc["Wilks' lambda", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Wilks' lambda", 'Pr > F'] = pval

    V = results.loc["Pillai's trace", 'Value']
    df1 = s * (2*m + s + 1)
    df2 = s * (2*n + s + 1)
    F = df2 / df1 * V / (s - V)
    results.loc["Pillai's trace", 'Num DF'] = df1
    results.loc["Pillai's trace", 'Den DF'] = df2
    results.loc["Pillai's trace", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Pillai's trace", 'Pr > F'] = pval

    U = results.loc["Hotelling-Lawley trace", 'Value']
    if n > 0:
        b = (p + 2*n) * (q + 2*n) / 2 / (2*n + 1) / (n - 1)
        df1 = p * q
        df2 = 4 + (p*q + 2) / (b - 1)
        c = (df2 - 2) / 2 / n
        F = df2 / df1 * U / c
    else:
        df1 = s * (2*m + s + 1)
        df2 = s * (s*n + 1)
        F = df2 / df1 / s * U
    results.loc["Hotelling-Lawley trace", 'Num DF'] = df1
    results.loc["Hotelling-Lawley trace", 'Den DF'] = df2
    results.loc["Hotelling-Lawley trace", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Hotelling-Lawley trace", 'Pr > F'] = pval

    sigma = results.loc["Roy's greatest root", 'Value']
    r = np.max([p, q])
    df1 = r
    df2 = v - r + q
    F = df2 / df1 * sigma
    results.loc["Roy's greatest root", 'Num DF'] = df1
    results.loc["Roy's greatest root", 'Den DF'] = df2
    results.loc["Roy's greatest root", 'F Value'] = F
    pval = stats.f.sf(F, df1, df2)
    results.loc["Roy's greatest root", 'Pr > F'] = pval
    return results


def _multivariate_ols_test(hypotheses, fit_results, exog_names,
                            endog_names):
    def fn(L, M, C):
        # .. [1] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm
        params, df_resid, inv_cov, sscpr = fit_results
        # t1 = (L * params)M
        t1 = L.dot(params).dot(M) - C
        # H = t1'L(X'X)^L't1
        t2 = L.dot(inv_cov).dot(L.T)
        q = matrix_rank(t2)
        H = t1.T.dot(inv(t2)).dot(t1)

        # E = M'(Y'Y - B'(X'X)B)M
        E = M.T.dot(sscpr).dot(M)
        return E, H, q, df_resid

    return _multivariate_test(hypotheses, exog_names, endog_names, fn)


@Substitution(hypotheses_doc=_hypotheses_doc)
def _multivariate_test(hypotheses, exog_names, endog_names, fn):
    """
    Multivariate linear model hypotheses testing

    For y = x * params, where y are the dependent variables and x are the
    independent variables, testing L * params * M = 0 where L is the contrast
    matrix for hypotheses testing and M is the transformation matrix for
    transforming the dependent variables in y.

    Algorithm:
        T = L*inv(X'X)*L'
        H = M'B'L'*inv(T)*LBM
        E =  M'(Y'Y - B'X'XB)M
    And then finding the eigenvalues of inv(H + E)*H

    .. [*] https://support.sas.com/documentation/cdl/en/statug/63033/HTML/default/viewer.htm#statug_introreg_sect012.htm

    Parameters
    ----------
    %(hypotheses_doc)s
    k_xvar : int
        The number of independent variables
    k_yvar : int
        The number of dependent variables
    fn : function
        a function fn(contrast_L, transform_M) that returns E, H, q, df_resid
        where q is the rank of T matrix

    Returns
    -------
    results : MANOVAResults
    """

    k_xvar = len(exog_names)
    k_yvar = len(endog_names)
    results = {}
    for hypo in hypotheses:
        if len(hypo) ==2:
            name, L = hypo
            M = None
            C = None
        elif len(hypo) == 3:
            name, L, M = hypo
            C = None
        elif len(hypo) == 4:
            name, L, M, C = hypo
        else:
            raise ValueError('hypotheses must be a tuple of length 2, 3 or 4.'
                             ' len(hypotheses)=%d' % len(hypo))
        if any(isinstance(j, str) for j in L):
            L = DesignInfo(exog_names).linear_constraint(L).coefs
        else:
            if not isinstance(L, np.ndarray) or len(L.shape) != 2:
                raise ValueError('Contrast matrix L must be a 2-d array!')
            if L.shape[1] != k_xvar:
                raise ValueError('Contrast matrix L should have the same '
                                 'number of columns as exog! %d != %d' %
                                 (L.shape[1], k_xvar))
        if M is None:
            M = np.eye(k_yvar)
        elif any(isinstance(j, str) for j in M):
            M = DesignInfo(endog_names).linear_constraint(M).coefs.T
        else:
            if M is not None:
                if not isinstance(M, np.ndarray) or len(M.shape) != 2:
                    raise ValueError('Transform matrix M must be a 2-d array!')
                if M.shape[0] != k_yvar:
                    raise ValueError('Transform matrix M should have the same '
                                     'number of rows as the number of columns '
                                     'of endog! %d != %d' %
                                     (M.shape[0], k_yvar))
        if C is None:
            C = np.zeros([L.shape[0], M.shape[1]])
        elif not isinstance(C, np.ndarray):
            raise ValueError('Constant matrix C must be a 2-d array!')

        if C.shape[0] != L.shape[0]:
            raise ValueError('contrast L and constant C must have the same '
                             'number of rows! %d!=%d'
                             % (L.shape[0], C.shape[0]))
        if C.shape[1] != M.shape[1]:
            raise ValueError('transform M and constant C must have the same '
                             'number of columns! %d!=%d'
                             % (M.shape[1], C.shape[1]))
        E, H, q, df_resid = fn(L, M, C)
        EH = np.add(E, H)
        p = matrix_rank(EH)

        # eigenvalues of inv(E + H)H
        eigv2 = np.sort(eigvals(solve(EH, H)))
        stat_table = multivariate_stats(eigv2, p, q, df_resid)

        results[name] = {'stat':stat_table, 'contrast_L':L,
                         'transform_M':M, 'constant_C':C}
    return results


class _MultivariateOLS(Model):
    """
    Multivariate linear model via least squares


    Parameters
    ----------
    endog : array_like
        Dependent variables. A nobs x k_endog array where nobs is
        the number of observations and k_endog is the number of dependent
        variables
    exog : array_like
        Independent variables. A nobs x k_exog array where nobs is the
        number of observations and k_exog is the number of independent
        variables. An intercept is not included by default and should be added
        by the user (models specified using a formula include an intercept by
        default)

    Attributes
    ----------
    endog : ndarray
        See Parameters.
    exog : ndarray
        See Parameters.
    """
    _formula_max_endog = None

    def __init__(self, endog, exog, missing='none', hasconst=None, **kwargs):
        if len(endog.shape) == 1 or endog.shape[1] == 1:
            raise ValueError('There must be more than one dependent variable'
                             ' to fit multivariate OLS!')
        super(_MultivariateOLS, self).__init__(endog, exog, missing=missing,
                                               hasconst=hasconst, **kwargs)
        self.nobs = self.endog.shape[0]

    def fit(self, method='svd'):
        self._fittedmod = _multivariate_ols_fit(
            self.endog, self.exog, method=method)
        return _MultivariateOLSResults(self)

class _MultivariateResults(): # Intended as parent class for _MultivariateOLSResults. NOT IMPLEMENTED
    """
    Class to contain multivariate model results

    Parameters
    ----------
    model : class instance
         the previously specified model instance (e.g. _MultivariateOLS)
    params : ndarray
         parameter estimates from the fit model
    """

    def __init__(self, _MultivariateOLS, params):
        self.initialize(_MultivariateOLS, params)
    
    def initialize(self, _MultivariateOLS, params):
        """
        Initialize (possibly re-initialize a _MultivariateResults instance
        """
        self.params = params
        self.model = _MultivariateOLS


class _MultivariateOLSResults(object):
    """
    _MultivariateOLS results class

    Attributes
    ----------
    params : pandas dataframe
         coefficients of intercept (if fit) and exog variables for each endog variable
    df_resid : int
         residual degrees of freedom
    hasconst : boolean
         True if model was fit with a constant, else False.
    formula : str
         R-style formula from which model was constructed.
    ssr : float or array of floats
         residual sum of squares. See method docstring for additional info.
    df_model : int
         model degrees of freedom (including constant if fit)
    loglike : float
         Value of log-liklihood function.
    AIC : float
         Akaike's information criteria. See method docstring for additional info.
         


    """
    def __init__(self, fitted_mv_ols):
        if (hasattr(fitted_mv_ols, 'data') and
                hasattr(fitted_mv_ols.data, 'design_info')):
            self.design_info = fitted_mv_ols.data.design_info
        else:
            self.design_info = None
        self.exog_names = fitted_mv_ols.exog_names
        self.endog_names = fitted_mv_ols.endog_names
        self.nobs = fitted_mv_ols.nobs        
        self._fittedmod = fitted_mv_ols._fittedmod


    @cache_readonly
    def params(self):
        """
        Returns
        ----------
        params : pandas dataframe
             coefficients of intercept (if fit) and exog variables for each endog variable

        dataframe shape:

                    Y1   Y2   ...
        intercept   .    .
        x1          .    .
        x2          .    .
        x3          .    .
        ...
        """
        params = self._fittedmod[0]
        
        return pd.DataFrame(params, columns=self.endog_names, index=self.exog_names)

    @cache_readonly
    def df_resid(self):
        return self._fittedmod[1]

    @cache_readonly
    def hasconst(self):
        """
        True if model was fit with a constant, else False.
        """
        if any(x in self.exog_names for x in ['const','Intercept']):
            return True
        else:
            return False
    
    @cache_readonly
    def df_model(self):
        """
        Total number of estimated parameters: e.g. number of dependent/exog variables plus constant if one was fit.
        """
        return self.params.shape[0]
        

    @cache_readonly
    def formula(self):
        """
        Extracts or builds R-style formula from model. For models created using `from_formula()`, returns
        `formula` attribute. Otherwise, constructs model formula using `exog_names` and `endog_names` attributes.
        Will aid in calling MANOVA.

        Returns
        ---------- 
        formula : str
            R-style formula from which model was constructed.
        """
        try:
            formula = self._fittedmod.formula
        except AttributeError:
            if self.hasconst:
                formula = " + ".join(self.endog_names) + " ~ " + " + ".join(self.exog_names[1:])
            else:
                formula = " + ".join(self.endog_names) + " ~ " + " + ".join(self.exog_names)
        
        return formula


    def ssr(self, multioutput="uniform_average"):
        """
        Multioutput handling: see sklearn.metrics.meansquarederror for reference
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
        
        Parameters
        ----------
        multioutput : string in ['raw_values', 'uniform_average'] or array-like of shape (n_outputs)
             Defines aggregating of multiple output values (e.g. for each Y/dependent/endog variable).
             Array-like value defines weights used to average errors.
        
        Returns
        ----------
        ssr : float or array of floats
             A non-negative floating point value (the best value is 0.0), or an
             array of floating point values, one for each individual Y/dependent/endog variable.
        """
        # get array of ssrs from tuple output of self._fittedmod
        ssrs = np.diagonal(self._fittedmod[3])
        if isinstance(multioutput, str):
            if multioutput == "uniform_average":
                ssr = np.mean(ssrs)
            elif multioutput == "raw_values":
                ssr = ssrs
            else:
                msg = "Expected either 'uniform_average' or 'raw_values', got {}.".format(multioutput)
                raise ValueError(msg)
            
            return ssr
        
        # if multioutput is array-like, treat as weights:
        elif hasattr(multioutput, "__len__"):
            if len(multioutput) != len(ssrs):
                msg = "Dimensions of output weights do not match dimensions of enodg/Y variables! Expected {}, got {}.".format(len(ssrs), len(multioutput))
                raise ValueError(msg)
            ssr = np.average(ssrs, weights = multioutput)
            return ssr
        else:
            msg = "Expected str ['uniform_average','raw_values'] or array-like, got {}".format(type(multioutput))
            raise ValueError(msg)


    @cache_readonly
    def loglike(self, scale=None):
        nobs = float(self.nobs)
        nobs2 = self.nobs / 2.0
        ssr=self.ssr()
        if scale is None:
            # profile log likelihood
            llf = -nobs2*np.log(2*np.pi) - nobs2*np.log(ssr / nobs) - nobs2
        else:
            # log-likelihood
            llf = -nobs2 * np.log(2 * np.pi * scale) - ssr / (2*scale)
        return llf
        
        
    @cache_readonly
    def aic(self):
        r"""
        Akaike's information criteria.
        
        :math:`-2llf + 2(df\_model)`.
        Note that df_model includes the constant (if fit).
        """
        return -2 * self.loglike + (2 * self.df_model)
    

    
    def __str__(self):
        return self.summary().__str__()

    @Substitution(hypotheses_doc=_hypotheses_doc)
    def mv_test(self, hypotheses=None):
        """
        Linear hypotheses testing

        Parameters
        ----------
        %(hypotheses_doc)s

        Returns
        -------
        results: _MultivariateOLSResults

        Notes
        -----
        Tests hypotheses of the form

            L * params * M = C

        where `params` is the regression coefficient matrix for the
        linear model y = x * params, `L` is the contrast matrix, `M` is the
        dependent variable transform matrix and C is the constant matrix.
        """
        k_xvar = len(self.exog_names)
        if hypotheses is None:
            if self.design_info is not None:
                terms = self.design_info.term_name_slices
                hypotheses = []
                for key in terms:
                    L_contrast = np.eye(k_xvar)[terms[key], :]
                    hypotheses.append([key, L_contrast, None])
            else:
                hypotheses = []
                for i in range(k_xvar):
                    name = 'x%d' % (i)
                    L = np.zeros([1, k_xvar])
                    L[i] = 1
                    hypotheses.append([name, L, None])

        results = _multivariate_ols_test(hypotheses, self._fittedmod,
                                          self.exog_names, self.endog_names)

        return MultivariateTestResults(results,
                                       self.endog_names,
                                       self.exog_names)

    def summary(self):
        raise NotImplementedError

class F_test_multivariate():  # this could probably just be a function
    """
    Performs F test to compare multivariate nested linear models.

    Usage: F_test_multivariate(restricted, unrestricted)
    
    Parameters
    ----------
    restricted : fit multivariate model
        Restricted/simple OLS linear model that has been fit to data.
        Must contain one fewer independent/exogenous variable than unrestricted model.
        expects class statsmodels.multivariate.multivariate_ols._MultivariateOLSResults
        
    unstricted : fit multivariate model
        Unrestricted/full/complex OLS linear model that has been fit to data.
        expects class statsmodels.multivariate.multivariate_ols._MultivariateOLSResults

    Attributes
    ----------
    ssr1 : float
        Residual sum of squares of restricted model
    ssr2 : float
        Residual sum of squares of unrestricted model
    df_model_1 : int
        Number of independent/exog variables in restricted model
    df_model_2 : int
        Number of independent/exog variables in unrestricted model
    df_resid : int
        Residual degrees of freedom for unrestricted model.
    nobs : int
        Number of observations. Should be equal for both models.
    
    F_statistic : float
        F statistic calculated from both models
    
    PrF : float
        P-value for F statistic. Is the difference in ssr between the two models 
        significantly more than would be expected by chance?
        
    results : dataframe
        Summarizes the parameters above.
    """
    def __init__(self, restricted, unrestricted):
        self.unrestricted = unrestricted # do I need this statement?
        self.restricted = restricted # do I need this statement?
        self.ssr1 = restricted.ssr()
        self.ssr2 = unrestricted.ssr()
        self.df_model_1 = restricted.df_model
        self.df_model_2 = unrestricted.df_model
        self.df_resid = unrestricted.df_resid
        if not self.df_model_2 > self.df_model_1:
            raise ValueError("Unrestricted model must have more model degrees of freedom than restricted model!")
        
        if restricted.nobs != unrestricted.nobs:
            raise ValueError("Number of observations (nobs) are not equivalent between models!")
        self.nobs = restricted.nobs
    
    @cache_readonly
    def F_statistic(self):
        """
        '1' refers to restricted model
        '2' refers to unrestricted model
        """
        num = (self.ssr1 - self.ssr2) / (self.df_model_2 - self.df_model_1)
        # NOTE: statsmodels.stats.anova_lm() has a "scale" parameter here. I'm not sure what that is.
        denom = self.ssr2 / (self.nobs - self.df_model_2)
        F_stat = num / denom
        return F_stat
    
    @cache_readonly
    def prF(self):
        df_diff = self.df_model_2 - self.df_model_1
        return stats.f.sf(self.F_statistic, df_diff, self.df_resid)
    
    @cache_readonly
    def results(self):
        models=[self.restricted, self.unrestricted]
        cols=['formula','df_resid','ssr', 'model_df','model_df_diff','ssr_diff','F_stat','Pr>F']
        df = pd.DataFrame(np.zeros((2,len(cols))), index=["restricted model","unrestricted model"], 
                          columns=cols)
        
        df['formula'] = [mdl.formula for mdl in models]
        df['df_resid'] = [mdl.df_resid for mdl in models]
        df['ssr'] = [mdl.ssr() for mdl in models]
        df['model_df'] = [mdl.df_model for mdl in models]
        df['model_df_diff'] = df['model_df'].diff()
        df['ssr_diff'] = df['ssr'].diff()
        df.loc['unrestricted model','F_stat'] = self.F_statistic
        df.loc['unrestricted model','Pr>F'] = self.prF
        
        df = df.replace(0, np.nan)
        return df


class MultivariateTestResults(object):
    """ Multivariate test results class
    Returned by `mv_test` method of `_MultivariateOLSResults` class

    Attributes
    ----------
    results : dict
       For hypothesis name `key`:
           results[key]['stat'] contains the multivariate test results
           results[key]['contrast_L'] contains the contrast_L matrix
           results[key]['transform_M'] contains the transform_M matrix
           results[key]['constant_C'] contains the constant_C matrix
    endog_names : str
    exog_names : str
    summary_frame : multiindex dataframe
        Returns results as a multiindex dataframe
    """
    def __init__(self, mv_test_df, endog_names, exog_names):
        self.results = mv_test_df
        self.endog_names = endog_names
        self.exog_names = exog_names

    def __str__(self):
        return self.summary().__str__()

    def __getitem__(self, item):
        return self.results[item]

    @property
    def summary_frame(self):
        """
        Return results as a multiindex dataframe
        """
        df = []
        for key in self.results:
            tmp = self.results[key]['stat'].copy()
            tmp.loc[:, 'Effect'] = key
            df.append(tmp.reset_index())
        df = pd.concat(df, axis=0)
        df = df.set_index(['Effect', 'index'])
        df.index.set_names(['Effect', 'Statistic'], inplace=True)
        return df

    def summary(self, show_contrast_L=False, show_transform_M=False,
                show_constant_C=False):
        """

        Parameters
        ----------
        contrast_L : True or False
            Whether to show contrast_L matrix
        transform_M : True or False
            Whether to show transform_M matrix
        """
        summ = summary2.Summary()
        summ.add_title('Multivariate linear model')
        for key in self.results:
            summ.add_dict({'':''})
            df = self.results[key]['stat'].copy()
            df = df.reset_index()
            c = df.columns.values
            c[0] = key
            df.columns = c
            df.index = ['', '', '', '']
            summ.add_df(df)
            if show_contrast_L:
                summ.add_dict({key:' contrast L='})
                df = pd.DataFrame(self.results[key]['contrast_L'],
                                  columns=self.exog_names)
                summ.add_df(df)
            if show_transform_M:
                summ.add_dict({key:' transform M='})
                df = pd.DataFrame(self.results[key]['transform_M'],
                                  index=self.endog_names)
                summ.add_df(df)
            if show_constant_C:
                summ.add_dict({key:' constant C='})
                df = pd.DataFrame(self.results[key]['constant_C'])
                summ.add_df(df)
        return summ
