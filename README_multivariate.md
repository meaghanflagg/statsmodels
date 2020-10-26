### Forked version with modifications to multivariate Tools

============

* Minor modifications that improve the usability of Multivariate OLS.  
* Added F test to compare nested multivariate models.
* Addition of model_selection module that functionalizes forward stepwise model selection for multivariate OLS models.



#### Note: This version is still under active development
It is recommended
to install in developer mode:  
```bash
git clone https://github.com/meaghanflagg/statsmodels.git
cd statsmodels
pip install -e .
# OR
python setup.py develop # I had problems with this
```


### Main files

============

statsmodels/multivariate/multivariate_ols.py  
* Added useful attributes to _MultivariateOLSResults class, including ssr, AIC, llf, etc. Modeled after attributes in Univariate OLSResults class.
* Added class F_test_multivariate. Performs F test to compare nested
  multivariate OLS models, returns dataframe of results. Modeled after lm.anova.


statsmodels/multivariate/model_selection.py  
  * Added module, which currently contains forward_stepwise function. Given a dataset and lists of X and Y variables, this function loops through X terms and iteratively adds terms that best capture variance in the dataset (by ssr or manova p value). Returns a dataframe of potential models and associated statistics, so the user can select the most appropriate model.

Multivariate_OLS_testing.ipynb  
* Jupyter notebook demonstrating function and validation of modifications.
