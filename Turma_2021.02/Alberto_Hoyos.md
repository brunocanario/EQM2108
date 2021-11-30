## Final project. Course: EQM 2108
## Title: Modeling and optimization of methane oxidative coupling using machine learning (Optuna and Particle Swarm)
## Student: Carlos Alberto Castro Hoyos
## Professor: Amanda Lemette 

## Introduction
### The oxidative coupling of methane (OCM) is an important route for the direct transformation of methane (from natural gas) to more valuable hydrocarbons. Nowadays, based on the importance of producing hydrocarbons for any human needs, it is required to develop new methods for obtention of these compounds (including: modeling, simulation and optimization). 
### Precisely, the main aim of this article is to show an optimization method sourced on Machine Learning, testing two techniques (Optuna and Particle Swarm) validating the performance of the model, according the variation/influence of a meta estimator´s hyperparameters. This estimator (random forest) fits a number of decision tree classifiers on various sub-samples of the dataset and uses averaging to improve the predictive accuracy and control over-fitting.

## Methodology
## This evaluation is divided into 4 steps for Optuna:
## The first step: Define the input and output variables provided from the dataset (OCM). According to the database, five variables are set as input variables (methane molar fraction, methane/oxygen feeding molar flow, temperature, pressure and volumetric flow) and seven variables as output variables (methane conversion, oxygen conversion, selectivity for C2, yield for C2, ethane yield, carbon dioxide yield and carbon monoxide yield).
## The second step: Set the output variables used on the model (yield of C2, ethane yield and carbon dioxide yield). For the proposal of this article, these output variables are selected randomly. This definition is shown in the figure No. 1.

<center><img src="https://github.com/amandalemette/EQM2108/blob/7551e8ce5d9ad368fcbb9501b23732d4edfd157d/Turma_2021.02/Imagens/1.png?raw=true"  width=900 height=525 /><center>

###### Figure No. 1

##  On this step, the assessment of the correlation among variables is done through the analysis of the correlation matrix.

## The third step: Evaluate the influence of the hyperparameters defined by the meta estimator on the model, through a sensitivity analysis, computing R^2 (the coefficient of determination). The hyperparameters to be assessed are: random_state, min_samples_split, min_samples_leaf, max_depth, n_estimators, bootstrap, warm_start, verbose, oob_score, n_jobs and max_samples. The variation on the upper limit for every hyperparameter must be tested, based on the R^2 achieved. This is shown in the figure No. 2

https://github.com/amandalemette/EQM2108/blob/504d08d0dcac5a9fc5774ca35305779df9d2178f/Turma_2021.02/Imagens/2.png
###### Figure No. 2

## As these elements (hyperparameters) are intended to be assessed, it is needed to define the dataset split (including the validation set) previously. The dataset is split into: Training (70%), testing (15%) and validation (15%).
## The fourth step: Select the hyperparameters with the most important impact on the model. These must be applied on the model again, expecting to obtain high performance (R^2 > 0.95) for the three splits (training, testing and validation). 
## The evaluation is divided into 3 steps for Particle Swarm: 
## The first step: Reply the selection dataset (definition of variables and uploading the dataset) from Optuna.
## The second step: Define the dataset split (including the validation set). The dataset is split into: Training (70%), testing (15%) and validation (15%).
## The third step: Select the same optimized hyperparameters from Optuna and apply the optimization model (particle swarm), expecting to obtain high performance (R^2 > 0.95).  It is need to set the value for: hyperparameters of particle swarm (c1,c2 and w), number of particles, then number of iterations and the minimum limit for the hyperparameters. 
## Results:
## Optuna: 
## It is obtained the correlation matrix as it is shown in the figure No. 3. Basically, it indicates that any input variable may be excluded because there is no correlation among them (any value different from 1 in the same row).

https://github.com/amandalemette/EQM2108/blob/b5dc1007cda97262a63e52a886eb47d4bf45a6d5/Turma_2021.02/Imagens/figure%201.png
###### Figure No. 3

## It is plotted the relation between the number of trials and the objective value. Basically, the result showed in the figure No. 4, indicates that as it was expected, during the training section the performance of the model is not high. However, as the testing section is achieved, the model is able to optimize and to reach the value of the objective function. Subsequently, this performance is confirmed as the validation section is achieved. The final R^2 got by setting the upper limits as 100 for every hyperparameter (excluding n_jobs), is equal to 0.9913787083312774.

https://github.com/amandalemette/EQM2108/blob/278ac6d7e7f520789287f54781fe4c0315001166/Turma_2021.02/Imagens/figure%202.png
###### Figure No. 4 

## It is acquired that min_samples_leaf, max_samples, min_samples_split and max_depth represent the most sensitivity on the model. This information is shown in the figure No. 5. 

https://github.com/amandalemette/EQM2108/blob/25de7b31e4e01e7e3b0b943a4881be330e1bace7/Turma_2021.02/Imagens/figure%203.png
###### Figure No. 5

## The values achieved for these hyperparameters are equal to: 12, 82, 11 and 66 respectively.  
## It is obtained high performances for every section: training (0.9882267866668047), testing (0.9877439463081822) and validation (0.9870512920212109).
## Particle Swarm:
## As the assessment parameter is set as (1-R2val), the PSO found the best solution at (min_samples_leaf = 60, max_samples = 5598, min_samples_split = 60, max_depth = 5598), with R^2 equal to (1 - 0.00796172584949384 = 0.99203827415). This value was achieved as these factors are settled: The minimum limit = 60, (c1, c2 and w) = 100, the number of iterations = 4  and the number of particles = 100.
## Discussion:
## Regarding the assessment of the performance for every hyperparameter using Optuna, it is reported that through modifying their upper limits, high/low performances (R^2) are achieved (variations). Basically, these variations are explained by the simple exploration of hyperparameters (an implicit criterion of the meta estimator Random Forest). So, in order to achieve high performances, it is required to understand the effect of every hyperparameter on the model, to check if they can be applied on the required model and just to apply "trial and error" on the objective function.  
## It is said that the more data is used, the more computing time, the more effort and the more memory is required. So, according to the results provided by this study in order to test high limit values for the hyperparameters, it is required to have high memory capacity on the virtual machine. So, the more proximity to high performances (R^2 > 0.9) for every testing, the less computing time, the less effort and the faster results are achieved. 
## Respect the particle swarm model, 
## Conclusion
## References
## 1) Tam, A. (2021). A Gentle Introduction to Particle Swarm Optimization. Retrieved 26 November 2021, from https://machinelearningmastery.com/a-gentle-introduction-to-particle-swarm-optimization/
## 2) optuna.visualization — Optuna 2.10.0 documentation. (2021). Retrieved 26 November 2021, from https://optuna.readthedocs.io/en/stable/reference/visualization/index.html
## 3) Scikit-learn: machine learning in Python — scikit-learn 1.0.1 documentation. (2021). Retrieved 26 November 2021, from https://scikit-learn.org/stable/.
## 4) Welcome to PySwarms’s documentation! — PySwarms 1.3.0 documentation. (2021). Retrieved 26 November 2021, from https://pyswarms.readthedocs.io/en/latest/



