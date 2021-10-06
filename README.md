# Python XGBoost 
I implemented xgboost trees from scratch! 

<p float="left", align='center'>
  <img src="/pred_sin.png" width="400" />
  <img src="/pred_2dgaussian.png" width="400" /> <br>
  (Left) Fitting a noisy sine wave with varius number of trees (Right) Fitting 2 dimensional gaussian data (in blue)
</p>

### Resources:
* [Original paper](https://arxiv.org/pdf/1603.02754.pdf)
* [StatQuest playlist](https://www.youtube.com/watch?v=OtD8wVaFm6E&list=PLblh5JKOoLULU0irPgs1SnKO6wqVjKUsQ&index=1&ab_channel=StatQuestwithJoshStarmer)
* [Analytics Vidhya Blog post](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/?utm_source=blog&utm_medium=4-boosting-algorithms-machine-learning)



## Why?
I've heard a lot about xgboost and gradient boosted trees, but never went deeply into understanding how they worked before. XGBoost is very popular, and I think understanding how it works could be very useful! Plus there really isn't a better way to gain understanding compared to implementing something from scratch. 

## What I did:
### Part 1:
Read through and watched the reasources I listed above, then proceeded to implement an ExtremeBoostedTree class that follows the greedy split algorithm described in the paper (for regression). I tested the algorithm with a sinusoidal function. 

### Part 2:
Added ensembling and adapted code to be able to use any loss function (given its 1st and 2nd derivatives). 

### Part 3:
Reconfigured everything to accept any dimensional data and added approximate splitting. 

### Timeline:
* (Oct 3) Created a basic extreme gradient boost tree for 1 dimensional data
* (Oct 4) Added ensembling and generalized greedy algorithm to any loss function (given 1st and 2nd derivative functions)
* (Oct 5) Implemented multi-dim and approximate splitting
