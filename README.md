# Python XGBoost 
I implemented xgboost trees from scratch! 

### Resources:
* [Original paper](https://arxiv.org/pdf/1603.02754.pdf)
* [StatQuest playlist](https://www.youtube.com/watch?v=OtD8wVaFm6E&list=PLblh5JKOoLULU0irPgs1SnKO6wqVjKUsQ&index=1&ab_channel=StatQuestwithJoshStarmer)
* [Analytics Vidhya Blog post](https://www.analyticsvidhya.com/blog/2018/09/an-end-to-end-guide-to-understand-the-math-behind-xgboost/?utm_source=blog&utm_medium=4-boosting-algorithms-machine-learning)


Timeline:
* (Oct 3) Created a basic extreme gradient boost tree for 1 dimensional data
* (Oct 4) Added ensembling and generalized greedy algorithm to any loss function (given 1st and 2nd derivative functions)


## Why?
I've heard a lot about xgboost and gradient boosted trees, but never went deeply into understanding how they worked before. XGBoost is very popular, and I think understanding how it works could be very useful! Plus there really isn't a better way to gain understanding compared to implementing something from scratch. 

## What I did:
### Part 1:
Read through and watched the reasources I listed above, then proceeded to implement an ExtremeBoostedTree class that follows the greedy split algorithm described in the paper (for regression). I tested the algorithm with a sinusoidal function. 

### Part 2:
Added ensembling and adapted code to be able to use any loss function (given its 1st and 2nd derivatives). 

## What I did that I didn't like:
2. Only implemented the greedy split algorithm (not sparse-aware or approximate as described in the paper)
3. Current implementation (Oct 3) only accepts one dimensional data 

## How I could fix the above:
1 + 2. Implement these! (Maybe in at a later day)

3. Add random feature selection for the splits

