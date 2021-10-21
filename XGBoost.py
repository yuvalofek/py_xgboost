from XGBoostTree import XGBoostTree


class XGBoost:
  def __init__(self, n_trees , max_depth=6, gamma = 0, lambd=0, eta=0.3):
    self.trees = [XGBoostTree(max_depth=max_depth, gamma = gamma, lambd=lambd) for _ in range(n_trees)]
    self.eta = eta
    self.baseline = None

  def fit(self, X,Y):
    if len(X) != len(Y):
      raise ValueError('Input lengths do not match')
      return 
    # Get the average for a baseline
    self.baseline = sum(Y)/len(Y)

    Y_ = Y
    # Loop over the trees, building them one at a time, and update the prediction each run
    for i in range(len(self.trees)):
      y_hat = self.predict(X, i)
      x_sort, y_sort = zip(*sorted(zip(X, y_hat)))
      # How much y is left to learn:
      Y_ = [y-p for y,p in zip(Y,y_hat)]
      self.trees[i].fit(X,Y_,y_hat)
      

  def predict(self, X, n_trees=None):
    if self.baseline is None:
      raise ValueError('Model not trained!')
      return
    
    # initialize prediction 
    y_pred = [self.baseline]*len(X)

    # parallelization potential!!
    trees = self.trees
    if n_trees is not None:
      trees = self.trees[:n_trees]
    for i in range(len(trees)):
      y_pred = [y_delta*self.eta+y_h for y_delta,y_h in zip(self.trees[i].predict(X), y_pred)]
    return y_pred

  def print_trees(self):
    print(f'Baseline: {self.baseline}')
    for i, tree in enumerate(self.trees):
      print(f'Tree {i}:')
      tree.print_tree()

