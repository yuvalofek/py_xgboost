class Node:
  def __init__(self, value, depth):
    self.value = value
    self.depth = depth
    self.gain = None
    self.left = None
    self.right = None


class XGBoostTree:
  def __init__(self, max_depth=6, gamma = 0, lambd=0,
               g=lambda y, y_h: y_h-y, h= lambda y, y_h: 1):
    self.max_depth = max_depth
    self.root = None
    self.gamma = gamma
    self.lambd = lambd

    # Equations to get the gradient and hermetian (1st and 2nd derivatives)
    # This is for 
    self.g = g
    self.h = h

  @staticmethod
  def __similarity(residuals, lambd=0):
    # Calculate similarity score of residuals
    return sum(residuals)/(len(residuals) + lambd)

  @staticmethod
  def __gain(right, left, root):
    # calculate gain
    return right + left - root

  def __get_split_greedy(self, X, Y, Y_HAT):
    if len(X) <= 1:
      # If we get one element, no split
      return None, None

    # Compute G^2 / (H + lambda)
    g_j = [self.g(y, y_h) for y, y_h in zip(Y, Y_HAT)]
    h_j = [self.h(y, y_h) for y, y_h in zip(Y, Y_HAT)]
    G = sum(g_j)
    H = sum(h_j)
    sim_root = G**2/(H+self.lambd)

    # sort residuals by x
    X, g_j, h_j = zip(*sorted(zip(X,g_j, h_j)))

    # Find split that results in least gain
    max_gain = -float('infinity')
    best_split = None
    for r in range(len(X)-1):
      split = sum(X[r:r+2])/2

      # Compute G_L^2 / (H_L + lambda)
      G_L = sum(g_j[:r+1])
      H_L = sum(h_j[:r+1])
      sim_left = G_L**2/(H_L+self.lambd)

      # Compute G_R^2 / (H_R + lambda)
      G_R = sum(g_j[r+1:])
      H_R = sum(h_j[r+1:])
      sim_right = G_R**2/(H_R+self.lambd)

      # Get the score
      g = self.__gain(sim_right, sim_left, sim_root)

      # Find min 
      if g > max_gain:
        max_gain = g
        best_split = split
    return best_split, max_gain

  def __fit(self, node, max_depth):
    """
    Recursively fit data 
    node - node object to fit to data, node.value contains data to fit
    depth - depth of node
    max_depth - maximum depth allowed
    """
    X, Y, Y_HAT = zip(*node.value)
    split, gain = self.__get_split_greedy(X,Y, Y_HAT)
    if split is None:
      # If there is no split --> one element, set prediction = label of element
      node.right = None
      node.left = None
      node.value = [Y[0]]
      return 
    
    if node.depth > max_depth:
      node.value = list(Y)
      return 

    # update node values
    node.value = split
    node.gain = gain

    # Build the rest of the tree
    node.right = Node([(x,y, y_hat) for x, y, y_hat in zip(X, Y, Y_HAT) if x>node.value], node.depth+1)
    self.__fit(node.right,  max_depth)
    node.left = Node([(x,y, y_hat) for x, y, y_hat in zip(X, Y, Y_HAT) if x<=node.value], node.depth+1)
    self.__fit(node.left, max_depth)

  def fit(self, X, Y, Y_HAT):
    self.root = Node(zip(X,Y,Y_HAT), 1)
    self.__fit(self.root, self.max_depth)
    if self.gamma is not None:
      self.prune()

  def __prune(self, node):
    """
    Recursive pruning using a gamma parameter
    """
    if node.left is None:
      # If leaf, return that it is pruned
      return True
    
    # prune children
    prune_left = self.__prune(node.left)
    prune_right = self.__prune(node.right)

    if (prune_left and prune_right):
      # if we pruned both children, check for pruning
      if node.gain < self.gamma:
        # If need to prune, take the values of the children and store them, clean out node
        node.value = node.left.value + node.right.value
        node.left = None
        node.right = None
        node.gain = None
        return True
    return False

  def prune(self):
    self.__prune(self.root)

  def __print_tree(self, node):
    """
    Recursively print tree
    """
    if node.value is None:
      return
    print('    '*node.depth + str(node.value), f'({node.gain})')
    if node.left is not None:
      self.__print_tree(node.left)
      self.__print_tree(node.right)

  def print_tree(self):
    """ 
    call the recursive print using the tree root
    """
    self.__print_tree(self.root)

  def __get_output(self, Y):
    return sum(Y)/(len(Y)+self.lambd)

  def __predict(self, x, node):
    """
    Recursively predict label for input value x
    """
    if node.left is None:
      return self.__get_output(node.value)
    
    if x > node.value:
      return self.__predict(x, node.right)
    else:
      return self.__predict(x, node.left)

  def predict(self, X):
    """
    Predict labels for iterable X
    """
    y = []
    for x in X:
      y.append(self.__predict(x, self.root))
    return y

