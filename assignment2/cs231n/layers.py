import numpy as np


def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
  examples, where each example x[i] has shape (d_1, ..., d_k). We will
  reshape each input into a vector of dimension D = d_1 * ... * d_k, and
  then transform it to an output vector of dimension M.

  Inputs:
  - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
  - w: A numpy array of weights, of shape (D, M)
  - b: A numpy array of biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################

  ##reshape
  d_dims = x.shape[1:]
  N = x.shape[0]

  secondDim = reduce( lambda x, y: x*y, d_dims)
  intoRows = np.reshape( x, (N, secondDim) )

  out = np.dot( intoRows, w ) + b

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################

  db = np.sum( dout, axis=0 )

  ##reshape X again
  d_dims = x.shape[1:]
  N = x.shape[0]
  if len( d_dims  ) > 1:
    secondDim = reduce(lambda x, y: x * y, d_dims)
  else:
    secondDim = d_dims[0]
  intoRows = np.reshape(x, (N, secondDim))

  ##should get shape of (D, M)
  ## intoRows.T . dout = (D, N) . (N, M)
  #print( dout.shape )
  #print( " ")
  #print( intoRows.shape )
  dw = np.dot( intoRows.T, dout )

  ## have to reshape the weights here
  temp = np.dot( dout, w.T )
  dx = np.reshape( temp, x.shape )

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum( 0, x )
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dout[ x <= 0]=0

  dx = dout
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx

def batchnorm_forward(x, gamma, beta, bn_param):
  """
  Forward pass for batch normalization.

  During training the sample mean and (uncorrected) sample variance are
  computed from minibatch statistics and used to normalize the incoming data.
  During training we also keep an exponentially decaying running mean of the mean
  and variance of each feature, and these averages are used to normalize data
  at test-time.

  At each timestep we update the running averages for mean and variance using
  an exponential decay based on the momentum parameter:

  running_mean = momentum * running_mean + (1 - momentum) * sample_mean
  running_var = momentum * running_var + (1 - momentum) * sample_var

  Note that the batch normalization paper suggests a different test-time
  behavior: they compute sample mean and variance for each feature using a
  large number of training images rather than using a running average. For
  this implementation we have chosen to use running averages instead since
  they do not require an additional estimation step; the torch7 implementation
  of batch normalization also uses running averages.

  Input:
  - x: Data of shape (N, D)
  - gamma: Scale parameter of shape (D,)
  - beta: Shift paremeter of shape (D,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: of shape (N, D)
  - cache: A tuple of values needed in the backward pass
  """
  mode = bn_param['mode']
  eps = bn_param.get('eps', 1e-5)
  momentum = bn_param.get('momentum', 0.9)

  N, D = x.shape
  running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
  running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

  out, cache = None, None
  if mode == 'train':
    #############################################################################
    # TODO: Implement the training-time forward pass for batch normalization.   #
    # Use minibatch statistics to compute the mean and variance, use these      #
    # statistics to normalize the incoming data, and scale and shift the        #
    # normalized data using gamma and beta.                                     #
    #                                                                           #
    # You should store the output in the variable out. Any intermediates that   #
    # you need for the backward pass should be stored in the cache variable.    #
    #                                                                           #
    # You should also use your computed sample mean and variance together with  #
    # the momentum variable to update the running mean and running variance,    #
    # storing your result in the running_mean and running_var variables.        #
    #############################################################################

    sample_mean = np.mean( x, axis=0 )
    sample_var = np.var( x, axis=0 )

    running_mean = momentum * running_mean + ( 1 - momentum ) * sample_mean
    running_var = momentum * running_var + ( 1 - momentum ) * sample_var


    stdev = np.sqrt( sample_var ) + eps
    x_hat = ( x - sample_mean ) * ( stdev ** -1.0 )
    gamma_x = x_hat * gamma
    out = gamma_x + beta

    cache = (x, sample_mean, sample_var, x_hat, stdev, gamma, beta, eps )

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  elif mode == 'test':
    #############################################################################
    # TODO: Implement the test-time forward pass for batch normalization. Use   #
    # the running mean and variance to normalize the incoming data, then scale  #
    # and shift the normalized data using gamma and beta. Store the result in   #
    # the out variable.                                                         #
    #############################################################################

    sample_mean = np.mean(x, axis=0)
    sample_var = np.var(x, axis=0)

    stdev = np.sqrt(sample_var) + eps
    x_hat = ( x - running_mean )  * ( stdev ** -1.0 )

    gamma_x = x_hat * gamma
    out = gamma_x + beta
    cache = (x, sample_mean, sample_var, x_hat, stdev, gamma, beta, eps )

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
  else:
    raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

  # Store the updated running means back into bn_param
  bn_param['running_mean'] = running_mean
  bn_param['running_var'] = running_var

  return out, cache

def batchnorm_backward(dout, cache):
  """
  Backward pass for batch normalization.

  For this implementation, you should write out a computation graph for
  batch normalization on paper and propagate gradients backward through
  intermediate nodes.

  Inputs:
  - dout: Upstream derivatives, of shape (N, D)
  - cache: Variable of intermediates from batchnorm_forward.

  Returns a tuple of:
  - dx: Gradient with respect to inputs x, of shape (N, D)
  - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
  - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
  """

  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #############################################################################

  #print len(cache)

  x, sample_mean, sample_var, x_hat, stdev, gamma, beta, eps = cache

  mu_sub = ( x - sample_mean)
  N,D = mu_sub.shape
  dbeta = np.sum( dout, axis = 0 )
  dgamma = np.sum( dout * x_hat, axis=0 )
  dxhat = dout * gamma
  dmu_sub = dxhat * ( stdev ** -1.0)

  divar = np.sum( dxhat*(x-sample_mean), axis=0)
  d_var =  -1.0 /(stdev**2) * divar
  dr_var = 0.5 * (1.0 / np.sqrt(sample_var + eps)) * d_var
  dsq = 1.0 / N * np.ones((N, D)) * dr_var

  dmu_sub += 2 * mu_sub * dsq
  dx = dmu_sub
  dmu = -1.0 * np.sum( dmu_sub, axis=0 )
  dx += 1.0 / N * np.ones((N, D)) * dmu

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
  """
  Alternative backward pass for batch normalization.

  For this implementation you should work out the derivatives for the batch
  normalizaton backward pass on paper and simplify as much as possible. You
  should be able to derive a simple expression for the backward pass.

  Note: This implementation should expect to receive the same cache variable
  as batchnorm_backward, but might not use all of the values in the cache.

  Inputs / outputs: Same as batchnorm_backward
  """
  dx, dgamma, dbeta = None, None, None
  #############################################################################
  # TODO: Implement the backward pass for batch normalization. Store the      #
  # results in the dx, dgamma, and dbeta variables.                           #
  #                                                                           #
  # After computing the gradient with respect to the centered inputs, you     #
  # should be able to compute gradients with respect to the inputs in a       #
  # single statement; our implementation fits on a single 80-character line.  #
  #############################################################################

  ##Doesn't work too well
  x, sample_mean, sample_var, x_hat, stdev, gamma, beta, eps = cache
  N, D = x.shape

  dbeta = np.sum(dout, axis=0)
  dgamma = np.sum(dout * x_hat, axis=0)

  dvar = -0.5 * (stdev ** (-3.0)) * gamma * np.sum(dout * (x - sample_mean), axis=0)
  dmean = - gamma * (stdev) ** (-1.0) * np.sum(dout, axis=0) - np.sum(2.0 * dvar * (x - sample_mean) / N, axis=0)
  dx = (dout * gamma) * (stdev ** -1.0) + 2.0*dvar*(x - sample_mean)/N + dmean/N

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta

def dropout_forward(x, dropout_param):
  """
  Performs the forward pass for (inverted) dropout.

  Inputs:
  - x: Input data, of any shape
  - dropout_param: A dictionary with the following keys:
    - p: Dropout parameter. We drop each neuron output with probability p.
    - mode: 'test' or 'train'. If the mode is train, then perform dropout;
      if the mode is test, then just return the input.
    - seed: Seed for the random number generator. Passing seed makes this
      function deterministic, which is needed for gradient checking but not in
      real networks.

  Outputs:
  - out: Array of the same shape as x.
  - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
    mask that was used to multiply the input; in test mode, mask is None.
  """
  p, mode = dropout_param['p'], dropout_param['mode']
  if 'seed' in dropout_param:
    np.random.seed(dropout_param['seed'])

  mask = None
  out = None

  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase forward pass for inverted dropout.   #
    # Store the dropout mask in the mask variable.                            #
    ###########################################################################

    mask = ( np.random.rand( *x.shape ) < p ) / p

    out = mask * x

    cache = ( dropout_param, mask )

    return out, cache

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    ###########################################################################
    # TODO: Implement the test phase forward pass for inverted dropout.       #
    ###########################################################################

    cache = ( dropout_param, None )
    return x, cache

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################

  cache = (dropout_param, mask)
  out = out.astype(x.dtype, copy=False)

  return out, cache


def dropout_backward(dout, cache):
  """
  Perform the backward pass for (inverted) dropout.

  Inputs:
  - dout: Upstream derivatives, of any shape
  - cache: (dropout_param, mask) from dropout_forward.
  """
  dropout_param, mask = cache
  mode = dropout_param['mode']
  
  dx = None
  if mode == 'train':
    ###########################################################################
    # TODO: Implement the training phase backward pass for inverted dropout.  #
    ###########################################################################

    dropout_param, mask = cache

    dx = mask * dout

    ###########################################################################
    #                            END OF YOUR CODE                             #
    ###########################################################################
  elif mode == 'test':
    dx = dout
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################

  N, C, H, W, F, C, HH, WW, P, S = ( x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], w.shape[1], w.shape[2], w.shape[3], conv_param['pad'], conv_param['stride'])

  ## if H == W these two should be the same
  W_prime = (W-WW+2*P)/S+1
  H_prime = (H-HH+2*P)/S+1

  out = np.zeros(shape=(N, F, H_prime, W_prime))

  ##pad that shit!!
  npad = ( (0,0), (0,0), (P,P), (P,P) )
  x = np.pad( x, npad, 'constant', constant_values=(0))

  ##for each of our n training data
  for d in range( 0, N):
    ##grab datum
    datum =  x[d]

    ##for each of our f filters
    for f in range( 0, F):
      ##this set of weights
      weights = w[f]

      ###make activation map for this filter
      ##we will think of this as the input being a large rectangle where we go as far to the right as possible, then move down and repeat
      for h in range(0, H_prime):
        for a in range(0, W_prime):
          out[d][f][h][a] = np.sum(  datum[:, h * S : HH + h * S , a * S : a * S + WW ] *  weights ) + b[f]

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)

  #print out.shape
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################

  ##print dout.shape

  x, w, b, conv_param = cache

  N, C, H, W, F, C, HH, WW, P, S = (
  x.shape[0], x.shape[1], x.shape[2], x.shape[3], w.shape[0], w.shape[1], w.shape[2], w.shape[3], conv_param['pad'],
  conv_param['stride'])

  ##remove padding from H and W
  H = H - 2*P
  W = W - 2*P

  ## if H == W these two should be the same
  W_prime = (W - WW + 2 * P) / S + 1
  H_prime = (H - HH + 2 * P) / S + 1

  db = np.zeros( shape=(F))

  ## for db
  for n in range(0, N):
    for f in range(0, F):
      db[f] += np.sum( dout[n][f] )

  dx = np.zeros(shape=(N, C, (H + 2* P), (W + 2 *P)))
  ##dx
  for n in range( 0, N ):
    for f in range( 0, F ):
      grads = dout[n][f]
      for h in range( 0, H_prime ):
        for s in range( 0, W_prime):
          dx[n,:, h * S : HH + h * S , s * S : s * S + WW ] += w[f] *  grads[h][s]
  ##trim padding
  dx = dx[:,:,1:-1,1:-1]


  ##print dx[0]
  dw = np.zeros( shape=( F, C, HH, WW ))
  for f in range( 0, F ):
    thisFilter = np.zeros( shape=(C, HH, WW) )
    for h in range( 0, H_prime ):
      for s in range( 0, W_prime ):
        for n in range( 0, N ):
          grads = dout[n][f]
          thisN = x[n]
          thisFilter += thisN[:, h * S : HH + h * S , s * S : s * S + WW ] * grads[h][s]
    dw[f] = thisFilter

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################

  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  N, C, H, W = x.shape
  vert = ( H - pool_height ) / stride + 1
  horiz = ( W - pool_width ) / stride + 1

  out = np.zeros( shape=(N,C,vert,horiz))

  for n in range( 0, N):
    for c in range( 0, C):
      for v in range(0, vert):
        for h in range(0, horiz):
          out[n][c][v][h] = x[n][c][ stride * v : stride * v + pool_height, stride*h : stride*h + pool_width].max(axis=0).max()

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################

  x, pool_param = cache
  pool_height, pool_width, stride = pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']
  N, C, H, W = x.shape
  vert = (H - pool_height) / stride + 1
  horiz = (W - pool_width) / stride + 1

  dx = np.zeros( shape=(N,C,H,W))

  for n in range(0, N):
    for c in range(0, C):
      for v in range(0, vert):
        for h in range(0, horiz):
          ##find index of max in x
          sect = x[n][c][stride * v: stride * v + pool_height, stride * h: stride * h + pool_width].flatten()
          ##for dx
          ret = np.zeros( shape=( sect.shape ))
          max_idx = sect.argmax()
          ret[max_idx] = dout[n][c][v][h]
          ##throw into gradients for dx
          dx[n][c][stride * v: stride * v + pool_height, stride * h: stride * h + pool_width] = ret.reshape( ( pool_height, pool_width))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
  """
  Computes the forward pass for spatial batch normalization.

  Inputs:
  - x: Input data of shape (N, C, H, W)
  - gamma: Scale parameter, of shape (C,)
  - beta: Shift parameter, of shape (C,)
  - bn_param: Dictionary with the following keys:
    - mode: 'train' or 'test'; required
    - eps: Constant for numeric stability
    - momentum: Constant for running mean / variance. momentum=0 means that
      old information is discarded completely at every time step, while
      momentum=1 means that new information is never incorporated. The
      default of momentum=0.9 should work well in most situations.
    - running_mean: Array of shape (D,) giving running mean of features
    - running_var Array of shape (D,) giving running variance of features

  Returns a tuple of:
  - out: Output data, of shape (N, C, H, W)
  - cache: Values needed for the backward pass
  """
  out, cache = None, None

  #############################################################################
  # TODO: Implement the forward pass for spatial batch normalization.         #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################

  N, C, H, W = x.shape
  flattened = x.transpose(1, 0, 2, 3).reshape( C, N*H*W )
  out, cache = batchnorm_forward( flattened.T, gamma, beta, bn_param )
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  out = out.T.reshape( C, N, H, W ).swapaxes(0,1)
  return out, cache

def spatial_batchnorm_backward(dout, cache):
  """
  Computes the backward pass for spatial batch normalization.

  Inputs:
  - dout: Upstream derivatives, of shape (N, C, H, W)
  - cache: Values from the forward pass

  Returns a tuple of:
  - dx: Gradient with respect to inputs, of shape (N, C, H, W)
  - dgamma: Gradient with respect to scale parameter, of shape (C,)
  - dbeta: Gradient with respect to shift parameter, of shape (C,)
  """
  dx, dgamma, dbeta = None, None, None

  #############################################################################
  # TODO: Implement the backward pass for spatial batch normalization.        #
  #                                                                           #
  # HINT: You can implement spatial batch normalization using the vanilla     #
  # version of batch normalization defined above. Your implementation should  #
  # be very short; ours is less than five lines.                              #
  #############################################################################
  N, C, H, W = dout.shape
  flattened = dout.transpose(1, 0, 2, 3).reshape(C, N*H*W)
  dx, dgamma, dbeta = batchnorm_backward( flattened.T, cache )

  dx = dx.T.reshape( C, N, H, W).swapaxes(0, 1)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return dx, dgamma, dbeta

def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx


def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
