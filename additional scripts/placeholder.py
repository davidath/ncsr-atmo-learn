# [_X,labels] = utils.load_mnist(dataset="training",path='/mnt/disk1/thanasis/ncsr-atmo-learn/examples/sda/')
# _X = _X[0:10,:]
_X = np.random.uniform(size=(10,784))
check = _X
_X = theano.shared(name='_X', value=np.asarray(_X,
                              dtype=theano.config.floatX),
                              borrow=True)
# x = T.matrix('x')
# x = theano.shared(name='x')
W = nnmodel._encoder_layer.W
w_eval = W.eval()
b = nnmodel._encoder_layer.b
out = T.dot(W,_X)
Y = nnmodel._encoder_layer.nonlinearity(out)
# grad = T.grad(Y,x)
# grad /= (T.sqrt(T.mean(T.square(grad))) + 1e-5)
# train = theano.function(inputs=[x],outputs=grad)
cost = -1 * Y.mean()
# grad = T.grad(prediction.mean(), _X)
params = [_X]
learning_rate = T.scalar(name='learning_rate')
updates = lasagne.updates.nesterov_momentum(
    cost, params, learning_rate=learning_rate)
train = theano.function(
    inputs=[learning_rate], outputs=cost, updates=updates)
base_lr = 0.1
# random = np.random.uniform(size=(10,784))
# check = random[0,:]
for epoch in xrange(100):
    epoch_loss = train(base_lr)
    x = T.matrix('x')
    newOut = T.dot(W,x)
    newY = nnmodel._encoder_layer.nonlinearity(newOut).mean()
    grad = T.grad(newY,x)
    grad /= (T.sqrt(T.mean(T.square(grad))) + 1e-5)
    getgrad = theano.function(inputs=[x],outputs=grad)
    _X += getgrad(_X.eval()) * epoch
    # print grad
    # _X += grad
    # print epoch_loss.shape
    # random += epoch_loss
    # Print loss every 10 epochs
    if epoch % 10 == 0:
        log(str(epoch) + ' ' + str(epoch_loss), label='DNT')
print np.array_equal(_X.eval(),check)
# print _X.eval(.shape
# print np.array_equal(check,random[0,:])
# random = random.reshape(10,28,28)
_X = _X.eval().reshape(10,28,28)
displayz(_X,2,5,0,28,28)
