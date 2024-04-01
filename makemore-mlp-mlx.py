import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from functools import partial
import matplotlib.pyplot as plt # for making figures
import time
import math

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
        nn.Linear(idim, odim, bias=True) for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def __call__(self, x):
        for l in self.layers[:-1]:
            x = nn.tanh(l(x))
        return self.layers[-1](x)

def to_samples(context_size, dataset):
    tokens = dataset.size
    window_size = context_size + 1  # include target
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]

def iterate_batches(batch_size, context_size, dataset):
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
            ids = perm[s : s + batch_size]
            yield inputs[ids], targets[ids]
            s += batch_size
        if s >= inputs.shape[0]:
            s = 0

# build the dataset
def build_dataset(dataset, context_size, stoi):
    X, Y = [], [] # inputs, labels
    for w in dataset:
        context = [0] * context_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append (rolling window of context)
    X = mx.array(X)
    Y = mx.array(Y)
    return X, Y

    

def main(dataset):
    batch_size = 32
    context_size = 3 # Context size in tokens of the model
    num_iters = 100_000 # Iterations to train for
    learning_rate = 0.1 # SGD learning rate
    # lr_warmup = 200 # LR linear warmup iterations
    # weight_decay = 1e-4 # Set the weight decay
    # steps_per_eval = 1000 # Number of training steps between validations
    # steps_per_report = 10 # Number of training steps between loss reporting
    optimizer = optim.SGD(
        learning_rate=learning_rate,
        # weight_decay=weight_decay
    )

    vocab = sorted(list(set(''.join(dataset))))
    stoi = {s:i+1 for i,s in enumerate(vocab)}
    stoi['.'] = 0
    itos = {i:s for s,i in stoi.items()}

    seed = 2147483647
    np.random.seed(seed)
    np.random.shuffle(dataset)
    n1 = int(0.8 * len(dataset))
    n2 = int(0.9 * len(dataset))
    X_train, Y_train = build_dataset(dataset[:n1], context_size, stoi)
    X_val, Y_val = build_dataset(dataset[n1:n2], context_size, stoi)
    X_test, Y_test = build_dataset(dataset[n2:], context_size, stoi)

    key = mx.random.key(seed)
    E = mx.random.normal(shape=(27, 10), key=key)
    input_dim = 3 * 10
    hidden_dim = 200
    output_dim = 27
    num_layers = 2
    model = MLP(input_dim, hidden_dim, output_dim, num_layers)
    mx.eval(model.parameters())

    def loss_fn(model, x, y):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses)
    
    def eval_fn(context_size, dataset):
        inputs, targets = map(mx.array, to_samples(context_size, dataset))
        loss = 0
        for s in range(0, targets.shape[0], 32):
            bx, by = inputs[s : s + 32], targets[s : s + 32]
            bx, by = map(mx.array, (bx, by))
            losses = loss_fn(model, inputs, targets, reduce=False)
            loss += mx.sum(losses).item()
        return loss / len(targets)
    
    state = [model.state, optimizer.state]
    
    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss
    
    iterations = []
    losses = []
    # tic = time.perf_counter()
    for it in range(num_iters):
        itx = mx.random.randint(0, X_train.shape[0], (batch_size,))

        inputs = E[X_train[itx]]
        inputs = inputs.reshape(inputs.shape[0], inputs.shape[1] * inputs.shape[2])
        targets = Y_train[itx]
        
        # optimizer.learning_rate = min(1, it / lr_warmup) * learning_rate
        optimizer.learning_rate = learning_rate if it < 100_000 else learning_rate / 10
        
        loss = step(inputs, targets)
        mx.eval(state)
        if it % 1000 == 0:
            print(f'iteration={it}, loss={loss.item()}')
        losses.append(loss.item())
        iterations.append(it)
        
    #     if (it + 1) % steps_per_report == 0:
    #         train_loss = np.mean(losses)
    #         toc = time.perf_counter()
    #         print(
    #         f"Iter {it + 1}: Train loss {train_loss:.3f}, "
    #         f"It/sec {steps_per_report / (toc - tic):.3f}"
    #         )
    #         losses = []
    #         tic = time.perf_counter()
    #     if (it + 1) % steps_per_eval == 0:
    #         val_loss = eval_fn(X_val)
    #         toc = time.perf_counter()
    #         print(
    #         f"Iter {it + 1}: "
    #         f"Val loss {val_loss:.3f}, "
    #         f"Val ppl {math.exp(val_loss):.3f}, "
    #         f"Val took {(toc - tic):.3f}s, "
    #         )
    #         tic = time.perf_counter()

    # test_loss = eval_fn(X_test)
    # test_ppl = math.exp(test_loss)
    # print(f"Test loss {test_loss:.3f}, Test ppl {test_ppl:.3f}.")
    print(f'Final loss: {losses[-1]}')
    plt.plot(iterations, losses)
    plt.show()

if __name__ == "__main__":
    mx.set_default_device(mx.gpu)
    words = open('names.txt', 'r').read().splitlines()
    main(words)