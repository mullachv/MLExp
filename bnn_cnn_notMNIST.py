# Downloads and runs classification on notMNIST_small dataset
# Courtesy: Y. Bulatov
# Source: http://yaroslavvb.com/upload/notMNIST/
#
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd
import seaborn as sns
sns.set_style('white')
sns.set_context('talk')

import pymc3 as pm
import theano.tensor as tt
import theano

from scipy.stats import mode, chisquare
import lasagne
import time
import tarfile

import sys, os
import PIL.Image as Image
import matplotlib.pyplot as plt
from six.moves import zip
from sklearn.metrics import accuracy_score
import argparse
from sklearn.model_selection import train_test_split


print("Starting: {}, {}".format(time.strftime("%x"), time.strftime("%X")))
plt.ioff()

def run_bnn_cnn(output):
    # (u'1.0.1', '0.2.dev1')
    print("theano: {}, lasagne: {}".format(theano.__version__, lasagne.__version__))

    data_dir = output

    def load_dataset():
        # support both python 2 and 3
        if sys.version_info[0] == 2:
            from urllib import urlretrieve
        else:
            from urllib.request import urlretrieve

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        def download(filename, source='http://yaroslavvb.com/upload/notMNIST/'):
            localfile = os.path.join(data_dir, filename)
            if not os.path.exists(localfile):
                print ("Downloading %s" % filename)
                urlretrieve(source + filename, localfile)

        # Extract tar file
        filename = 'notMNIST_small.tar.gz'
        download(filename)
        localfile = os.path.join(data_dir, filename)
        tar = tarfile.open(localfile)
        tar.extractall(data_dir)
        tar.close()

        def reshape_X(X_pre):
            X_post = np.zeros((X_pre.shape[0], 1, 28, 28))
            for i in range(X_pre.shape[0]):
                X_post[i, 0, :, :] = X_pre[i, :, :]
            return X_post

        # 'A' : 0, 'B' : 1 etc.
        mapping = {}

        def generate_dataset(folder):
            print folder
            max_count = 0
            for (root, dirs, files) in os.walk(folder):
                for f in files:
                    if f.endswith('.png'):
                        max_count += 1

            print 'Found %s files' % (max_count,)
            data = np.zeros((28, 28, max_count))
            labels = np.zeros((max_count,))
            count = 0
            for (root, dirs, files) in os.walk(folder):
                for f in files:
                    if f.endswith('.png'):
                        try:
                            # print("opening {}, {}, {}".format(root, f, os.path.join(root, f)))
                            img = Image.open(os.path.join(root, f));
                            # print("{}, {}".format(f, dirs))
                            data[:, :, count] = np.asarray(img)
                            surround_folder = os.path.split(root)[-1]
                            assert len(surround_folder) == 1
                            ordinal = ord(surround_folder) - ord('A')
                            labels[count] = ordinal
                            mapping[ordinal] = surround_folder
                            count += 1
                        except Exception as e:
                            print("exception {}".format(e))
                            pass

            # Plot NUM samples
            fig = plt.figure()
            COLS, NUM = 3, 8
            np.random.seed(NUM)
            for i in range(NUM):
                ax = fig.add_subplot(COLS, np.ceil(NUM / float(COLS)), i + 1)
                x = np.random.randint(max_count)
                a0, c0 = data[:, :, x], labels[x]
                plt.imshow(a0)
                plt.title(mapping[c0], fontsize=46)
            fig.set_size_inches(np.array(fig.get_size_inches()) * NUM)
            plt.savefig(os.path.join(data_dir, 'random_samples.png'), bbox_inches='tight')
            plt.close(fig)

            # print("data: {}, {}".format(data[:, :, :count].shape, labels[:count].shape))
            ndata = np.zeros((max_count, 28, 28))
            for i in range(max_count):
                ndata[i, :, :] = data[:, :, i]

            fig = plt.figure()
            x = np.random.randint(max_count)
            # print("ndata: {},{},{}".format(x, ndata[x].shape, ndata[x, :, :]))
            a0, c0 = ndata[x, :, :], labels[x]
            plt.imshow(a0)
            plt.title(mapping[c0], fontsize=46)
            plt.savefig(os.path.join(data_dir, 'tx-sample.png'), bbox_inches='tight')
            plt.close(fig)

            X_train, X_test, y_train, y_test = train_test_split(ndata[:count], labels[:count], test_size=0.1)
            X_train, X_test = reshape_X(X_train), reshape_X(X_test)
            return X_train, X_test, y_train, y_test

        return generate_dataset(os.path.join(data_dir, 'notMNIST_small'))

    X_train, X_test, y_train, y_test = load_dataset()

    print("Dataset loaded")

    # theano shared vars
    input_var = theano.shared(X_train[:500, ...].astype(np.float64))
    target_var = theano.shared(y_train[:500, ...].astype(np.float64))

    # ADVI minibatches
    #
    minibatch_tensors = [input_var, target_var]

    #create minibatches
    def create_minibatch(data, batchsize=500):
        rng = np.random.RandomState(0)
        start_idx = 0
        # Return random samples of batchsize on each iteration
        while True:
            ixs = rng.randint(data.shape[0], size=batchsize)
            yield data[ixs]

    minibatches = zip(
        create_minibatch(X_train, 500),
        create_minibatch(y_train, 500)
    )
    total_size = len(y_train)

    #advi sampler
    def run_advi(likelihood, advi_iters=50000):
        # Train
        input_var.set_value(X_train[:500, ...])
        target_var.set_value(y_train[:500, ...])
        v_params = pm.variational.advi_minibatch(
            n=advi_iters,
            minibatch_tensors=minibatch_tensors,
            minibatch_RVs=[likelihood],
            minibatches=minibatches,
            total_size=total_size,
            learning_rate=1e-2,
            epsilon=1.0
        )

        trace = pm.variational.sample_vp(v_params, draws=500)

        # Predict on test data
        input_var.set_value(X_test)
        target_var.set_value(y_test)

        ppc = pm.sample_ppc(trace, samples=100)
        y_pred = mode(ppc['out'], axis=0).mode[0, :]

        return v_params, trace, ppc, y_pred

    #Normal Priors on NN weights
    class GaussianWeights(object):
        def __init__(self):
            self.count = 0

        def __call__(self, shape):
            self.count += 1
            return pm.Normal('w%d' % self.count,
                             mu=0,
                             sd=.1,
                             testval=np.random.normal(size=shape).astype(np.float64),
                             shape=shape)

    def build_ann_conv(init):
        network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28), input_var=input_var)
        network = lasagne.layers.Conv2DLayer(network,
                                             num_filters=32,
                                             filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.tanh,
                                             W=init
                                             )
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.Conv2DLayer(network,
                                             num_filters=32,
                                             filter_size=(5, 5),
                                             nonlinearity=lasagne.nonlinearities.tanh,
                                             W=init)
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
        network = lasagne.layers.DenseLayer(network,
                                            num_units=256,
                                            nonlinearity=lasagne.nonlinearities.tanh,
                                            b=init,
                                            W=init
                                            )
        network = lasagne.layers.DenseLayer(network,
                                            num_units=10,
                                            nonlinearity=lasagne.nonlinearities.softmax,
                                            b=init,
                                            W=init)
        prediction = lasagne.layers.get_output(network)

        return pm.Categorical('out', prediction, observed=target_var)

    print("Training started: {}".format(time.strftime("%X")))

    #Now, stitch them together and call the neural net
    with pm.Model() as cnn:
        likelihood = build_ann_conv(GaussianWeights())
        v_params, trace, ppc, y_pred = run_advi(likelihood)

    print("Training completed: {}".format(time.strftime("%X")))

    print("Accuracy on test: {}".format(accuracy_score(y_test, y_pred)*100))

    miss_class = np.where(y_test != y_pred)[0]
    corr_class = np.where(y_test == y_pred)[0]
    preds = pd.DataFrame(ppc['out']).T
    chis = preds.apply(lambda x: chisquare(x).statistic, axis='columns')

    fig = plt.figure()
    sns.distplot(chis.loc[miss_class].dropna(), label='Error')
    sns.distplot(chis.loc[corr_class].dropna(), label='Correct')
    plt.legend()
    sns.despine()
    plt.xlabel('Chi-Square Statistic')
    plt.savefig(os.path.join(data_dir, 'Chi-sq stats.png'), bbox_inches='tight')
    plt.close(fig)

print("Stopping: {}, {}".format(time.strftime("%x"), time.strftime("%X")))

if __name__ ==  '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-op', '--output', type=str, help='Output Directory for Plots', default='data/notMNIST')

    args = parser.parse_args()
    run_bnn_cnn(args.output)
