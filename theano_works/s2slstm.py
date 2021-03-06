#coding:utf-8
""" s2slstm.py
-------------------------------

A Theano-based program for training and running sequence-to-sequence lstm-based neural networks.

Reference:
[1] Sutskever I, Vinyals O, Le Q V. Sequence to sequence learning with neural networks[C]//Advances in neural information processing systems. 2014: 3104-3112.
[2] http://deeplearning.net/tutorial/lstm.html
[3] Wang Q, Luo T, Wang D, et al. Chinese Song Iambics Generation with Neural Attention-based Model[J].
-------------------------------

Date: 2016-05-09
Author: Aodong Li
e-mail: liaodong@bupt.edu.cn
Location: CSLT@THU
"""
### Libraries
# Standard libraries 
import time
import cPickle
import os
import sys
from collections import OrderedDict

# Third-party libraries
import theano
import theano.tensor as T
import numpy as np

theano.config.floatX = 'float32'
"""
Important to know before preceding!

https://github.com/Theano/Theano/issues/2995
https://github.com/Theano/Theano/issues/3162
https://github.com/Lasagne/Lasagne/issues/332#issuecomment-122328992
"""
#theano.config.optimizer='fast_compile'
#theano.config.warn_float64='warn'
#theano.config.exception_verbosity='high'

# Set the random number generators' seeds for consistency
SEED = 123
np.random.seed(SEED)

options = {}
options['word_emb_size'] = 50
options['word_count_threshold'] = 5
options['hidden_layer_size'] = 100
options['softmax_layer_size'] = 1 # This will be changed according to voc size in getVoc
options['learning_rate'] = 0.3
options['batch_size'] = 700
options['max_epochs'] = 1000
options['fluctuate_times'] = 4
options['save_to'] = 'lstm.npz'
# word vector size = 50
# word count threshold = 5

def is_unwanted_words(word, w_list): # Used in getVoc
    return word in w_list

def getVoc(raw_data_file='dljv4', vector_file='dljvec', forced_new=True):
    """
    Get word2id, id2word, word vectors from data files.
    """
    if os.path.exists("word_misc.pkl") and not forced_new:
        return cPickle.load(open("word_misc.pkl", "rb"))
    else:
        # All necessary data will be loaded into word_misc
        word_misc, word2id, id2word, word_counts = {}, {}, {}, {}
        voc_list, voc_all, word_emb = [], [], []
        word_count = 1

        print("Building vocabulary...")
        # Character statistics
        for lines in open(raw_data_file).readlines():
            words = lines.split()
            for word in words:
                if not is_unwanted_words(word, ['|', '\n', '']):
                    word_counts[word] = word_counts.get(word, 0) + 1

        voc_list = [w for w in word_counts if word_counts[w] >= options['word_count_threshold']]
        voc_all = [w for w in word_counts]
        # Get vocabulary from statistics
        word2id['</s>'] = 0
        id2word[0] = '</s>'
        for w in voc_list:
            word2id[w] = word_count
            id2word[word_count] = w
            word_count += 1;
        word2id['UNK'] = word_count # Add unknown character
        id2word[word_count] = 'UNK'

        #options['softmax_layer_size'] = word_count
        print "Vocabulary size:", word_count

        # Initialization is refered to in https://www.tensorflow.org/versions/r0.7/tutorials/word2vec/index.html
        word_emb = (1/np.sqrt(word_count)*(2*np.random.rand(word_count, options['word_emb_size']) - 1)).tolist()

        # load word vectors
        for lines in open(vector_file).readlines()[1:]:
            word = lines.split()[0]
            #if word == '</s>' or word not in word2id.keys():
            #    continue
            if word not in word2id.keys():
                continue
            ids = word2id[word]
            #print ids, lines, len(word_emb)
            word_emb[ids] = [float(w) for w in lines.split()[1:]]

        print len(word_emb), "words have been loaded with", len(word_emb[0]), "dimensions"

        word_misc['word2id'] = word2id
        word_misc['id2word'] = id2word
        word_misc['word_emb'] = word_emb
        word_misc['voc_list'] = voc_list
        word_misc['voc_all'] = voc_all
        word_misc['word_count'] = word_count
        cPickle.dump(word_misc, open("word_misc.pkl", "wb"))
        return word_misc

def init_sftmax_layer_size(word_count):
    options['softmax_layer_size'] = word_count

def numpy_floatX(data):
    return np.asarray(data, dtype=theano.config.floatX)

def load_params(path, params):
    #Load arrays or pickled objects from .npy, .npz or pickled files.
    pp = np.load(path)
    for kk, vv in pp.iteritems():
        if kk not in params:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]
    return params

# params to tparams(shared)
def p_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

# convert tparams(键，共享变量) to params(键，数值)
def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

def init_params(is_init = True):
    # There MUST be no useless parameters, otherwise it will cause theano.gradient.DisconnectedInputError
    params = {}
    if is_init:
        params['W'] = 1/np.sqrt(options['word_emb_size'])*(2*np.random.rand(options['word_emb_size'], options['hidden_layer_size']*4)-1).astype(theano.config.floatX)
        params['U'] = 1/np.sqrt(options['hidden_layer_size'])*(2*np.random.rand(options['hidden_layer_size'], options['hidden_layer_size']*4)-1).astype(theano.config.floatX)
        params['B'] = 1/np.sqrt(4*options['hidden_layer_size'])*(2*np.random.rand(options['hidden_layer_size']*4)-1).astype(theano.config.floatX)
        #params['V'] = 1/np.sqrt(options['hidden_layer_size'])*(2*np.random.rand(options['hidden_layer_size'], options['hidden_layer_size'])-1).astype(theano.config.floatX)
        params['W_de'] = 1/np.sqrt(options['word_emb_size'])*(2*np.random.rand(options['word_emb_size'], options['hidden_layer_size']*4)-1).astype(theano.config.floatX)
        params['U_de'] = 1/np.sqrt(options['hidden_layer_size'])*(2*np.random.rand(options['hidden_layer_size'], options['hidden_layer_size']*4)-1).astype(theano.config.floatX)
        params['B_de'] = 1/np.sqrt(4*options['hidden_layer_size'])*(2*np.random.rand(options['hidden_layer_size']*4)-1).astype(theano.config.floatX)
        #params['V_de'] = 1/np.sqrt(options['hidden_layer_size'])*(2*np.random.rand(options['hidden_layer_size'], options['hidden_layer_size'])-1).astype(theano.config.floatX)
        params['W_sftmax'] = 1/np.sqrt(options['hidden_layer_size'])*(2*np.random.rand(options['hidden_layer_size'], options['softmax_layer_size'])-1).astype(theano.config.floatX)
        params['B_sftmax'] = 1/np.sqrt(options['softmax_layer_size'])*(2*np.random.rand(options['softmax_layer_size'])-1).astype(theano.config.floatX)
    else: # This allows prediction process to load trained parameters into the uninitialized parameters
        params['W'] = []
        params['U'] = []
        params['B'] = []
        #params['V'] = []
        params['W_de'] = []
        params['U_de'] = []
        params['B_de'] = []
        #params['V_de'] = []
        params['W_sftmax'] = []
        params['B_sftmax'] = []
    return params

def lstm_encoder(X, params):
    def _slice(_x, n, dim):
        # For n: 0-i, 1-f, 2-o, 3-c
        return _x[n * dim:(n+1) * dim]

    def _step(_x, _h, _c):
        # http://deeplearning.net/tutorial/lstm.html
        preact = T.dot(_h, params['U'])
        preact += _x
        preact += params['B']
        i = T.nnet.sigmoid(_slice(preact, 0, options['hidden_layer_size']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['hidden_layer_size']))
        o = T.nnet.sigmoid(_slice(preact, 2, options['hidden_layer_size']))
        c = T.tanh(_slice(preact, 3, options['hidden_layer_size']))
        c = i * c + f * _c
        #o = T.nnet.sigmoid(_slice(preact + T.dot(c, params['V']), 2, options['hidden_layer_size']))
        h = o * T.tanh(c)
        return h, c

    w_emb = T.dot(X, params['W'])
    h_c, updates = theano.scan(_step, # Initialize _h, _c to 0.0
                                sequences = [w_emb],
                                outputs_info = [T.alloc(numpy_floatX(0.), options['hidden_layer_size']),
                                                T.alloc(numpy_floatX(0.), options['hidden_layer_size'])])

    return h_c[0][-1] # h at last step

def lstm_decoder(y_prev, h, params, word_emb, k):
    def _slice(_x, n, dim):
        # For n: 0-i, 1-f, 2-o, 3-c
        return _x[n * dim:(n+1) * dim]

    def _step(_x, _h, _c, _word_emb):
        # http://deeplearning.net/tutorial/lstm.html
        preact = T.dot(_h, params['U_de'])
        preact += T.dot(_x, params['W_de'])
        preact += params['B_de']
        i = T.nnet.sigmoid(_slice(preact, 0, options['hidden_layer_size']))
        f = T.nnet.sigmoid(_slice(preact, 1, options['hidden_layer_size']))
        o = T.nnet.sigmoid(_slice(preact, 2, options['hidden_layer_size']))
        c = T.tanh(_slice(preact, 3, options['hidden_layer_size']))
        c = i * c + f * _c
        #o = T.nnet.sigmoid(_slice(preact + T.dot(c, params['V']), 2, options['hidden_layer_size']))
        h = o * T.tanh(c)
        # Output layer
        y_temp = T.nnet.softmax(T.dot(h, params['W_sftmax'])+params['B_sftmax']) # 3D tensor with rows (matrix) for each element
        # http://deeplearning.net/software/theano/library/tensor/basic.html
        # https://groups.google.com/forum/#!topic/theano-users/HF3-mJk3iX8
        y_temp = y_temp.reshape((y_temp.shape[1],)) # Convert row (matrix) to vector while keep the size unchanged
        y_i = T.argmax(y_temp)
        y_p = _word_emb[y_i]
        return y_p, h, c, y_i, y_temp

    y_h_c, updates = theano.scan(_step, # Initialize y_prev to </s>, _h to lstm_encoder's last h, _c to 0.0
                                outputs_info = [y_prev,
                                                h,
                                                T.alloc(numpy_floatX(0.), options['hidden_layer_size']), None, None],
                                non_sequences = word_emb,
                                n_steps = k)

    return y_h_c # y, h, c at each step

def build_model_for_training(params):
    """
    Training model initialization
    """
    # network input
    X = T.fmatrix('X')
    y_prev = T.fvector('y_prev')
    word_emb = T.fmatrix('word_emb')
    Y = T.fmatrix('Y')  # target, which should be 1hot coding form
    k = T.shape(Y)[0]
    
    h = lstm_encoder(X, params)
    y_h_c = lstm_decoder(y_prev, h, params, word_emb, k) # T.shape() returns array([len(row), len(column)])
    #theano.function([y_prev, h, word_emb, k], y_h_c)
    #print y_h_c[-1].type, y_h_c[-1][0].type
    cost = T.nnet.categorical_crossentropy(y_h_c[-1], Y)
    cost_sentence = T.mean(cost) 
    
    return X, y_prev, word_emb, Y, k, cost_sentence, y_h_c[-2] # output every predicted word

def sgd(lr, tparams, grads, X, y_prev, word_emb, ksteps, Y, cost, y_i):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    #print type(lr), type(X), type(y_prev), type(word_emb), type(Y), type(k), type(Y)
    # Function that computes gradients for a sentence, but do not
    # updates the weights.
    f_grad_shared = theano.function([X, y_prev, word_emb, ksteps, Y],
                                     [cost, y_i], 
                                     updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def get_training_data(raw_data_file = 'dljv4', mod = 2):
    # mod-1: training sentences and target sentences share the same sentences
    # mod-2: training sentences are former of target sentences
    # return training_in, target_out data sets
    # sentences in one training and target pair share one common index in training_in and target_out data sets
    training_in, target_out = [], []
    if mod == 1:
        for lines in open(raw_data_file).readlines():
            if len(lines) <= 1: # Delete blank lines
                continue
            ps_temp = lines.split('|')
            ps = [p for p in ps_temp]
            training_in += ps
            target_out += ps
    if mod == 2:
        for lines in open(raw_data_file).readlines():
            if len(lines) <= 1: # Delete blank lines
                continue
            ps_temp = lines.split('|')
            for ti, to in zip(ps_temp[:-1], ps_temp[1:]):
                training_in.append(ti)
                target_out.append(to)
    print len(training_in), "samples in all."
    return training_in, target_out

def word_2_vector(character, word2id, word_count, word_emb): # notion of character share the same meaning with notion of word
    return word_emb[word2id.get(character, word_count-1)]

def sentence_2_vector(sentence, word2id, word_count, word_emb): # notion of character share the same meaning with notion of word
    # Words of sentences are seperated by <space>, that is, ' '
    w_list = sentence.split()
    v_list = [word_2_vector(word, word2id, word_count, word_emb) for word in w_list if word != '']
    return v_list

def word_2_1hot(character, word2id, word_count): # 1-of-V encoding
    temp = np.zeros(word_count, dtype = 'int32')
    temp[word2id.get(character, word_count-1)] = 1
    return temp

def sentence_2_1hot(sentence, word2id, word_count):
    w_list = sentence.split()
    bag_list = [word_2_1hot(word, word2id, word_count) for word in w_list if word != '']
    return bag_list

def training_process(raw_data_file = 'dljv4'):
    """
    Each song is a sample.
    Once we use a song, that is, a single line in raw_data_file, as an input, we will use itself as an output or target.
    """
    print "Loading data..."
    training_in, target_out = get_training_data(raw_data_file, 2)

    print "Preparing word lists and parameters..."
    word_misc = getVoc(raw_data_file)
    word_count = word_misc['word_count']
    word2id = word_misc['word2id']
    id2word = word_misc['id2word']
    word_emb = word_misc['word_emb']
    init_sftmax_layer_size(word_count)
    EOS = numpy_floatX((word_emb[0])) # </s> for y_prev, end of sentence
    #print EOS, len(in_out_vector), len(in_out_vector[0]), np.shape(in_out_vector[0])
    params = init_params()
    # ????Why???? Just for shared?
    params = p_tparams(params)

    print "Building model..."
    X, y_prev, _word_emb, Y, k, cost_sentence, y_i = build_model_for_training(params)
    grads = T.grad(cost_sentence, wrt = params.values())
    lrate = T.fscalar('lrate')
    #print type(lrate), type(X), type(y_prev), type(word_emb), type(Y), type(k), type(Y)
    f_grad_shared, f_update = sgd(lrate, params, grads, X, y_prev, _word_emb, k, Y, cost_sentence, y_i)

    print "Training..."
    start_time = time.clock()
    lr = options['learning_rate']
    mean_of_cost_all = 0. # current cost
    best_cost = 0.
    best_p = None # last step parameters
    difference = 0.
    count = 0
    for epoch in range(options['max_epochs']):
        cost_all = []
        for iteration in range(options['batch_size']):
            idx = np.random.randint(0, len(training_in)-1)
            sample_in, sample_out = training_in[idx], target_out[idx]
            #sample = sample[::-1] # reverse training sample as paper says
            sample_in_vector = numpy_floatX(sentence_2_vector(sample_in, word2id, word_count, word_emb)) # input, that is, X
            sample_out_1hot = numpy_floatX(sentence_2_1hot(sample_out, word2id, word_count)) # Supervisation of softmax, that is, Y
            cost_sample, y_i = f_grad_shared(sample_in_vector, EOS, numpy_floatX(word_emb), len(sample_out_1hot), sample_out_1hot) # y_i is decoder's temporal outputs
            #f_grad_shared(X, y_prev, word_emb, ksteps, Y)
            f_update(lr)
            ##### For debug
            #print sample
            #print cost_sample
            #for i in y_i:
            #    print id2word[i]
            #####
            if np.isnan(cost_sample) or np.isinf(cost_sample):
                print "Bad cost detected on", epoch, "th epooch:", cost_sample
                return 1., 1., 1.
            cost_all.append(cost_sample)
        print epoch, "th epoch finished."

        ### Adjust the learning rate ###
        mean_of_cost_all = np.mean(cost_all)
        if epoch == 0:
            difference = mean_of_cost_all
        else:
            difference = best_cost - mean_of_cost_all

        if difference > 0:
            best_p = unzip(params) # convert shared variable to normal variable
            best_cost = mean_of_cost_all
            count = 0
            np.savez(options['save_to'], **best_p) # Convenient for debugging
        elif count >= options['fluctuate_times']: # Fluctuation happens, use lower learning rate to recalculate
            #params = p_tparams(best_p) # whether use last time's parameters 
            lr /= 1.2
            count = 0
            continue
        else:
            count += 1

        print mean_of_cost_all, ':', epoch, ':', lr
        
        '''if difference < 0.001 and difference > 0:
            print "May have reached the minimal."
            break'''
        
    # Save parameters
    print "Saving parameters..."
    np.savez(options['save_to'], **best_p)

    end_time = time.clock()
    print 'Training took %.1fs' % (end_time - start_time)

def build_model_for_prediction(params):
    """
    Prediction model initialization
    """
    # network input
    X = T.fmatrix('X')
    y_prev = T.fvector('y_prev')
    word_emb = T.fmatrix('word_emb')
    k = T.iscalar('k')

    h = lstm_encoder(X, params)
    y_h_c = lstm_decoder(y_prev, h, params, word_emb, k) # constant iteration number

    f_predict = theano.function([X, y_prev, word_emb, k], y_h_c[-2], name='predict_f', mode = 'FAST_RUN', allow_input_downcast=True)
    
    return X, y_prev, word_emb, k, f_predict

def prediction_process(test):
    print "Loading word misc..."
    word_misc = cPickle.load(open('word_misc.pkl', 'rb'))
    word_count = word_misc['word_count']
    word2id = word_misc['word2id']
    id2word = word_misc['id2word']
    word_emb = word_misc['word_emb']
    init_sftmax_layer_size(word_count)
    EOS = numpy_floatX((word_emb[0])) # </s> for y_prev, end of sentence
    # Load parameters
    print "Loading parameters..."
    param_file = options['save_to']
    params = init_params(False)
    params = load_params(param_file, params)
    params = p_tparams(params)

    print "Building model..."
    X, y_prev, _word_emb, k, f_predict = build_model_for_prediction(params)
    
    #test = "我时常默默地问自己"   
    test = unicode(test, 'utf-8')
    test = ' '.join(test)
    test = test.encode('utf-8')

    sample_vector = numpy_floatX(sentence_2_vector(test, word2id, word_count, word_emb))
    #print type(sample_vector), type(EOS), type(f_predict)
    y_i = f_predict(sample_vector, EOS, numpy_floatX(word_emb), 15)
    for i in y_i:
        print(id2word[i])

if __name__ == '__main__':
    '''word_misc = getVoc()
    init_sftmax_layer_size(word_misc['word_count'])
    params = init_params();
    result = theano.function()'''
    #training_process('dljv4')
    prediction_process('我时常默默地问自己')
