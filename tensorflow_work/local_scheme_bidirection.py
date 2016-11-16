#coding:utf-8
""" lad_seq2seq.py
-------------------------------

A tensorflow-based program for training and running sequence-to-sequence gru and attention-based neural networks.

Reference:
[1] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate[J]. arXiv preprint arXiv:1409.0473, 2014.
[2] Shang L, Lu Z, Li H. Neural responding machine for short-text conversation[J]. arXiv preprint arXiv:1503.02364, 2015.
[3] Cho K, Van Merriënboer B, Gulcehre C, et al. Learning phrase representations using RNN encoder-decoder for statistical machine translation[J]. arXiv preprint arXiv:1406.1078, 2014.
-------------------------------

Date: 2016-06-26
Author: Aodong Li
e-mail: liaodong@bupt.edu.cn
Location: CSLT@THU
"""

# Use trained and shared word embedding for both encoder and decoder
# The encoder is a bidirection encoder

### Libraries
# Standard libraries 
import time
import os
import sys
import cPickle

# Thrid-party libraries
import tensorflow as tf
import numpy as np

# Global settings
SEED = 1
tf.set_random_seed(SEED)

RAW_DATA_PATH = 'data/'
options = {}
options['post_data_file'] = RAW_DATA_PATH + 'train_words.en'
options['response_data_file'] = RAW_DATA_PATH + 'train_words.fr'
options['vocab_file'] = RAW_DATA_PATH + 'vocab'
#options['vector_file'] = RAW_DATA_PATH + 'wordvec'
options['vocab_size'] = 0
options['embedding_size'] = 500 #620
options['state_size'] = 500 # 1000
options['batch_size'] = 64 # 512
options['iteration'] = 10000001
options['learning_rate'] = 0.005 # 0.0025
options['lr_decay'] = 0.983

def is_unwanted_words(word, w_list): # Used in getVoc
    return word in w_list

def get_weibo_data(vocab_file, vector_file):
    """
    Get word dictionary and its corresponding word embedding.
    """
    if os.path.exists("word_misc.pkl"):
        return cPickle.load(open("word_misc.pkl", "rb"))

    word_misc, word2id, id2word = {}, {}, {}
    word_count = 0

    # vocab file
    print "Building vocabulary ..."
    for lines in open(vocab_file).readlines():
        word = lines.split()[0]
        if not is_unwanted_words(word, ['', '\n']):
            word2id[word] = word_count
            id2word[word_count] = word
            word_count += 1
    word2id['_START'] = word_count
    id2word[word_count] = '_START'
    word_count += 1
    word2id['_END'] = word_count
    id2word[word_count] = '_END'
    word_count += 1
    word2id['_UNK'] = word_count
    id2word[word_count] = '_UNK'
    word_count += 1
    word2id['_MASK'] = word_count
    id2word[word_count] = '_MASK'
    word_count += 1
    print "Vocabulary size:", word_count

    # Initialization is refered to in https://www.tensorflow.org/versions/r0.7/tutorials/word2vec/index.html
    word_emb = (1/np.sqrt(word_count)*(2*np.random.rand(word_count, options['embedding_size']) - 1)).tolist()

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

    # load word misc
    word_misc['id2word'] = id2word
    word_misc['word2id'] = word2id
    word_misc['word_count'] = word_count
    word_misc['word_emb'] = word_emb
    cPickle.dump(word_misc, open("word_misc.pkl", "wb"))
    print "Dump complete."
    return word_misc

class GRUCell(object):
    """Gated Recurrent Unit cell, batch incorporated.

    Based on [3].
    GRU's output is equal to state.
    The arg wt_attention is associated with arg context.
    """
    def __init__(self, state_size, input_size, wt_attention=False, activation=tf.tanh):
        # Initialize parameters
        self._state_size = state_size
        self._activation = activation
        self._input_size = input_size
        self._wt_attention = wt_attention
        self._W_z_r = tf.Variable(tf.random_uniform([self._input_size, self._state_size * 2], -0.1, 0.1, tf.float32))
        self._U_z_r = tf.Variable(tf.random_uniform([self._state_size, self._state_size * 2], -0.1, 0.1, tf.float32))
        self._B_z_r = tf.Variable(tf.ones([self._state_size * 2], tf.float32)) # this is broadcastable
        self._W_h = tf.Variable(tf.random_uniform([self._input_size, self._state_size], -0.1, 0.1, tf.float32))
        self._U_h = tf.Variable(tf.random_uniform([self._state_size, self._state_size], -0.1, 0.1, tf.float32))
        self._B_h = tf.Variable(tf.ones([self._state_size], tf.float32)) # this is broadcastable
        if self._wt_attention:
            self._C_z_r = tf.Variable(tf.random_uniform([self._state_size, self._state_size * 2], -0.1, 0.1, tf.float32))
            self._C_h = tf.Variable(tf.random_uniform([self._state_size, self._state_size], -0.1, 0.1, tf.float32))

    def __call__(self, inputs, state, context=None):
        """Get input and former state, return output and state.

        input's shape should be (batch_size * input_size)
        state's shape should be (batch_size * state_size)

        The computation is according to formula.
        """
        if not self._wt_attention:
            r, z = tf.split(1, 2, tf.matmul(tf.concat(1, [inputs, state]), tf.concat(0, [self._W_z_r, self._U_z_r])) + self._B_z_r)
            ###### Debug
            #print tf.concat(1, [inputs, state]).get_shape(), tf.concat(0, [self._W_z_r, self._U_z_r]).get_shape()
        else:
            if context is None:
                raise ValueError("Attention mechanism used, while context vector is not received.")
            r, z = tf.split(1, 2, tf.matmul(tf.concat(1, [inputs, state, context]), tf.concat(0, [self._W_z_r, self._U_z_r, self._C_z_r])) + self._B_z_r)
        r, z = tf.sigmoid(r), tf.sigmoid(z)
        if not self._wt_attention:
            ###### Debug
            #print r.get_shape(), z.get_shape(), state.get_shape()
            c = self._activation(tf.matmul(tf.concat(1, [inputs, r * state]), tf.concat(0, [self._W_h, self._U_h])) + self._B_h)
        else:
            c = self._activation(tf.matmul(tf.concat(1, [inputs, r * state, context]), tf.concat(0, [self._W_h, self._U_h, self._C_h])) + self._B_h)
        h = z * state + (1 - z) * c
        return h, h

class Alignment(object):
    """Alignment model assigns weight to each encoder state.

    Based on [1] (3.3.2)
    """
    def __init__(self, state_size, activation=tf.tanh):
        self._state_size = state_size
        self._activation = activation
        self._W_a = tf.Variable(tf.random_uniform([self._state_size, self._state_size], -0.1, 0.1, tf.float32))
        # self._U_a is modified for bidirectional rnn
        self._U_a = tf.Variable(tf.random_uniform([self._state_size * 2, self._state_size], -0.1, 0.1, tf.float32))
        self._V_a = tf.Variable(tf.random_uniform([self._state_size, 1], -0.1, 0.1, tf.float32))

    def shortcut(self, states):
        """states is a list of tensor encoder's outputs"""
        l = len(states)
        t_ = tf.matmul(tf.concat(0, states), self._U_a)
        return tf.split(0, l, t_), l

    def __call__(self, state_s, h_U):
        # h_U is already the product of h and U
        """h_U: [batch_size * state_size]
        state_s: [batch_size * state_size]
        """
        a = self._activation(tf.matmul(state_s, self._W_a) + h_U)
        a = tf.matmul(a, self._V_a)
        return a # [batch_size * 1]

class Softmax_layer(object):
    """Softmax layer on the decoder"""
    def __init__(self, in_state_size, class_num, activation=tf.nn.softmax, loss_f=tf.nn.sparse_softmax_cross_entropy_with_logits):
        self._in_state_size = in_state_size
        self._class_num = class_num
        self._activation = activation
        self._loss_f = loss_f # Loss function
        self._W_sftm = tf.Variable(tf.random_uniform([self._in_state_size, self._class_num], -0.1, 0.1, tf.float32))
        self._B_sftm = tf.Variable(tf.ones([self._class_num], tf.float32))

    def __call__(self, in_state, labels, label_mask, predict=False):
        """Return (1) an int indicating index of symbol [batch_size], (2) possibility matrix [batch_size * class_num], (3) cross entropy loss [batch_size]

        in_state: [batch_size * in_state_size]
        labels: [batch_size] and the dtype int64 in [0, class_num)

        If predict is true, there is no loss returned
        """
        t_ = tf.matmul(in_state, self._W_sftm) + self._B_sftm # t_: [batch_size * class_num]
        #t_ = tf.expand_dims(label_mask, 1) * t_
        t_sftm_ = self._activation(t_)
        if not predict:
            #labels_1hot = tf.one_hot(labels, self._class_num, 1.0, 0.0)
            loss = self._loss_f(t_, labels)
            loss = loss * label_mask
            return tf.argmax(t_sftm_, 1), t_sftm_, loss
        else:
            return tf.argmax(t_sftm_, 1), t_sftm_

def create_embedding(num_symbol, embedding_size, embedding_name):
    """Generate embedding for posts, or response."""
    return tf.Variable(tf.random_uniform([num_symbol, embedding_size], -0.1, 0.1, tf.float32), name=embedding_name, trainable=True)

def generate_final_state(tensor, mask):
    """
    tensor is usually the output matrix of encoder, and we want the last state to be put into encoder, we use this function to generate state of each sequence.
    
    Args:
    tensor: [max_seq_len * batch_size * state_size]
    mask: [batch_size * max_seq_len], in which there is only one "True" signal indicating the end character

    Return:
    [batch_size * state_size]
    """
    tt = tf.transpose(tensor, [1, 0, 2])
    return tf.boolean_mask(tt, mask)

def bidirectional_gru_encoder(cell_fw, cell_bw, embedding, init_state, batch_input_fw, batch_input_bw, batch_mask):
    """
    A bidirectional implementation.
    return a list of [time][batch][cell_fw.output_size + cell_bw.output_size]
    """
    # Forward direction
    outputs_fw, _ = gru_encoder(cell_fw, embedding, init_state, batch_input_fw, batch_mask)

    # Backward direction
    outputs_bw, _ = gru_encoder(cell_bw, embedding, init_state, batch_input_bw, batch_mask)

    outputs = [tf.concat(1, [fw, bw]) for fw, bw in zip(outputs_fw, outputs_bw)]
    return outputs

def gru_encoder(cell, embedding, init_state, batch_input, batch_mask):
    """Return all the cell outputs, namely hj, given inputs.
    batch_input: [batch_size * input_len], numpy array
    """
    #batch_size = batch_input.get_shape()[0]
    #state = tf.zeros([batch_size, options['state_size']], tf.float32) # initialize the state
    outputs = []
    #split_inputs = tf.split(1, batch_input.get_shape()[0], batch_input)
    
    with tf.device("/cpu:0"):
        embedded_list = tf.nn.embedding_lookup(embedding, batch_input)
        #embedded_list = batch_mask * tf.transpose(embedded_list, [2, 0, 1]) # Add mask to change embedding into zeros
        #embedded_list = tf.transpose(embedded_list, [2, 1, 0])
        embedded_list = tf.transpose(embedded_list, [1, 0, 2])
        embedded_list = tf.unpack(embedded_list) # list of embedding
    
    # min_sequence_length = tf.reduce_min(seq_len)
    #max_sequence_length = tf.reduce_max(seq_len)

    state = init_state
    for time, (embedded, i_mask) in enumerate(zip(embedded_list, tf.unpack(tf.transpose(batch_mask)))):
        #embedded = tf.nn.embedding_lookup(embedding, tf.reshape(inputs, [-1])) # deprecated
        #embedded = embedded * tf.reshape(tf.convert_to_tensor(batch_mask[:, time], tf.float32), [batch_size, 1]) # deprecated
        #copy_cond = (time >= seq_len)
        #new_output, new_state = cell(embedded, state)
        output, state = cell(embedded, state)#tf.select(copy_cond, zero_output, new_output), tf.select(copy_cond, state, new_state)
        output = tf.expand_dims(i_mask, 1) * output
        outputs.append(output)
    #outputs = batch_mask * tf.transpose(tf.pack(outputs), [2, 0, 1])
    #outputs = tf.unpack(tf.transpose(outputs, [2, 1, 0]))
    return outputs, state

def softmax_wt_mask(value, mask):
    """value is a tensor of [batch_size * class_num] and mask share the shape with value"""
    numerator = tf.exp(value) * mask
    sum_ = tf.reduce_sum(numerator, 1, keep_dims=True)
    return numerator / sum_ # broadcastable

def gru_training_decoder(cell, attention, sftm, embedding, states, in_mask, in_bool_mask, batch_output, out_mask):
    # batch wise
    """Return all decoder's output by id and loss.
    cell: decoder cell
    states: list of encoder's output, namely h
    batch_output: [batch_size * output_len], numpy array
    """
    l_state = generate_final_state(tf.pack(states), in_bool_mask) # initialize the state, the encoder's last state
    outputs, loss, possib, symbol = [], [], [], []
    sstates, _ = attention.shortcut(states)

    with tf.device("/cpu:0"):
        embedded_list = tf.nn.embedding_lookup(embedding, batch_output)
        #embedded_list = out_mask * tf.transpose(embedded_list, [2, 0, 1]) # Add mask to change embedding into zeros
        #embedded_list = tf.transpose(embedded_list, [2, 1, 0])
        embedded_list = tf.transpose(embedded_list, [1, 0, 2])
        embedded_list = tf.unpack(embedded_list) # list of embedding

    for time, (embedded, target, t_mask) in enumerate(zip(embedded_list[:-1], tf.unpack(tf.transpose(batch_output))[1:], tf.unpack(tf.transpose(out_mask))[1:])):
        eij = []
        #embedded = tf.nn.embedding_lookup(embedding, tf.reshape(i, [-1])) # deprecated
        #embedded = embedded * tf.reshape(tf.convert_to_tensor(out_mask[:, time], tf.float32), [batch_size, 1]) # deprecated
        for h in sstates:
            eij.append(attention(l_state, h))
        eij = tf.concat(1, eij)
        alphaij = softmax_wt_mask(eij, in_mask)  # Add mask to change embedding into zeros
        #alphaij = tf.nn.softmax(eij) # PROBLEM!!!!
        #alphaij = alphaij * in_mask
        ##### Debug
        #print sess.run(alphaij)    #print in_mask    #print sess.run(alphaij)    #print states
        t_ = alphaij * tf.transpose(tf.pack(states)) # broadcastable
        t_ = tf.transpose(t_)
        ci = tf.reduce_sum(t_, 0)
        output, l_state = cell(embedded, l_state, ci)
        output = output * tf.expand_dims(t_mask, 1) # Add mask
        outputs.append(output)
        res = sftm(output, tf.cast(target, tf.int64), t_mask)
        symbol.append(res[0])
        possib.append(res[1])
        loss.append(res[2])
    #cost = tf.reduce_mean(tf.add_n(loss))
    total_size = tf.add_n(tf.unpack(tf.transpose(out_mask))[1:])
    total_size += 1e-12
    cost = tf.add_n(loss) / total_size
    cost = tf.reduce_mean(cost)
    return outputs, symbol, possib, cost, loss

def get_training_data(post_file, response_file, word2id, word_count):
    training_in, target_out = [], []
    in_mask, out_mask = [], []
    max_seq_len = 0
    for ti, to in zip(open(post_file).readlines(), open(response_file).readlines()):
        to = '_START ' + to + ' _END' # Two signals for decoder, to[:-1] for input, to[1:] for target
        training_in.append(sentence_2_id(ti, word2id, word_count))
        in_mask.append([1] * len(training_in[-1]))
        target_out.append(sentence_2_id(to, word2id, word_count))
        out_mask.append([1] * len(target_out[-1]))
        t_ = max(len(training_in[-1]), len(target_out[-1]))
        if max_seq_len < t_:
            max_seq_len = t_

    num_sample = len(training_in)
    print num_sample, "samples in all."
    return num_sample, training_in, target_out, in_mask, out_mask, max_seq_len

def sentence_2_id(sentence, word2id, word_count):
    # Words of sentences are seperated by <space>, that is, ' '
    w_list = sentence.split()
    id_list = [word2id.get(word, word_count - 2) for word in w_list if word != ''] # word_count-1 for 'MASK'
    return id_list

def id_2_sentence(ids, id2word):
    word_list = [id2word[i] for i in ids]
    return word_list

def get_batch_sample(range_sample, batch_size): # deprecated
    # range_sample is the range of idx
    batch_sample = np.random.randint(0, range_sample, size=batch_size, dtype="int64")
    return batch_sample

def init_post_vocab_size(post_vocab_size):
    # Set post_vocab_size
    options['post_vocab_size'] = post_vocab_size

def init_response_vocab_size(response_vocab_size):
    # Set response_vocab_size
    options['response_vocab_size'] = response_vocab_size

def init_vocab_size(vocab_size):
    # Set post_vocab_size
    options['vocab_size'] = vocab_size

def initialization():
    # Initialize all variables, namely parameters
    encoder_fw = GRUCell(options['state_size'], options['embedding_size'])
    encoder_bw = GRUCell(options['state_size'], options['embedding_size'])
    attention = Alignment(options['state_size'])
    sftm = Softmax_layer(options['state_size'], options['vocab_size'])
    decoder = GRUCell(options['state_size'], options['embedding_size'], wt_attention=True)
    #post_embedding = create_embedding(options['post_vocab_size'], options['embedding_size'], 'post_embedding')
    #response_embedding = create_embedding(options['response_vocab_size'], options['embedding_size'], 'response_embedding')
    #learning_rate = tf.Variable(1.0, trainable=False)
    return encoder_fw, encoder_bw, attention, sftm, decoder#, post_embedding, response_embedding, learning_rate

#def training_process(encoder, decoder, attention, sftm, post_embedding, response_embedding):
def training_process():
    print "Preparing word lists and parameters..."
    word_misc = get_weibo_data(options["vocab_file"], options["vector_file"])
    id2word = word_misc['id2word']
    word2id = word_misc['word2id']
    word_count = word_misc['word_count']
    word_emb = word_misc['word_emb']

    with tf.Graph().as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            #init_post_vocab_size(post_word_count)
            #init_response_vocab_size(response_word_count)
            init_vocab_size(word_count)

            with tf.device("/cpu:0"): # Does this generate embedding in CPU's memory, not in GPU's memory?
                encoder_fw, encoder_bw, attention, sftm, decoder = initialization()
                embedding = tf.Variable(word_emb, trainable=False)

            print "Loading training and prediction data..."
            num_sample, training_in, target_out, training_in_mask, target_out_mask, max_seq_len = get_training_data(options["post_data_file"], options["response_data_file"], word2id, word_count)
            #print max_seq_len, training_in_mask[-1]

            print "Loading model..."
            batch_input_fw = tf.placeholder(tf.int64, shape=(None, max_seq_len))
            batch_input_bw = tf.placeholder(tf.int64, shape=(None, max_seq_len))
            batch_output = tf.placeholder(tf.int64, shape=(None, max_seq_len))
            in_mask = tf.placeholder(tf.float32, shape=(None, max_seq_len))
            out_mask = tf.placeholder(tf.float32, shape=(None, max_seq_len))
            in_bool_mask = tf.placeholder(tf.bool, shape=(None, max_seq_len))
            zero_state = tf.placeholder(tf.float32, shape=(None, options['state_size']))

            #outputs, _ = gru_encoder(encoder, embedding, zero_state, batch_input, in_mask)
            outputs = bidirectional_gru_encoder(encoder_fw, encoder_bw, embedding, zero_state, batch_input_fw, batch_input_bw, in_mask)
            _, _, _, cost, loss = gru_training_decoder(decoder, attention, sftm, embedding, outputs, in_mask, in_bool_mask, batch_output, out_mask)
            #optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
            optimizer = tf.train.AdamOptimizer(learning_rate=options['learning_rate'])#learning_rate)
            train_op = optimizer.minimize(cost)
            '''
            grads_and_vars = optimizer.compute_gradients(cost)
            masked_grads_and_vars = [] # Mask the _START, _END, UNK, MASK
            for g, v in grads_and_vars:
                if isinstance(g, tf.IndexedSlices) and v.get_shape()[0] == response_word_count:
                    masked_grads_and_vars.append((tf.sparse_mask(g, [0, 1, response_word_count-2, response_word_count-1]), v))
                elif isinstance(g, tf.IndexedSlices) and v.get_shape()[0] == post_word_count:
                    masked_grads_and_vars.append((tf.sparse_mask(g, [post_word_count-2, post_word_count-1]), v))
                else:
                    masked_grads_and_vars.append((g, v))
            train_op = optimizer.apply_gradients(masked_grads_and_vars)
            '''
            init = tf.initialize_all_variables()
            saver = tf.train.Saver(max_to_keep=100)
            sess.run(init)
            #saver.restore(sess, "tmp/4.67672477722model.ckpt")
            #options['learning_rate'] = sess.run(learning_rate)
            #print "Model restored with learning rate as", options['learning_rate']
            ####### DEBUG
            #for g, v in grads_and_vars:
            #    if isinstance(g, tf.IndexedSlices):
            #        print [g, v], g.indices, g.values
            #    else:
            #        print [g,v], g.get_shape(), v.get_shape()

            print "Training..."
            start_time = time.clock()

            #### Debug
            #for tv in tf.trainable_variables():
            #    print tv.name
            total_cost = 0.0
            best_cost = 0.0
            #sess.run(tf.assign(learning_rate, options['learning_rate']))
            for _i in range(options['iteration']): # training process starts
                #batch_sample = get_batch_sample(num_sample, options['batch_size'])
                _idx = _i*options['batch_size']%num_sample
                #_idx = np.random.randint(num_sample+1) % num_sample
                batch_in = training_in[_idx:_idx+options['batch_size']]
                batch_out = target_out[_idx:_idx+options['batch_size']]
                batch_in_mask = training_in_mask[_idx:_idx+options['batch_size']]
                batch_out_mask = target_out_mask[_idx:_idx+options['batch_size']]
                batch_size = len(batch_in) # Maybe change at the last few samples

                # List of lists into numpy array with equal lengh
                batch_in_fw = np.array([l + [word_count-1]*(max_seq_len-len(l)) for l in batch_in], "int64")
                batch_in_bw = np.array([l[::-1] + [word_count-1]*(max_seq_len-len(l)) for l in batch_in], "int64")
                batch_in_mask = np.array([l + [0]*(max_seq_len-len(l)) for l in batch_in_mask], "float32")
                batch_out = np.array([l + [word_count-1]*(max_seq_len-len(l)) for l in batch_out], "int64")
                batch_out_mask = np.array([l + [0]*(max_seq_len-len(l)) for l in batch_out_mask], "float32")
                batch_in_bool_mask = np.zeros((batch_size, max_seq_len), "bool")
                for _t, line in enumerate(batch_in_mask):
                    batch_in_bool_mask[_t][int(np.sum(line)-1)] = True
                init_s = np.zeros((batch_size, options['state_size']), "float32")
                #mask_indices = [0, 1, 2, word_count-1, word_count-2]

                feed_dict = {
                            batch_input_fw: batch_in_fw,
                            batch_input_bw: batch_in_bw,
                            batch_output: batch_out,
                            in_mask: batch_in_mask,
                            out_mask: batch_out_mask,
                            in_bool_mask: batch_in_bool_mask,
                            zero_state: init_s,
                            }

                _, t_cost = sess.run([train_op, cost], feed_dict)
                total_cost += t_cost

                if _i % 100 == 0 and _i is not 0:
                    ###### Debug
                    #print sess.run(post_embedding)
                    #print sess.run(response_embedding)
                    pause_time = time.clock()
                    print "Training took %.1fs until %d iterations, reaching cost %f." % (pause_time - start_time, _i, total_cost/100)
                    # learning rate decay
                    ''' 
                    if _i % 500 == 0:
                        options['learning_rate'] = options['learning_rate'] * options['lr_decay']
                        sess.run(tf.assign(learning_rate, max(options['learning_rate'], 1e-5)))
                    print "Learning rate has changed to", max(options['learning_rate'], 1e-5) 
                    '''
                    '''
                    if best_cost == 0.0:
                        best_cost = total_cost/100
                    elif total_cost/100 < best_cost:
                        best_cost = total_cost/100
                    else:
                        options['learning_rate'] = options['learning_rate'] * options['lr_decay']
                        sess.run(tf.assign(learning_rate, max(options['learning_rate'], 1e-5)))
                    print "Learning rate has changed." 
                    '''
                    if _i % 1000 == 0: # Save checkpoint file
                        save_path = saver.save(sess, "tmp/"+str(total_cost/100)+"model.ckpt")
                        print("Model saved in file: %s" % save_path)
                    total_cost = 0.0
                    sys.stdout.flush() # flush the stdout

def predict_process_4_single_input(checkpoint_file): # single predict
    print "Preparing word lists and parameters..."
    word_misc = get_weibo_data(options["vocab_file"], options["vector_file"])
    id2word = word_misc['id2word']
    word2id = word_misc['word2id']
    word_count = word_misc['word_count']
    word_emb = word_misc['word_emb']

    def predict_bidirection_encoder(cell_fw, cell_bw, embedding, zero_state, inputs_fw, inputs_bw):
        outputs_fw, l_state_fw = predict_encoder(cell_fw, embedding, zero_state, inputs_fw)
        outputs_bw, l_state_bw = predict_encoder(cell_bw, embedding, zero_state, inputs_bw)
        outputs = [tf.concat(1, [fw, bw]) for fw, bw in zip(outputs_fw, outputs_bw)]
        l_state = tf.concat(1, [l_state_fw, l_state_bw])
        return outputs, l_state

    def predict_encoder(cell, embedding, zero_state, inputs):
        embedded_list = tf.nn.embedding_lookup(embedding, inputs)
        outputs = []
        state = zero_state
        for embedded in tf.unpack(embedded_list):
            embedded = tf.expand_dims(embedded, 0)
            output, state = cell(embedded, state)
            outputs.append(output)
        return outputs, state

    def predict_decoder(cell, attention, sftm, embedding, states, init_state, seq_len):
        l_state = init_state
        outputs, symbol, possib = [], [], []
        sstates, _ = attention.shortcut(states)
        for i in range(seq_len):
            if i == 0:
                embedded = tf.nn.embedding_lookup(embedding, [0]) # 0 for _START, also as a begin signal
            else:
                embedded = tf.nn.embedding_lookup(embedding, symbol[-1])
            eij = []
            for h in sstates:
                eij.append(attention(l_state, h))
            eij = tf.concat(1, eij)
            alphaij = tf.nn.softmax(eij) # pay attention to softmax, it computes along axis=1
            t_ = alphaij * tf.transpose(tf.pack(states)) # broadcastable
            t_ = tf.transpose(t_)
            ci = tf.reduce_sum(t_, 0)
            output, l_state = cell(embedded, l_state, ci)
            outputs.append(output)
            res = sftm(l_state, None, None, True)
            if sess.run(res[0])[0] == 1: # encountered with symbol _END
                break
            symbol.append(res[0])
            possib.append(res[1])
        return outputs, symbol, possib

    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.device("/cpu:0"):
                #init_post_vocab_size(post_word_count)
                #init_response_vocab_size(response_word_count)
                init_vocab_size(word_count)

                encoder_fw, encoder_bw, attention, sftm, decoder = initialization()
                embedding = tf.Variable(word_emb, trainable=False)

                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_file)
                print("Model restored.")

                inputs = "我 时常 默默地 问 自己"
                for iline in open('data/diff_betw_test_and_train_post').readlines():
#                for iline, oline in zip(open('data/test_words.en').readlines(), open('data/test_words.fr').readlines()):
                    inputs = iline.replace('\n', '')
                    print 'Post:', inputs
                    #print 'Reference response:', oline.replace('\n', '')
                    ids_fw = sentence_2_id(inputs, word2id, word_count)
                    ids_bw = ids_fw[::-1]

                    zero_state = np.zeros((1, options['state_size']), "float32")
                    #outputs, l_state = predict_encoder(encoder, post_embedding, zero_state, ids)
                    outputs, l_state = predict_bidirection_encoder(encoder_fw, encoder_bw, embedding, zero_state, ids_fw, ids_bw)
                    _, symbols, _ = predict_decoder(decoder, attention, sftm, embedding, outputs, l_state, 50)
                    
                    _s = sess.run(symbols)
                    s = []
                    for i in _s:
                        s.append(i[0])
                    ds = id_2_sentence(s, id2word)
                    print 'Response:', ' '.join(ds), '\n'

if __name__ == '__main__':
    training_process()
    #predict_process_4_single_input('tmp/4.47481470108model.ckpt')
