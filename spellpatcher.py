import pandas as pd
import numpy as np
import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
from collections import namedtuple
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
import time
import re
from sklearn.model_selection import train_test_split

# To log training/testing graphs
import wandb
wandb.init(project="SpellPatcher", entity="gokul-sunilkumar")


def load_book(path):
    input_file = os.path.join(path)
    with open(input_file) as f:
        book = f.read()
    return book

path = './books/'
book_files = [f for f in listdir(path) if isfile(join(path, f))]
book_files = book_files[1:]

books = []
for book in book_files:
    books.append(load_book(path+book))

print("Books loaded.")
print("")

for i in range(len(books)):
    print("There are {} words in {}.".format(len(books[i].split()), book_files[i]))


def ids_to_sentences(sentence_id_list):
    resulting_sentence_list = []
    pad = vocab_to_int["<PAD>"] 
    for sentence in sentence_id_list:
        resulting_sentence_list.append("".join([int_to_vocab[i] for i in sentence if i != pad]))

    return resulting_sentence_list

def clean_text(text):
    text = re.sub(r'\n', ' ', text) 
    text = re.sub(r'[{}@_*>()\\#%+=\[\]]','', text)
    text = re.sub('a0','', text)
    text = re.sub('\'92t','\'t', text)
    text = re.sub('\'92s','\'s', text)
    text = re.sub('\'92m','\'m', text)
    text = re.sub('\'92ll','\'ll', text)
    text = re.sub('\'91','', text)
    text = re.sub('\'92','', text)
    text = re.sub('\'93','', text)
    text = re.sub('\'94','', text)
    text = re.sub('\.','. ', text)
    text = re.sub('\!','! ', text)
    text = re.sub('\?','? ', text)
    text = re.sub(' +',' ', text)
    return text


clean_books = []
for book in books:
    clean_books.append(clean_text(book))

print("Sentences cleaned.")
print("")

vocab_to_int = {}
count = 0
for book in clean_books:
    for character in book:
        if character not in vocab_to_int:
            vocab_to_int[character] = count
            count += 1

# <GO> token for start state of the decoder

codes = ['<PAD>','<EOS>','<GO>']
for code in codes:
    vocab_to_int[code] = count
    count += 1


vocab_size = len(vocab_to_int)
print("The vocabulary contains {} characters.".format(vocab_size))
print(sorted(vocab_to_int))


int_to_vocab = {}
for character, value in vocab_to_int.items():
    int_to_vocab[value] = character

sentences = []
for book in clean_books:
    for sentence in book.split('. '):
        sentences.append(sentence + '.')
print("There are {} sentences.".format(len(sentences)))


int_sentences = []

for sentence in sentences:
    int_sentence = []
    for character in sentence:
        int_sentence.append(vocab_to_int[character])
    int_sentences.append(int_sentence)

lengths = []
for sentence in int_sentences:
    lengths.append(len(sentence))
lengths = pd.DataFrame(lengths, columns=["counts"])

lengths.describe()

max_length = 200
min_length = 10

good_sentences = []

for sentence in int_sentences:
    if len(sentence) <= max_length and len(sentence) >= min_length:
        good_sentences.append(sentence)
        

print("{} sentences available to train and test our model.".format(len(good_sentences)))



# Split the data into training and remaining sentences
training, remaining = train_test_split(good_sentences, test_size = 0.25, random_state = 2)

print(ids_to_sentences([training[0]]))

print("Number of training sentences:", len(training))
print("Number of testing and validation sentences:", len(remaining))

validation, testing = train_test_split(remaining, test_size = 0.4, random_state = 2)
print("Number of validation sentences:", len(validation))
print("Number of testing sentences:", len(testing))

training_sorted = []
validation_sorted = []
testing_sorted = []


for i in range(min_length, max_length+1):
    for sentence in training:
        if len(sentence) == i:
            training_sorted.append(sentence)
    for sentence in validation:
        if len(sentence) == i:
            validation_sorted.append(sentence)
    for sentence in testing:
        if len(sentence) == i:
            testing_sorted.append(sentence)

for i in range(5):
    print(training_sorted[i], len(training_sorted[i]))

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m',
           'n','o','p','q','r','s','t','u','v','w','x','y','z',]

def noise_maker(sentence, threshold):
    
    noisy_sentence = []
    i = 0
    while i < len(sentence):
        random = np.random.uniform(0,1,1)
        if random < threshold:
            noisy_sentence.append(sentence[i])
        else:
            new_random = np.random.uniform(0,1,1)
            if new_random > 0.67:
                # If last character, delete 
                if i == (len(sentence) - 1):
                    continue
                # Else, swap
                else:
                    noisy_sentence.append(sentence[i+1])
                    noisy_sentence.append(sentence[i])
                    i += 1
            elif new_random < 0.33:
                # Choose a random character and insert
                random_letter = np.random.choice(letters, 1)[0]
                noisy_sentence.append(vocab_to_int[random_letter])
                noisy_sentence.append(sentence[i])
            else:
                pass     
        i += 1
    return noisy_sentence


threshold = 0.99
for sentence in training_sorted[:5]:
    print(sentence)
    print(noise_maker(sentence, threshold))
    print()


def model_inputs():
    
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    inputs_length = tf.placeholder(tf.int32, (None,), name='inputs_length')
    targets_length = tf.placeholder(tf.int32, (None,), name='targets_length')
    max_target_length = tf.reduce_max(targets_length, name='max_target_len')

    return inputs, targets, keep_prob, inputs_length, targets_length, max_target_length



def process_encoding_input(targets, vocab_to_int, batch_size):
    
    with tf.name_scope("process_encoding"):
        ending = tf.strided_slice(targets, [0, 0], [batch_size, -1], [1, 1])
        dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input



def encoding_layer(rnn_size, sequence_length, num_layers, rnn_inputs, keep_prob, direction):
    
    if direction == 1:
        with tf.name_scope("RNN_Encoder_Cell_1D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    lstm = tf.contrib.rnn.LSTMCell(rnn_size)

                    drop = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.dynamic_rnn(drop,rnn_inputs,sequence_length,dtype=tf.float32)

            return enc_output, enc_state
        
        
    if direction == 2:
        with tf.name_scope("RNN_Encoder_Cell_2D"):
            for layer in range(num_layers):
                with tf.variable_scope('encoder_{}'.format(layer)):
                    cell_fw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,input_keep_prob = keep_prob)

                    cell_bw = tf.contrib.rnn.LSTMCell(rnn_size)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,input_keep_prob = keep_prob)

                    enc_output, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,rnn_inputs,sequence_length,dtype=tf.float32)

            enc_output = tf.concat(enc_output,2)
            return enc_output, enc_state[0]



def training_decoding_layer(dec_embed_input, targets_length, dec_cell, initial_state, output_layer, 
                            vocab_size, max_target_length):
    
    with tf.name_scope("Training_Decoder"):
        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,sequence_length=targets_length,time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,training_helper,initial_state,output_layer) 

        training_logits, _ = tf.contrib.seq2seq.dynamic_decode(training_decoder,output_time_major=False,impute_finished=True,maximum_iterations=max_target_length)

        return training_logits



def inference_decoding_layer(embeddings, start_token, end_token, dec_cell, initial_state, output_layer,max_target_length, batch_size):
    
    with tf.name_scope("Inference_Decoder"):
        start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')

        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens,end_token)

        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,inference_helper,initial_state,output_layer)

        inference_logits, _ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,output_time_major=False,impute_finished=True,maximum_iterations=max_target_length)

        return inference_logits



def decoding_layer(dec_embed_input, embeddings, enc_output, enc_state, vocab_size, inputs_length, targets_length, 
                   max_target_length, rnn_size, vocab_to_int, keep_prob, batch_size, num_layers, direction):
    
    with tf.name_scope("RNN_Decoder_Cell"):
        for layer in range(num_layers):
            with tf.variable_scope('decoder_{}'.format(layer)):
                lstm = tf.contrib.rnn.LSTMCell(rnn_size)
                dec_cell = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
    
    output_layer = Dense(vocab_size,kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.1))
    
    attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_size,enc_output,inputs_length,normalize=False,name='BahdanauAttention')
    
    with tf.name_scope("Attention_Wrapper"):
        dec_cell = tf.contrib.seq2seq.AttentionWrapper(dec_cell,attn_mech,rnn_size)
    
    initial_state =  dec_cell.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=enc_state)

    with tf.variable_scope("decode"):

        training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,sequence_length=targets_length,time_major=False)

        training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,training_helper,initial_state,output_layer) 

        training_logits, _ ,_ = tf.contrib.seq2seq.dynamic_decode(training_decoder,output_time_major=False,impute_finished=True,maximum_iterations=max_target_length)
        
    with tf.variable_scope("decode", reuse=True):

        start_tokens = tf.tile(tf.constant([vocab_to_int['<GO>']], dtype=tf.int32), [batch_size], name='start_tokens')
        end_token = (tf.constant(vocab_to_int['<EOS>'], dtype=tf.int32))
        inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embeddings,start_tokens,end_token)


        inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,inference_helper,initial_state,output_layer)


        inference_logits, _ ,_ = tf.contrib.seq2seq.dynamic_decode(inference_decoder,output_time_major=False,impute_finished=True,maximum_iterations=max_target_length)

    return training_logits, inference_logits



def seq2seq_model(inputs, targets, keep_prob, inputs_length, targets_length, max_target_length, 
                  vocab_size, rnn_size, num_layers, vocab_to_int, batch_size, embedding_size, direction):
    
    enc_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    enc_embed_input = tf.nn.embedding_lookup(enc_embeddings, inputs)
    enc_output, enc_state = encoding_layer(rnn_size, inputs_length, num_layers,enc_embed_input, keep_prob, direction)
    
    dec_embeddings = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1, 1))
    dec_input = process_encoding_input(targets, vocab_to_int, batch_size)
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    
    training_logits, inference_logits  = decoding_layer(dec_embed_input,dec_embeddings,enc_output,enc_state,vocab_size,inputs_length, 
                                                        targets_length,max_target_length,rnn_size,vocab_to_int,keep_prob,batch_size,num_layers,direction)
    
    return training_logits, inference_logits



def pad_sentence_batch(sentence_batch):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(sentences, batch_size, threshold):
    
    for batch_i in range(0, len(sentences)//batch_size):
        start_i = batch_i * batch_size
        sentences_batch = sentences[start_i:start_i + batch_size]
        
        sentences_batch_noisy = []
        for sentence in sentences_batch:
            sentences_batch_noisy.append(noise_maker(sentence, threshold))
            
        sentences_batch_eos = []
        for sentence in sentences_batch:
            sentence.append(vocab_to_int['<EOS>'])
            sentences_batch_eos.append(sentence)
            
        pad_sentences_batch = np.array(pad_sentence_batch(sentences_batch_eos))
        pad_sentences_noisy_batch = np.array(pad_sentence_batch(sentences_batch_noisy))
        
        pad_sentences_lengths = []
        for sentence in pad_sentences_batch:
            pad_sentences_lengths.append(len(sentence))
        
        pad_sentences_noisy_lengths = []
        for sentence in pad_sentences_noisy_batch:
            pad_sentences_noisy_lengths.append(len(sentence))
        
        yield pad_sentences_noisy_batch, pad_sentences_batch, pad_sentences_noisy_lengths, pad_sentences_lengths


#Hyperparameters
epochs = 20
batch_size = 128
num_layers = 4
rnn_size = 512
embedding_size = 128
learning_rate = 0.0005
direction = 2
threshold = 0.99
keep_probability = 0.75



def build_graph(keep_prob, rnn_size, num_layers, batch_size, learning_rate, embedding_size, direction):

    tf.reset_default_graph()
    
    inputs, targets, keep_prob, inputs_length, targets_length, max_target_length = model_inputs()

    training_logits, inference_logits = seq2seq_model(tf.reverse(inputs, [-1]),targets,keep_prob,inputs_length,targets_length,max_target_length,len(vocab_to_int)+1,
                                                      rnn_size,num_layers,vocab_to_int,batch_size,embedding_size,direction)

    training_logits = tf.identity(training_logits.rnn_output, 'logits')

    with tf.name_scope('predictions'):
        predictions = tf.identity(inference_logits.sample_id, name='predictions')
        tf.summary.histogram('predictions', predictions)

    masks = tf.sequence_mask(targets_length, max_target_length, dtype=tf.float32, name='masks')
    
    with tf.name_scope("cost"):
        cost = tf.contrib.seq2seq.sequence_loss(training_logits,targets,masks)
        tf.summary.scalar('cost', cost)

    with tf.name_scope("optimze"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

    merged = tf.summary.merge_all()    

    export_nodes = ['inputs', 'targets', 'keep_prob', 'cost', 'inputs_length', 'targets_length',
                    'predictions', 'merged', 'train_op','optimizer']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])

    return graph



def train(model, epochs, log_string):
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        testing_loss_summary = []

        iteration = 0
        
        display_step = 50 
        stop_early = 0 
        stop = 25 
        per_epoch = 3 
        testing_check = (len(training_sorted)//batch_size//per_epoch)-1

        print()
        print("Training Model: {}".format(log_string))

        train_writer = tf.summary.FileWriter('./logs/1/train/{}'.format(log_string), sess.graph)
        test_writer = tf.summary.FileWriter('./logs/1/test/{}'.format(log_string))

        for epoch_i in range(1, epochs+1): 
            batch_loss = 0
            batch_time = 0
            
            for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                    get_batches(training_sorted, batch_size, threshold)):
                start_time = time.time()

                summary, loss, _ = sess.run([model.merged,
                                             model.cost, 
                                             model.train_op], 
                                             {model.inputs: input_batch,
                                              model.targets: target_batch,
                                              model.inputs_length: input_length,
                                              model.targets_length: target_length,
                                              model.keep_prob: keep_probability})


                batch_loss += loss
                wandb.log({"Training_Loss": loss})
                end_time = time.time()
                batch_time += end_time - start_time

                train_writer.add_summary(summary, iteration)

                iteration += 1

                if batch_i % display_step == 0 and batch_i > 0:
                    print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs, 
                                  batch_i, 
                                  len(training_sorted) // batch_size, 
                                  batch_loss / display_step, 
                                  batch_time))
                    batch_loss = 0
                    batch_time = 0

                if batch_i % testing_check == 0 and batch_i > 0:
                    batch_loss_testing = 0
                    batch_time_testing = 0
                    for batch_i, (input_batch, target_batch, input_length, target_length) in enumerate(
                            get_batches(validation_sorted, batch_size, threshold)):
                        start_time_testing = time.time()
                        summary, loss = sess.run([model.merged,
                                                  model.cost], 
                                                     {model.inputs: input_batch,
                                                      model.targets: target_batch,
                                                      model.inputs_length: input_length,
                                                      model.targets_length: target_length,
                                                      model.keep_prob: 1})

                        batch_loss_testing += loss
                        wandb.log({"Testing_Loss":loss})
                        end_time_testing = time.time()
                        batch_time_testing += end_time_testing - start_time_testing

                        test_writer.add_summary(summary, iteration)

                    n_batches_testing = batch_i + 1
                    print('Testing Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(batch_loss_testing / n_batches_testing, 
                                  batch_time_testing))
                    
                    batch_time_testing = 0

                    testing_loss_summary.append(batch_loss_testing)
                    if batch_loss_testing <= min(testing_loss_summary):
                        print('New Record!') 
                        stop_early = 0
                        checkpoint = "./model3/{}.ckpt".format(log_string)
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == stop:
                            break

            if stop_early == stop:
                print("Stopping Training.")
                break



for keep_probability in [0.75]:
    for num_layers in [2]:
        for threshold in [0.95]:
            log_string = 'kp={},nl={},th={}'.format(keep_probability,
                                                    num_layers,
                                                    threshold) 
            model = build_graph(keep_probability, rnn_size, num_layers, batch_size, 
                                learning_rate, embedding_size, direction)
            train(model, epochs, log_string)
