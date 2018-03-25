from util import *
import random
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity




def classify(document):

    file_train_instances = "train_stances.csv"
    file_train_bodies = "train_bodies.csv"
    file_test_instances = "test_stances_unlabeled.csv"
    file_test_instances = "test_stances_unlabeled.csv"
    file_test_bodies = "test_bodies.csv"
    file_predictions = 'predictions_test.csv'

    label_ref = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
    label_ref_rev = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}


# Initialise hyperparameters
    r = random.Random()
    lim_unigram = 5000
    target_size = 4
    hidden_size = 100
    train_keep_prob = 0.6
    l2_alpha = 0.00001
    learn_rate = 0.01
    clip_ratio = 5
    batch_size_train = 500
    epochs = 90

    raw_train = FNCData(file_train_instances, file_train_bodies)
    raw_test = FNCData(file_test_instances, file_test_bodies)
    n_train = len(raw_train.instances)
    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
    pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
    feature_size = len(train_set[0])

    features_pl = tf.placeholder(tf.float32, [None, feature_size], 'features')
    stances_pl = tf.placeholder(tf.int64, [None], 'stances')
    keep_prob_pl = tf.placeholder(tf.float32)

    # Infer batch size
    batch_size = tf.shape(features_pl)[0]

    # Define multi-layer perceptron
    hidden_layer = tf.nn.dropout(tf.nn.relu(tf.contrib.layers.linear(features_pl, hidden_size)), keep_prob=keep_prob_pl)
    logits_flat = tf.nn.dropout(tf.contrib.layers.linear(hidden_layer, target_size), keep_prob=keep_prob_pl)
    logits = tf.reshape(logits_flat, [batch_size, target_size])

    # Define L2 loss
    tf_vars = tf.trainable_variables()
    l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf_vars if 'bias' not in v.name]) * l2_alpha

    # Define overall loss
    loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, stances_pl) + l2_loss)

    # Define prediction
    softmaxed_logits = tf.nn.softmax(logits)
    predict = tf.arg_max(softmaxed_logits, 1)
    predict_score = tf.reduce_max(softmaxed_logits, reduction_indices=[1])
    file_train_instances = "train_stances.csv"
    file_train_bodies = "train_bodies.csv"
    raw_train = FNCData(file_train_instances, file_train_bodies)
    train_set, train_stances, bow_vectorizer, tfreq_vectorizer, tfidf_vectorizer = \
        pipeline_train(raw_train, raw_test, lim_unigram=lim_unigram)
    feature_size = len(train_set[0])
    
    
    test_set = []
    heads_track = {}
    bodies_track = {}
    cos_track = {}

    head, body = document

    head_bow = bow_vectorizer.transform([head]).toarray()
    head_tf = tfreq_vectorizer.transform(head_bow).toarray()[0].reshape(1, -1)
    head_tfidf = tfidf_vectorizer.transform([head]).toarray().reshape(1, -1)

    body_bow = bow_vectorizer.transform([body]).toarray()
    body_tf = tfreq_vectorizer.transform(body_bow).toarray()[0].reshape(1, -1)
    body_tfidf = tfidf_vectorizer.transform([body]).toarray().reshape(1, -1)
    tfidf_cos = cosine_similarity(head_tfidf, body_tfidf)[0].reshape(1, 1)

    feat_vec = np.squeeze(np.c_[head_tf, body_tf, tfidf_cos])
    test_set.append(feat_vec)

    with tf.Session() as sess:
        
        load_model(sess)

        test_feed_dict = {features_pl: test_set, keep_prob_pl: 1.0}
        test_pred = sess.run(predict, feed_dict=test_feed_dict)
        test_pred_score = sess.run(predict_score, feed_dict=test_feed_dict)


    return [label_ref_rev[test_pred[0]], test_pred_score[0]]