#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os

model_path = None  # assign if using notebook directly
decomposer_path = None
softmax = False
if not model_path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m')
    parser.add_argument('--decomposer_path', '-d')
    parser.add_argument('--softmax', '-s', action='store_true')
    parser.add_argument('--nas', '-n', action='store_true')
    parser.add_argument('--cas', '-c', action='store_true')
    parser.add_argument('--tag', '-t')

    args, _ = parser.parse_known_args()
    model_path = args.model_path
    decomposer_path = args.decomposer_path
    softmax = args.softmax
    if not (args.nas or args.cas):
        raise parser.error('No action requested, add --nas and/or --cas')

    if args.nas:
        print("Calculating all NAS modes")
    if args.cas:
        print("Calculating CAS mode")
    if softmax:
        print("Requested Gumbel-Softmax replacement")

assert os.path.isdir(model_path)
assert os.path.exists(decomposer_path)

# In[ ]:


import csv
import gc
import amp.data_utils.data_loader as data_loader
import amp.models.discriminators.amp_classifier_noCONV as noconv_classifier
import amp.utils.classifier_utils as cu
import numpy as np
import tensorflow as tf
from amp.config import MAX_LENGTH, MIN_LENGTH, VOCAB_PAD_SIZE
from amp.utils.seed import set_seed
from sklearn import metrics, model_selection
from amp.evaluation.nas_cas import generate_sequences_unconstrained_based, generate_sequences_template_based, \
    generate_sequences_template_based_boundary
from amp.data_utils.sequence import pad, to_one_hot
from amp.utils.generate_peptides import translate_peptide
from collections import defaultdict
from keras import backend as K

set_seed(36)

# In[ ]:


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)

# In[ ]:


data_manager = data_loader.AMPDataManager(
    '../../../data/unlabelled_positive.csv',
    '../../../data/unlabelled_negative.csv',
    min_len=MIN_LENGTH,
    max_len=MAX_LENGTH)

x, y = data_manager.get_merged_data()
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=36)


# In[ ]:


def train_loop_nas(xs, ys, model_path=None, model_name=None, decomposer_path=None, ratio=None, run_id=None, mode=None,
                   filter_out=None):
    assert os.path.isdir(model_path)
    assert run_id is not None

    result_file_path = os.path.join(model_path, "nas_cas_filtered_fix.csv")
    if not os.path.isfile(result_file_path):
        with open(result_file_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            col_names = ["mode", "filter_out", "run_id", "ratio",
                         "validation_auc", "validation_accuracy", "validation_f1",
                         "test_auc", "test_accuracy", "test_f1"]
            writer.writerow(col_names)

    oof_preds = []
    out_path_classifier = os.path.join(model_path, "classifier_weight", mode)

    test_preds = []
    cv = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=36)
    if not os.path.exists(out_path_classifier):
        os.makedirs(out_path_classifier)
    for split_index, (train, val) in enumerate(cv.split(xs, ys)):
        K.clear_session()
        print(len(train))
        print('Fold %s' % split_index)
        logger = cu.ClassifierLogger(
            out_fn=f'/{split_index}.h5',
            patience=10,
            out_path=out_path_classifier,
        )
        amp_classifier_model = noconv_classifier.NoConvAMPClassifierFactory.get_default(max_length=MAX_LENGTH)
        amp_classifier_model = amp_classifier_model()
        amp_classifier_model.fit(
            xs[train], ys[train],
            epochs=100,
            batch_size=128,
            validation_data=(x_train, y_train),
            verbose=1,
            callbacks=[logger]
        )
        amp_classifier_model.load_weights(os.path.join(out_path_classifier, f'{split_index}.h5'))
        test_pred = amp_classifier_model.predict(x_test, batch_size=64)
        test_preds.append(test_pred)

        oof_preds.append(amp_classifier_model.predict(x_train))
        del amp_classifier_model
        _ = gc.collect()

    test_avg = np.mean(np.asarray(test_preds), axis=0).flatten()
    test_avg_bin = test_avg > 0.5
    val_preds_bin = np.array(oof_preds) > 0.5

    oof_preds = np.concatenate(oof_preds)
    ys = np.concatenate(10 * [y_train])

    validation_auc = metrics.roc_auc_score(ys, oof_preds)
    test_auc = metrics.roc_auc_score(y_test, test_avg)

    print(f'Validation AUC: {validation_auc}')
    print(f'Test AUC: {test_auc}')

    validation_accuracy = metrics.accuracy_score(ys, val_preds_bin)
    test_accuracy = metrics.accuracy_score(y_test, test_avg_bin)
    print(f'Validation accuracy: {validation_accuracy}')
    print(f'Test accuracy: {test_accuracy}')

    validation_f1 = metrics.f1_score(ys, val_preds_bin)
    test_f1 = metrics.f1_score(y_test, test_avg_bin)

    print(f'Validation F1: {validation_f1}')
    print(f'Test F1: {test_f1}')

    with open(result_file_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        row = [mode, str(filter_out), run_id, ratio,
               validation_auc, validation_accuracy, validation_f1,
               test_auc, test_accuracy, test_f1]
        writer.writerow(row)


# In[ ]:


def train_loop_cas(xs, ys,
                   model_path=None, model_name=None, decomposer_path=None,
                   ratio=None, run_id=None, filter_out=None):
    mode = "CAS"
    assert os.path.isdir(model_path)
    assert run_id is not None

    result_file_path = os.path.join(model_path, f"nas_cas_criteria_{args.tag}.csv")
    if not os.path.isfile(result_file_path):
        with open(result_file_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            col_names = ["mode", "filter_out", "run_id", "ratio",
                         "train_f1", "train_real_f1", "train_gen_f1", "criteria_f1",
                         "val_f1", "val_real_f1", "val_real_f1"]
            writer.writerow(col_names)

    out_path_classifier = os.path.join(model_path, "classifier_weight", mode, str(ratio))

    if not os.path.exists(out_path_classifier):
        os.makedirs(out_path_classifier)

    results = defaultdict(list)

    cv_real = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=36)
    for split_index, (train, val) in enumerate(cv_real.split(x_train, y_train)):

        split_x_train = np.concatenate([x_train[train]], xs)
        split_y_train = np.concatenate([y_train[train]], ys)

        split_x_val = np.concatenate([x_train[val]], x_gen_test)
        split_y_val = np.concatenate([y_train[val]], y_gen_test)

        K.clear_session()
        print(len(train))
        print('Fold %s' % split_index)
        logger = cu.ClassifierLogger(
            out_fn=f'/{split_index}.h5',
            patience=10,
            out_path=out_path_classifier,
        )
        amp_classifier_model = noconv_classifier.NoConvAMPClassifierFactory.get_default(max_length=MAX_LENGTH)
        amp_classifier_model = amp_classifier_model()
        amp_classifier_model.fit(
            split_x_train, split_y_train,
            epochs=100,
            batch_size=128,
            validation_data=(x_train[val], y_train[val]),
            verbose=1,
            callbacks=[logger]
        )
        amp_classifier_model.load_weights(os.path.join(out_path_classifier, f'{split_index}.h5'))

        train_pred = amp_classifier_model.predict(split_x_train, batch_size=10000) > 0.5
        train_real_pred = amp_classifier_model.predict(x_train[train], batch_size=10000) > 0.5

        val_pred = amp_classifier_model.predict(split_x_val, batch_size=10000) > 0.5
        val_real_pred = amp_classifier_model.predict(x_train[val], batch_size=10000) > 0.5
        val_gen_pred = amp_classifier_model.predict(x_gen_test, batch_size=10000) > 0.5

        if ratio != 0.0:  # this is like normal training - no generated data
            train_gen_pred = amp_classifier_model.predict(xs, batch_size=10000) > 0.5

        results['train'].append(metrics.f1_score(split_y_train, train_pred))
        results['val'].append(metrics.f1_score(split_y_val, val_pred))
        results['real_train'].append(metrics.f1_score(y_train[train], train_real_pred))
        results['real_val'].append(metrics.f1_score(y_train[val], val_real_pred))
        results['gen_val'].append(metrics.f1_score(y_gen_validation, val_gen_pred))

        if ratio != 0.0:
            results['gen_train'].append(metrics.f1_score(ys, train_gen_pred))
        else:
            results['gen_train'].append(0.0)

        del amp_classifier_model
        _ = gc.collect()

    with open(result_file_path, 'a+', newline='') as f:
        writer = csv.writer(f)
        row = [mode, str(filter_out), run_id, ratio,
               np.mean(results['train']), np.mean(results['real_train']), np.mean(results['gen_train']),
               np.mean(results['val']), np.mean(results['real_val']), np.mean(results['gen_val'])]
        writer.writerow(row)


# ## unconstrained-based NAS and CAS

# In[ ]:


for filter_out in [False, True]:
    print("Test data generation...")
    amp_gen_test, nonamp_gen_test = generate_sequences_unconstrained_based(model_path, decomposer_path, y_test,
                                                                           filter_out=filter_out, use_softmax=softmax)
    x_gen_test = np.concatenate([amp_gen_test, nonamp_gen_test])
    y_gen_test = np.concatenate([np.ones(len(amp_gen_test)), np.zeros(len(nonamp_gen_test))])
    print('Done!')
    for run_id in range(30):
        # prepare data
        amp_seqs, nonamp_seqs = generate_sequences_unconstrained_based(model_path, decomposer_path, y_train,
                                                                       filter_out=filter_out, use_softmax=softmax)
        seqs = pad(to_one_hot(amp_seqs + nonamp_seqs))
        targets = np.concatenate([np.ones(len(amp_seqs)), np.zeros(len(nonamp_seqs))])

        # NAS
        if args.nas:
            train_loop_nas(seqs, targets,
                           model_path=model_path, decomposer_path=decomposer_path,
                           mode="NAS", ratio=None, run_id=run_id, filter_out=filter_out)

        # CAS
        if args.cas:
            for ratio in np.linspace(0.0, 1.0, 6):
                # we need 'ratio ' portion, bu we
                # take only 'ratio' portion of generated sequences
                gen_seq_amp_idx = int(len(amp_seqs) * ratio)
                gen_seq_nonamp_idx = int(len(nonamp_seqs) * ratio)
                gen_x_train = amp_seqs[:gen_seq_amp_idx] + nonamp_seqs[:gen_seq_nonamp_idx]
                gen_y_train = np.concatenate([np.ones(gen_seq_amp_idx), np.zeros(gen_seq_nonamp_idx)])
                gen_x_train = pad(to_one_hot(gen_x_train))

                train_loop_cas(gen_x_train, gen_y_train,
                               model_path=model_path, decomposer_path=decomposer_path,
                               ratio=ratio, run_id=run_id, filter_out=filter_out)

# ## template-based NAS

# In[ ]:


# NAS conditioned
if args.nas:
    for run_id in range(10):
        amp_seqs, nonamp_seqs = generate_sequences_template_based(model_path,
                                                                  decomposer_path,
                                                                  np.array([translate_peptide(x) for x in x_train]),
                                                                  y_train,
                                                                  use_softmax=softmax,
                                                                  constraint='absolute')
        seqs_cond = pad(to_one_hot(amp_seqs + nonamp_seqs))
        targets_cond = np.concatenate([np.ones(len(amp_seqs)), np.zeros(len(nonamp_seqs))])
        train_loop(seqs_cond, targets_cond, model_path=model_path, decomposer_path=decomposer_path,
                   mode="NAS_conditioned", ratio=None, run_id=run_id)

        amp_seqs, nonamp_seqs = generate_sequences_template_based_boundary(model_path,
                                                                           decomposer_path,
                                                                           np.array(
                                                                               [translate_peptide(x) for x in x_train]),
                                                                           use_softmax=softmax,
                                                                           constraint='absolute')
        seqs_cond = pad(to_one_hot(amp_seqs + nonamp_seqs))
        targets_cond = np.concatenate([np.ones(len(amp_seqs)), np.zeros(len(nonamp_seqs)), y_train])
        seqs_cond = np.concatenate([seqs_cond, x_train])
        train_loop(seqs_cond, targets_cond, model_path=model_path, decomposer_path=decomposer_path,
                   mode="NAS_conditioned_boundary", ratio=None, run_id=run_id)

