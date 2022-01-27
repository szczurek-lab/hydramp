import argparse
import csv
import gc
import os
from collections import defaultdict

import numpy as np
from keras import backend as K
from sklearn import metrics, model_selection

import amp.data_utils.data_loader as data_loader
import amp.models.discriminators.amp_classifier_noCONV as noconv_classifier
import amp.utils.classifier_utils as cu
from amp.config import MAX_LENGTH, MIN_LENGTH
from amp.data_utils.sequence import pad, to_one_hot
from amp.evaluation.nas_cas import (
    generate_sequences_unconstrained_based)
from amp.utils.seed import set_seed


def train_loop_cas(xs, ys, model_path=None, ratio=None, run_id=None):
    mode = "CAS"
    assert os.path.isdir(model_path)
    assert run_id is not None

    result_file_path = os.path.join(model_path, f"nas_cas_criteria_{args.tag}.csv")
    if not os.path.isfile(result_file_path):
        with open(result_file_path, 'a+', newline='') as f:
            writer = csv.writer(f)
            col_names = ["mode", "filter_out", "run_id", "ratio",
                         "train_f1", "train_real_f1", "train_gen_f1", "val_real_f1", "val_gen_f1"]
            writer.writerow(col_names)

    out_path_classifier = os.path.join(model_path, "classifier_weight", mode, str(ratio))

    if not os.path.exists(out_path_classifier):
        os.makedirs(out_path_classifier)

    results = defaultdict(list)

    cv_real = model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=36)
    for split_index, (train, val) in enumerate(cv_real.split(x_train, y_train)):

        split_x_train = np.concatenate([x_train[train], xs])
        split_y_train = np.concatenate([y_train[train], ys])

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

        val_real_pred = amp_classifier_model.predict(x_train[val], batch_size=10000) > 0.5
        val_gen_pred = amp_classifier_model.predict(x_gen_test, batch_size=10000) > 0.5

        if ratio != 0.0:  # this is like normal training - no generated data
            train_gen_pred = amp_classifier_model.predict(xs, batch_size=10000) > 0.5

        results['train'].append(metrics.f1_score(split_y_train, train_pred))
        results['real_train'].append(metrics.f1_score(y_train[train], train_real_pred))
        results['real_val'].append(metrics.f1_score(y_train[val], val_real_pred))
        results['gen_val'].append(metrics.f1_score(y_gen_test, val_gen_pred))

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
               np.mean(results['real_val']), np.mean(results['gen_val'])]
        writer.writerow(row)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', required=True)
    parser.add_argument('--decomposer_path', '-d', required=True)
    parser.add_argument('--softmax', '-s', action='store_true')
    parser.add_argument('--tag', '-t')
    parser.add_argument('--runs', '-r', default=10)

    args, _ = parser.parse_known_args()
    model_path = args.model_path
    decomposer_path = args.decomposer_path
    softmax = args.softmax
    if softmax:
        print("Requested Gumbel-Softmax replacement")

    assert os.path.isdir(model_path)
    assert os.path.exists(decomposer_path)
    set_seed(36)

    data_manager = data_loader.AMPDataManager(
        '../../../data/unlabelled_positive.csv',
        '../../../data/unlabelled_negative.csv',
        min_len=MIN_LENGTH,
        max_len=MAX_LENGTH)

    x, y = data_manager.get_merged_data()
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=36)

    for filter_out in [False, True]:
        print("Test data generation...")
        amp_gen_test, nonamp_gen_test = generate_sequences_unconstrained_based(model_path, decomposer_path, y_test,
                                                                               filter_out=filter_out, use_softmax=softmax)
        x_gen_test = pad(to_one_hot(np.concatenate([amp_gen_test, nonamp_gen_test])))
        y_gen_test = np.concatenate([np.ones(len(amp_gen_test)), np.zeros(len(nonamp_gen_test))])
        print('Done!')
        for run_id in range(args.runs):
            # prepare data
            amp_seqs, nonamp_seqs = generate_sequences_unconstrained_based(model_path, decomposer_path, y_train,
                                                                           filter_out=filter_out, use_softmax=softmax)
            seqs = pad(to_one_hot(amp_seqs + nonamp_seqs))
            targets = np.concatenate([np.ones(len(amp_seqs)), np.zeros(len(nonamp_seqs))])

            # CAS
            for ratio in np.linspace(0.2, 1.0, 5):
                # we need 'ratio ' portion, bu we
                # take only 'ratio' portion of generated sequences
                gen_seq_amp_idx = int(len(amp_seqs) * ratio)
                gen_seq_nonamp_idx = int(len(nonamp_seqs) * ratio)
                gen_x_train = amp_seqs[:gen_seq_amp_idx] + nonamp_seqs[:gen_seq_nonamp_idx]
                gen_y_train = np.concatenate([np.ones(gen_seq_amp_idx), np.zeros(gen_seq_nonamp_idx)])
                gen_x_train = pad(to_one_hot(gen_x_train))

                train_loop_cas(gen_x_train, gen_y_train, model_path=model_path, ratio=ratio, run_id=run_id)


