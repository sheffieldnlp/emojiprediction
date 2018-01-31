#!/usr/bin/python
from __future__ import division
import os
import torch
import copy
from torch.utils.data import DataLoader
import utils.DataProcessing_img as DP
import utils.LSTMClassifier_img as LSTMC
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from sklearn import metrics
torch.manual_seed(123)
torch.cuda.manual_seed(123)

use_plot = True
use_save = True
if use_save:
    import pickle
    from datetime import datetime

DATA_DIR = 'data'
TRAIN_DIR = 'train_txt_emoji'
TEST_DIR = 'test_txt_emoji'
TRAIN_FILE = 'train_txt_emoji.txt'
TEST_FILE = 'test_txt_emoji.txt'
TRAIN_LABEL = 'train_label_emoji.txt'
TEST_LABEL = 'test_label_emoji.txt'
TRAIN_IMG = 'train_img_emoji'
TEST_IMG = 'test_img_emoji'



## parameter setting
epochs = 50
batch_size = 1000
use_gpu = torch.cuda.is_available()
learning_rate = 0.01

def adjust_learning_rate(optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

if __name__=='__main__':
    ### parameter setting
    embedding_dim = 300
    hidden_dim = 500
    lin_dim = 300
    sentence_len = 22
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)
    test_file = os.path.join(DATA_DIR, TEST_FILE)
    fp_train = open(train_file, 'r')
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]
    filenames = copy.deepcopy(train_filenames)
    fp_train.close()
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]
    fp_test.close()
    filenames.extend(test_filenames)

    corpus = DP.Corpus(DATA_DIR, filenames)
    nlabel = 20

    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim, lin_dim=lin_dim,
                           vocab_size=len(corpus.dictionary),label_size=nlabel, batch_size=batch_size, use_gpu=use_gpu)
    if use_gpu:
        model = model.cuda()
    ### data processing
    dtrain_set = DP.TxtDatasetProcessing(DATA_DIR, TRAIN_DIR, TRAIN_IMG, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)

    train_loader = DataLoader(dtrain_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4
                         )
    dtest_set = DP.TxtDatasetProcessing(DATA_DIR, TEST_DIR, TEST_IMG, TEST_FILE, TEST_LABEL, sentence_len, corpus)

    test_loader = DataLoader(dtest_set,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=4
                         )

#    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    test_prec_micro = []
    test_prec_macro = []
    test_rec_micro = []
    test_rec_macro = []


    ### training procedure
    for epoch in range(epochs):
        optimizer = adjust_learning_rate(optimizer, epoch)

        ## training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader):
            train_inputs, train_imgs, train_labels = traindata
            train_labels = torch.squeeze(train_labels)

            if use_gpu:
                train_inputs, train_imgs, train_labels = Variable(train_inputs.cuda()),Variable(train_imgs.cuda(), requires_grad=False, volatile=False), train_labels.cuda()
            else: train_inputs, train_imgs = Variable(train_inputs), Variable(train_imgs, requires_grad=False, volatile=False)
            model.train()
            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs.t(), train_imgs)

            loss = loss_function(output, Variable(train_labels))

            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == train_labels).sum()
            total += len(train_labels)
            total_loss += loss.data[0]

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)
        ## testing epoch
        total_acc = 0.0
        total_loss = 0.0
        ind_true = 0.0
        ind_fake = 0.0
        total = 0.0
        it = 0.0
        for iter, testdata in enumerate(test_loader):
            test_inputs, test_imgs, test_labels = testdata
            test_labels = torch.squeeze(test_labels)
            it += 1

            if use_gpu:
                test_inputs, test_imgs, test_labels = Variable(test_inputs.cuda()), Variable(test_imgs.cuda(), requires_grad=False), test_labels.cuda()
            else: test_inputs, test_imgs = Variable(test_inputs), Variable(test_imgs, requires_grad=False)

            model.eval()
            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test_inputs.t(), test_imgs)

            loss = loss_function(output, Variable(test_labels))
            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum()
            total += len(test_labels)
            total_loss += loss.data[0]
            #acc = metrics.accuracy_score(test_labels.cpu().numpy(), predicted.cpu().numpy())
            prec_micro = metrics.precision_score(test_labels.cpu().numpy(),
                    predicted.cpu().numpy(), average='micro')
            prec_macro = metrics.precision_score(test_labels.cpu().numpy(),
                    predicted.cpu().numpy(), average='macro')
            rec_micro = metrics.recall_score(test_labels.cpu().numpy(),
                    predicted.cpu().numpy(), average='micro')
            rec_macro = metrics.recall_score(test_labels.cpu().numpy(),
                    predicted.cpu().numpy(), average='macro')

        test_loss_.append(total_loss / total)
        test_acc_.append(total_acc / total)
        test_prec_micro.append(prec_micro / it)
        test_prec_macro.append(prec_macro / it)
        test_rec_micro.append(rec_micro / it)
        test_rec_macro.append(rec_macro / it)




#        print('[Ep: %3d/%3d] TrL: %.8f, TeL: %.8f, TrAcc: %.5f, TeAcc: %.5f, True: %.5f, Fake: %.5f'
#              % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch]*100, test_acc_[epoch]*100, \
#              test_corr_acc[epoch]*100, test_fake_acc[epoch]*100))

        print('[Ep: %3d/%3d] TrL: %.8f, TeL: %.8f, TrAcc: %.2f, TeAcc: %.2f, MiP: %.2f, MaP: %.2f, MiR: %.2f, MaR: %.2f'
              % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch]*100, test_acc_[epoch]*100, \
              test_prec_micro[epoch]*100, test_prec_macro[epoch]*100,
              test_rec_micro[epoch]*100, test_rec_macro[epoch]*100))


    param = {}
    param['lr'] = learning_rate
    param['batch size'] = batch_size
    param['embedding dim'] = embedding_dim
    param['hidden dim'] = hidden_dim
    param['sentence len'] = sentence_len

    result = {}
    result['train loss'] = train_loss_
    result['test loss'] = test_loss_
    result['train acc'] = train_acc_
    result['test acc'] = test_acc_
    result['param'] = param

    if use_plot:
        import PlotFigure as PF
        PF.PlotFigure(result, use_save)
    if use_save:
        filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
        result['filename'] = filename

        fp = open(filename, 'wb')
        pickle.dump(result, fp)
        fp.close()
        print('File %s is saved.' % filename)
