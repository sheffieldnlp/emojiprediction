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
import numpy as np
from torch.autograd import Variable
from sklearn import metrics
from datetime import datetime
torch.manual_seed(123)
torch.cuda.manual_seed(123)

use_plot = False
use_save = False
if use_save:
    import pickle

DATA_DIR = 'data'
TRAIN_DIR = 'train_txt_emoji'
TEST_DIR = 'test_txt_emoji'
TASK_DIR = 'test_shared_txt_emoji'
TRAIN_FILE = 'train_txt_emoji.txt'
TEST_FILE = 'test_txt_emoji.txt'
TASK_FILE = 'test_shared_txt_emoji.txt'
TRAIN_LABEL = 'train_label_emoji.txt'
TEST_LABEL = 'test_label_emoji.txt'
TASK_LABEL = 'test_shared_label_emoji.txt'
TRAIN_IMG = 'train_img_emoji'
TEST_IMG = 'test_img_emoji'
TASK_IMG = 'test_shared_img_emoji'



## parameter setting
epochs = 10
batch_size = 5000
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
    hidden_dim = 300
    lin_dim = 300
    sentence_len = 22
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)
    test_file = os.path.join(DATA_DIR, TEST_FILE)

    task_file = os.path.join(DATA_DIR, TASK_FILE)

    pemb = torch.from_numpy(np.load('pretrained_emoji_vocab.npy'))
    fp_train = open(train_file, 'r')
    train_filenames = [os.path.join(TRAIN_DIR, line.strip()) for line in fp_train]
    filenames = copy.deepcopy(train_filenames)
    fp_train.close()
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, line.strip()) for line in fp_test]
    fp_test.close()
    filenames.extend(test_filenames)

    fp_task = open(task_file, 'r')
    task_filenames = [os.path.join(TASK_DIR, line.strip()) for line in fp_task]
    fp_task.close()
    filenames.extend(task_filenames)

    corpus = DP.Corpus(DATA_DIR, filenames)
    nlabel = 20


    ### create model
    model = LSTMC.LSTMClassifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim, lin_dim=lin_dim,
                           vocab_size=len(corpus.dictionary),label_size=nlabel,
                           batch_size=batch_size, use_gpu=use_gpu,
                           pretrained=pemb)
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

    dtask_set = DP.TxtDatasetProcessing(DATA_DIR, TASK_DIR, TASK_IMG,
            TASK_FILE, TASK_LABEL, sentence_len, corpus)

    task_loader = DataLoader(dtask_set,
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
    test_f1_micro = []
    test_f1_macro = []
    test_rec = []
    test_prec = []


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
        tpred = []
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

            f1_micro = metrics.f1_score(test_labels.cpu().numpy(),
                    predicted.cpu().numpy(), average='micro')
            f1_macro = metrics.f1_score(test_labels.cpu().numpy(),
                    predicted.cpu().numpy(), average='macro')
            rec = metrics.recall_score(test_labels.cpu().numpy(),
                    predicted.cpu().numpy(), average='micro')
            prec = metrics.precision_score(test_labels.cpu().numpy(),
                    predicted.cpu().numpy(), average='micro')


            tpred += predicted.cpu().numpy().tolist()

        test_loss_.append(total_loss / total)
        test_acc_.append(total_acc / total)
        test_f1_micro.append(f1_micro / it)
        test_f1_macro.append(f1_macro / it)
        test_rec.append(rec / it)
        test_prec.append(prec / it)





#        print('[Ep: %3d/%3d] TrL: %.8f, TeL: %.8f, TrAcc: %.5f, TeAcc: %.5f, True: %.5f, Fake: %.5f'
#              % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch]*100, test_acc_[epoch]*100, \
#              test_corr_acc[epoch]*100, test_fake_acc[epoch]*100))


        print('[Ep: %3d/%3d] TrL: %.8f, TeL: %.8f, TrAcc: %.2f, TeAcc: %.2f, MiF: %.2f, MaF: %.2f, P: %.2f, R: %.2f'
              % (epoch, epochs, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch]*100, test_acc_[epoch]*100, \
              test_f1_micro[epoch]*100, test_f1_macro[epoch]*100,
              test_rec[epoch]*100, test_prec[epoch]*100))



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
    fname = 'prediction' + datetime.now().strftime("%d-%h-%m-%s") + '.txt'
    print 'Done . . . Saving as a prediction file: ', fname
    np.save(fname,
            tpred)



    taskpred = []
    for iter, taskdata in enumerate(task_loader):
        task_inputs, task_imgs, task_labels = taskdata
        task_labels = torch.squeeze(task_labels)
        it += 1

        if use_gpu:
            task_inputs, task_imgs, task_labels = Variable(task_inputs.cuda()), Variable(task_imgs.cuda(), requires_grad=False), task_labels.cuda()
        else: task_inputs, task_imgs = Variable(task_inputs), Variable(task_imgs, requires_grad=False)

        model.eval()
        model.batch_size = len(task_labels)
        model.hidden = model.init_hidden()
        output = model(task_inputs.t(), task_imgs)

        _, predicted = torch.max(output.data, 1)

        taskpred += predicted.cpu().numpy().tolist()


    fname = 'task' + datetime.now().strftime("%d-%h-%m-%s")
    print 'Done . . . Saving as a prediction file: ', fname
    np.save(fname,
            taskpred)



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
