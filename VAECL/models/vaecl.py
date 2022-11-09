from utils import *
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type = int, default = 64, help = 'batch size')
parser.add_argument('--l_f', type = int, default = 249, help = 'freezing layer for fine-tuning')
parser.add_argument('--pre_epoch', type = int, default = 20, help = 'training epoch for pre-training')
parser.add_argument('--ft_epoch', type = int, default = 100, help = 'training epoch for fine-tuning')
parser.add_argument('--lr', type = float, default = 0.0001, help = 'learning rate')
parser.add_argument('--S', type = int, default = 256, help = 'pixel size of the input image for CNN1')
parser.add_argument('--M', type = int, default = 512, help = 'pixel size of the input image for CNN2')
parser.add_argument('--train', type = bool, default = False, help = 'training or not')
parser.add_argument('--SaveModlFile_1', type = str, default = "./save/model_v5_b1_tune.h5", help = 'save path for CNN1')
parser.add_argument('--SaveModlFile_2', type = str, default = "./save/model_v5_b2_tune.h5", help = 'save path for CNN2')


if __name__ == '__main__':
    start = time.time()
    args = parser.parse_args()

    gpu_setting()
    set_seed()

    # train CNN1
    x_train_1, x_val_1, y_train_1, y_val_1, x_test_1, y_test = get_data(args.S, reshape=False)
    if args.train:
        train_model(x_train_1, x_val_1, y_train_1, y_val_1, args, args.SaveModlFile_1)

    prediction_test_cnn1 = predict(x_test_1, args.SaveModlFile_1)
    accuracy, precision, recall, f1_score, MCC, AUC = evaluate(y_test, prediction_test_cnn1)
    print('Performance for CNN1:')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1_score)
    print('MCC: ', MCC)
    print('AUC: ', AUC)
    print('#########################')

    # train CNN2
    x_train_2 = resize(x_train_1, (args.M, args.M))
    x_test_2 = resize(x_test_1, (args.M, args.M))
    x_val_2 = resize(x_val_1, (args.M, args.M))

    if args.train:
        train_model(x_train_2, x_val_2, y_train_1, y_val_1, args, args.SaveModlFile_2)

    prediction_test_cnn2 = predict(x_test_2, args.SaveModlFile_2)
    accuracy, precision, recall, f1_score, MCC, AUC = evaluate(y_test, prediction_test_cnn2)
    print('Performance for CNN2:')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1_score)
    print('MCC: ', MCC)
    print('AUC: ', AUC)
    print('#########################')

    # ensemble
    prediction_test_vael = ensemble(x_test_1, x_test_2, args)
    accuracy, precision, recall, f1_score, MCC, AUC = evaluate(y_test, prediction_test_vael)
    print('Performance for VAEL:')
    print('Accuracy: ', accuracy)
    print('Precision: ', precision)
    print('Recall: ', recall)
    print('F1-score: ', f1_score)
    print('MCC: ', MCC)
    print('AUC: ', AUC)

    end = time.time()
    print(end - start)
