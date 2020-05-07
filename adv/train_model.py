from __future__ import absolute_import
from __future__ import print_function

import argparse
from util import get_data, get_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

def train(dataset='mnist', batch_size=128, epochs=50,pretrained=False):
    """
    Train one model with data augmentation: random padding+cropping and horizontal flip
    :param args: 
    :return: 
    """
    print('Data set: %s' % dataset)
    X_train, Y_train, X_test, Y_test = get_data(dataset)
    acc=0
    if pretrained==True:
        model = load_model('data/model_%s.h5' % dataset)
        score = model.evaluate(X_test, Y_test,verbose=0)
        acc=score[1]
    else:    
        model = get_model(dataset)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )
    
#     # training without data augmentation
#     model.fit(
#         X_train, Y_train,
#         epochs=epochs,
#         batch_size=batch_size,
#         shuffle=True,
#         verbose=1,
#         validation_data=(X_test, Y_test)
#     )

    # training with data augmentation
    # data augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    
    model.fit_generator(
        datagen.flow(X_train, Y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) / batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_test, Y_test))
    if pretrained==True:
        score = model.evaluate(X_test, Y_test,verbose=0)
        if score[1]>acc:
            print("model refined!")
            model.save('data/model_%s.h5' % dataset)
    else:
        model.save('data/model_%s.h5' % dataset)

def main(args):
    """
    Train model with data augmentation: random padding+cropping and horizontal flip
    :param args: 
    :return: 
    """
    assert args.dataset in ['mnist', 'cifar', 'svhn', 'all'], \
        "dataset parameter must be either 'mnist', 'cifar', 'svhn' or all"
    if args.dataset == 'all':
        for dataset in ['mnist', 'cifar', 'svhn']:
            train(dataset, args.batch_size, args.epochs)
    else:
        train(args.dataset, args.batch_size, args.epochs,args.pretrained)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar', 'svhn' or 'all'",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )
    parser.add_argument(
        '-p', '--pretrained',
        help="Where train from a pretrained model.",
        required=False, type=bool
    )
    parser.set_defaults(epochs=120)
    parser.set_defaults(batch_size=100)
    parser.set_defaults(pretrained=False)
    args = parser.parse_args()
    main(args)
