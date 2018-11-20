'''
yang song
ysong.sc@gmail.com
'''
import csv
import cv2
import numpy as np
import sys, argparse
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import time, calendar

def createLeNetModel():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,25), (0,0))))
    model.add(Conv2D(6, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(6, (5,5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))
    return model

def createNvidiaModel():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,25), (0,0))))
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def getTimestampStr():
    return str(calendar.timegm(time.gmtime()))

def plotHistory(history_object, outfilename):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    #plt.show()
    img_url = './images/loss_metrics_{}.png'.format(outfilename)
    plt.savefig(img_url)
    print("Save loss metrics to {}".format(img_url))

def main(arguments):
    parser = argparse.ArgumentParser(description='Model Trainer')
    parser.add_argument('-m', '--model', help="model name")
    parser.add_argument('-b', '--batchsize', help="batch size", type=int, default=60)
    parser.add_argument('-e', '--epochs', help="number of epochs", type=int, default=3)
    args = parser.parse_args(arguments)
    #print(args.model)
    outputfile = '{}_model_batch_{}_epoch_{}_{}'.format(args.model, str(args.batchsize), str(args.epochs), getTimestampStr())
    if args.model == 'lenet':
        print('Build LeNet Model ...')
    elif args.model == 'nvidia':
        print('Build Nvidia Model ...')
    else:
        print('Undefined model type. Only accepts lenet or nvidia')
        sys.exit(1)

    model = createLeNetModel() if args.model == 'lenet' else createNvidiaModel()

    # refactor codes used in the tutorial videos
    lines = []
    i = 0
    print('Read data from drive logs ...')
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            if i > 0: # bypass the csv header line
                lines.append(line)
            i+=1
    images = []
    measurements = []
    offsets = [0, -0.2, 0.2]
    for line in lines:
        for img_idx in range(3):
            source_path = line[img_idx]
            filename = source_path.split('/')[-1]
            current_path = './data/IMG/' +filename
            orig_image = cv2.imread(current_path)
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            images.append(image)
            measurement = float(line[3])
            # center if offset = 0, left if offset < 0, right if offset > 0
            measurements.append(measurement+offsets[img_idx])
            # Flipping
            images.append(cv2.flip(image,1))
            measurements.append(-1*(measurement+offsets[img_idx]))

    X_train = np.array(images)
    y_train = np.array(measurements)

    print("X_train.shape = ", X_train.shape)
    print("y_train.shape = ", y_train.shape)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print('Start training model ...')
    start_time = time.time()
    #model.fit(X_train_1, y_train_1, validation_split=0.2, shuffle=True, nb_epoch=7)
    history = model.fit(X_train, y_train, nb_epoch=args.epochs, batch_size = args.batchsize, validation_split=0.2, shuffle=True, verbose=1)
    print("Training took {0} seconds.".format(time.time() - start_time))
    model.save(outputfile+'.h5')
    print("Model saved as {}.h5".format(outputfile))
    plotHistory(history, outputfile)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
