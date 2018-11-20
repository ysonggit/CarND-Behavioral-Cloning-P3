'''
yang song
ysong.sc@gmail.com
'''
import csv
import cv2
import numpy as np
import sys, argparse
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
from keras.layers.pooling import MaxPooling2D

def createLeNetModel():
    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Convolution2D(6,5,5,activation='relu'))
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
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-m', '--model', help="model name")
    parser.add_argument('-o', '--outfile', help="output file")

    args = parser.parse_args(arguments)
    #print(args.model)
    #print(args.outfile)
    outputfile = args.outfile
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

    X_train = np.array(images)
    y_train = np.array(measurements)

    print("X_train.shape = ", X_train.shape)
    print("y_train.shape = ", y_train.shape)

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)
    model.save(outputfile)
    print("Model saved as %s" %outputfile)

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
