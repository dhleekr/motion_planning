import os
import numpy as np
import cv2

class DataLoader:

    def __init__(self, shuffle=False):
        self.datadir = os.listdir('./dataset')
        print(len(self.datadir))
        self.labels = []
        self.imgs = np.empty((len(self.datadir), 500, 500, 3))
        self.shuffle = shuffle

    def create(self, ratio1=0.7, ratio2=0.9):
        print('create')
        for idx, img in enumerate(self.datadir):
            temp_label = self.datadir[idx].split('_')[0]
            self.labels.append(int(temp_label))

            img = cv2.imread('./dataset/'+img, cv2.IMREAD_COLOR)
            self.imgs[idx] = img
        
        data = self.imgs
        labels = self.labels

        if self.shuffle:
            shuffle = np.arange(len(self.datadir))
            np.random.shuffle(shuffle)

            data = self.imgs[shuffle]
            labels = self.labels[shuffle]

        split1 = int(ratio1*data.shape[0])
        split2 = int(ratio2*data.shape[0])

        [train_data, validation_data, test_data] = np.split(data, [split1, split2])
        [train_label, validation_label, test_label] = np.split(labels, [split1, split2])
        # print(test_label)

        np.savez('./results/train', data=train_data, label=train_label)
        np.savez('./results/validation', data=validation_data, label=validation_label)
        np.savez('./results/test', data=test_data, label=test_label)

    def load(self):
        print('load')
        train_set = np.load('./results/train.npz')
        validation_set = np.load('./results/validation.npz')
        test_set = np.load('./results/test.npz')

        return train_set, validation_set, test_set


if __name__ == "__main__":
    dl =DataLoader()
    # dl.create()
    dl.load()