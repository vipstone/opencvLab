#
# coding=utf-8
# description:
#
import pickle


def load(file_name):
    with open(file_name, "rb") as fs:
        data = pickle.load(fs)
    return data


def save(data, file_name):
    with open(file_name, "wb") as fs:
        pickle.dump(data, fs)


if __name__ == '__main__':
    pass
