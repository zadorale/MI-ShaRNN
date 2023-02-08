import numpy as np
import csv
import scipy.io
import os

def file_parse(path,filename):
    file = open(path+filename)
    csvreader = csv.reader(file)
    rows = []
    skip = False
    for row in csvreader:
        if(skip):
            rows.append(np.expand_dims(np.array(row),1))
        else:
            skip = True
    rows = np.concatenate(rows,1)
    return rows

def load_directory(direct):
    print("[0m[1;30;43m"+'loading directory: '+"[0m"+direct)
    filenames = os.listdir(direct)
    loadbar = 32
    filelists = list()
    le = len(filenames)
    i = 1
    jump =max(int(le/loadbar),1)
    printsize = max(int(loadbar/le),1)
    last = 0
    for filename in filenames:
        filelists.append(file_parse(direct,filename))
        if(int(loadbar*(i/le)-int(last)>=1)):
            print("[0m[1;30;46m"+("~"*(int(loadbar*(i/le))-int(last)))+"[0m", end = '')
            last = loadbar*(i/le)
        i = i+1;
    print("")
    return filelists

def get_full_dataset(direct):
    print("[0m[1;32;41m"+'data load start'+"[0m")
    train_dir = path+"/data_files/train_data/"
    test_dir = path+"/data_files/test_data/"
    train_data = load_directory(train_dir)
    test_data = load_directory(test_dir)
    print("[0m[1;31;42m"+'data load end'+"[0m")
    return train_data,test_data



if __name__ == '__main__':
    """ Simple Test """
    path = os.getcwd() + "/GesturePod"
    lis = get_full_dataset(path)