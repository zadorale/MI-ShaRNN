import os
import wget
import tarfile
#DATA SETS
directory =os.getcwd()

test = ('chinese-dog-breeds-4797219-hero-2a1e9c5ed2c54d00aef75b05c5db399c.jpg',"https://www.thesprucepets.com/thmb/7TDhfkK5CAKBWEaJfez6607J48Y=/1500x0/filters:no_upscale():max_bytes(150000):strip_icc()/chinese-dog-breeds-4797219-hero-2a1e9c5ed2c54d00aef75b05c5db399c.jpg",os.getcwd(),os.getcwd(),{})

GoogleSpeech = ("speech_commands_v0.01.tar.gz","http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz",os.getcwd()+"/GoogleSpeech/",os.getcwd()+"/GoogleSpeech/Raw/",
                {os.getcwd()+"/GoogleSpeech/",os.getcwd()+"/GoogleSpeech/Raw/"
                 ,os.getcwd()+"/GoogleSpeech/Extracted"})

GesturePod = ("dataTR_v1.tar.gz","https://www.microsoft.com/en-us/research/uploads/prod/2018/05/dataTR_v1.tar.gz",os.getcwd()+"/GesturePod/",os.getcwd()+"/GesturePod/",
                {os.getcwd()+"/GesturePod/",os.getcwd()+"/GesturePod/"})

HAR = ("UCI%20HAR%20Dataset.zip","https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",os.getcwd()+"/HAR/",os.getcwd()+"/HAR/",
                {os.getcwd()+"/HAR/",os.getcwd()+"/HAR/"})

DAS = ("data.zip","https://archive.ics.uci.edu/ml/machine-learning-databases/00256/data.zip",os.getcwd()+"/DAS/",os.getcwd()+"/DAS/",
                {os.getcwd()+"/DAS/",os.getcwd()+"/DAS/"})

#DATA LOADER FUNCTION
def bar_custom(current, total, width=80):
    print("Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total))
    print ("\033[A                             \033[A")

def download_file(tup):
    filename,url,directory,_,_ = tup
    if( filename in os.listdir(directory)):
        print("[0m[1;37;41m"+'FILE ALREADY IN DICERCTORY'+"[0m")
    else:
        print("[0m[1;30;43m"+'downloading: '+"[0m"+filename)
        print("[0m[1;30;43m"+'from: '+"[0m"+url)
        print("[0m[1;30;43m"+'into: '+"[0m"+directory)
        wget.download(url, bar=bar_custom,out = directory)

def extract_file(tup):
    filename,_,directory,location,_ = tup
    if(not filename in os.listdir(directory)):
        print("[0m[1;37;41m"+'NO SUCH FILE, NEED DOWNLOAD FIRST'+"[0m")
        return
    if('extracted_this.txt' in os.listdir(location)):
        print("[0m[1;37;41m"+'FILES ALREADY EXTRACTED'+"[0m")
        return
    file = tarfile.open(directory+filename)
    print("[0m[1;30;43m"+'extracting: '+"[0m"+filename)
    print("[0m[1;30;43m"+'into: '+"[0m"+location)
    file.extractall(location)
    file.close()
    f = open(location+'extracted_this.txt', 'w')
    f.write("extracted")
    f.close()
    print("[0m[1;30;42m"+'EXTRACTION COMPLETE'+"[0m")

def directory_create(tup):
    print("[0m[1;30;43m"+'creating directory for the experiment'+"[0m")
    _,_,_,_,direclis = tup
    for path in direclis:
        try:
            print("[0m[1;30;43m"+'creating path: '+"[0m"+path)
            os.mkdir(path)
        except OSError:
            print("[0m[1;30;43m"+'this path already exists: '+"[0m"+path)
    return

if __name__ == '__main__':
    """ Simple Test """
    download_file(test)




