import numpy as np
import librosa
import librosa.display

import os
from os.path import isdir, join, dirname
from pathlib import Path
import re
import hashlib
from tqdm import tqdm

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name).encode('utf-8')
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result

def which_fold(filename, folds = 10):
    """Determines which fold the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
    filename: File path of the data sample.
    validation_percentage: How much of the data set to use for validation.
    testing_percentage: How much of the data set to use for testing.

    Returns:
    String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name).encode('utf-8')
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                      (MAX_NUM_WAVS_PER_CLASS + 1)) *
                     (100.0 / MAX_NUM_WAVS_PER_CLASS))


    for i in range(1,11):
        if percentage_hash<folds*i:
            result = i-1
            break

    return result

def get_folds(path_dataset,folds = 10):

    dirs = [f for f in os.listdir(path_dataset) if isdir(join(path_dataset, f))]
    dirs.sort()

    fold0 = []
    fold1 = []
    fold2 = []
    fold3 = []
    fold4 = []
    fold5 = []
    fold6 = []
    fold7 = []
    fold8 = []
    fold9 = []


    for direct in dirs:

        if direct == '_background_noise_':
            pass
        else:
            waves = [f for f in os.listdir(join(path_dataset,direct)) if (f.endswith('.wav'))]
            for wave in waves:
                #print(wave)
                this = which_fold(wave)
                #print(this)
                if this == 0:
                    fold0.append(direct+"/"+wave)
                elif this == 1:
                    fold1.append(direct+"/"+wave)
                elif this == 2:
                    fold2.append(direct+"/"+wave)
                elif this == 3:
                    fold3.append(direct+"/"+wave)
                elif this == 4:
                    fold4.append(direct+"/"+wave)
                elif this == 5:
                    fold5.append(direct+"/"+wave)
                elif this == 6:
                    fold6.append(direct+"/"+wave)
                elif this == 7:
                    fold7.append(direct+"/"+wave)
                elif this == 8:
                    fold8.append(direct+"/"+wave)
                elif this == 9:
                    fold9.append(direct+"/"+wave)


    return [fold0,fold1,fold2,fold3,fold4,fold5,fold5,fold7,fold8,fold9]


def get_word_dict():
    """Returns two dictionaries, one that maps the words to a numbered label,
    one that maps the numbers to its corresponding word
    """
    word_to_label = {'yes':0,'no':1,'up':2,'down':3,'left':4,'right':5,
                 'on':6,'off':7,'stop':8,'go':9,
                 'backward':10, 'bed':10,'bird':10,'cat':10,'dog':10,
                 'follow':10,'forward':10,'happy':10,'house':10,'learn':10,
                 'marvin':10,'sheila':10,'tree':10,'visual':10,'wow':10,
                 'zero':10,'one':10,'two':10,'three':10,'four':10,
                 'five':10,'six':10,'seven':10,'eight':10,'nine':10}

    label_to_word = dict([[v,k] for k,v in word_to_label.items()])
    label_to_word[10] = '<unk>'
    label_to_word[11] = '<silence>'

    return word_to_label,label_to_word


def get_word_dict_v2():
    """Returns two dictionaries, one that maps the words to a numbered label,
    one that maps the numbers to its corresponding word for version 2
    """
    word_to_label = {'yes':0,'no':1,'up':2,'down':3,'left':4,'right':5,
                     'on':6,'off':7,'stop':8,'go':9,
                     'zero':10,'one':11,'two':12,'three':13,'four':14,
                     'five':15,'six':16,'seven':17,'eight':18,'nine':19,
                     'backward':20, 'bed':20,'bird':20,'cat':20,'dog':20,
                     'follow':20,'forward':20,'happy':20,'house':20,'learn':20,
                     'marvin':20,'sheila':20,'tree':20,'visual':20,'wow':20
                     }

    label_to_word = dict([[v,k] for k,v in word_to_label.items()])
    label_to_word[20] = '<unk>'
    label_to_word[21] = '<silence>'

    return word_to_label,label_to_word


def get_word_dict_2words():
    """Returns two dictionaries, one that maps the words to a numbered label,
    one that maps the numbers to its corresponding word for the two words problem
    """
    word_to_label = {'yes':2,'no':2,'up':2,'down':2,'left':0,'right':1,
                     'on':2,'off':2,'stop':2,'go':2,
                     'zero':2,'one':2,'two':2,'three':2,'four':2,
                     'five':2,'six':2,'seven':2,'eight':2,'nine':2,
                     'backward':2, 'bed':2,'bird':2,'cat':2,'dog':2,
                     'follow':2,'forward':2,'happy':2,'house':2,'learn':2,
                     'marvin':2,'sheila':2,'tree':2,'visual':2,'wow':2
                     }

    label_to_word = dict([[v,k] for k,v in word_to_label.items()])
    label_to_word[2] = '<unk>'
    label_to_word[3] = '<silence>'

    return word_to_label,label_to_word


def get_dataset_partition(path_lib,val_percent,test_percent):
    """ Returns three lists contining the files for the training,
    validation and testing sets. This function makes sure that speakers
    don't appear accross sets.

    Parameters:
    path_lib: path to the directories containing the audio files.
    val_percent: percentage of the dataset to be included in the validation set.
    test_percent: percentage of the dataset to be included in the testing set.

    Returns:
    training_files, validation_files, testing_files: lists containing the filenames
    for each set.
    """
    dirs = [f for f in os.listdir(path_lib) if isdir(join(path_lib, f))]
    dirs.sort()

    training_files = []
    validation_files = []
    testing_files = []

    for direct in dirs:
        if direct == '_background_noise_':
            pass
        else:
            waves = [f for f in os.listdir(join(path_lib,direct)) if (f.endswith('.wav'))]
            for wave in waves:
                this = which_set(wave,val_percent,test_percent)
                if this == 'validation':
                    validation_files.append(direct+'/'+wave)
                elif this == 'training':
                    training_files.append(direct+'/'+wave)
                elif this == 'testing':
                    testing_files.append(direct+'/'+wave)
    return training_files,validation_files, testing_files

def reduce_examples(set_names,label,keep_prob,word_to_label):
    """Reduce the number of examples in a class

    Parameters:
    set_names: a list containing the names of the files to be imported,
    the format should be "[label]/[filename].wav get_dataset_partition()
    label: The class label to reduce, integer.
    keep_prob: The percentage to keep from the class

    Returns:
    A reduced list
    """
    to_del = []

    for i, filename in enumerate(set_names):

        direct = dirname(filename)

        if word_to_label[direct] == label:
            if np.random.choice([0,1],None,p = [keep_prob,1-keep_prob]):
                to_del.append(i)
    return np.delete(set_names,to_del)

def check_label_dist(set_names,word_to_label,label_to_word,labels):
    """Return a dictionary showing the number of examples for each class.
    """
    y = []
    dist = {}

    for i, filename in enumerate(set_names):
        label = word_to_label[dirname(filename)]
        y.append(label)

    y = np.array(y)

    for i in range(labels):
        frequency = np.sum(y==i)
        dist[label_to_word[i]] = frequency
    return dist

def export_partition_file(set_names,target_filename):
    """Export a .txt file from a list containing a partition of the set
    """
    with open(target_filename, 'w') as f:
        for item in set_names:
            f.write("%s\n" % item)

def read_list(path_dataset,filename):
    """Return a list with the filenames in the given file
    """
    with open(path_dataset+'/'+filename) as f:
        lines = f.read().splitlines()

    return lines

def load_data(file_names,sr,file_length,path_lib,word_to_label):
    """Loads data from a list obtained with get_dataset_partition()

    Parameters:
    file_names: a list containing the names of the files to be imported,
    the format should be "[label]/[filename].wav get_dataset_partition()
    returns this automatically.
    sr: The sample rate of the files
    file_length: The size for the files, if a loaded file is not the same
    as file_length it will be zero padded or truncated accordingly.
    path_lib: path to the directories containing the audio files.
    word_to_label: Dictonary that maps the words to the labels.Check
    get_word_dict()

    Returns: the examples x, and the labels y.
    """

    N = len(file_names)

    x = np.zeros((len(file_names),file_length))
    y = np.zeros(len(file_names))

    for i,file in enumerate(tqdm(file_names)):
        key = os.path.dirname(file)
        path = path_lib+'/'+file
        samples,_ = librosa.load(path,sr=sr)

        if len(samples)>file_length:
            samples=samples[:file_length]
        elif len(samples)<file_length:
            diff = file_length-len(samples)
            samples = np.pad(samples,(0,diff),'constant')

        x[i] = samples
        y[i] = word_to_label[key]
    return x,y


def remove_extra(x,y,target_size,label_num):
    """Reduces the examples of a class to target_size.

    Parameters:
    x: The set to reduce
    y: The labels of the set to reduce
    target_size: How many examples to keep
    label_num: Which label to reduce

    Returns:
    The reduced versions of x and y.
    """

    rem_size = np.sum(y==label_num)-target_size
    to_rem = np.random.choice(np.where(y==label_num)[0],rem_size,replace = False)
    x = np.delete(x,to_rem,axis=0)
    y = np.delete(y,to_rem,axis=0)

    return x,y

def files_in_dir(path_lib,direct):
    return [direct+'/'+f for f in os.listdir(path_lib+'/'+direct) if f.endswith('.wav')]

def load_direct(path_lib,direct,sr):
    """Load all the files on a directory

    Parameters:
    path_lib: The path to the dataset
    direct: The name of the directory to load
    sr: Sample Rate

    Returns:
    A list containing each of the waveforms (might be different sizes).
    """
    file_names = [f for f in os.listdir(path_lib+'/'+direct) if f.endswith('.wav')]
    waveforms = []

    for file in file_names:
        this_file,_ = librosa.load(path_lib+'/'+direct+'/'+file,sr = sr)

        if file =='pink_noise.wav' or file == 'white_noise.wav':
            this_file*= 0.2

        waveforms.append(this_file)

    return waveforms

def divide_file(wave,target_size,hop_length):
    """Divides a waveform into several smaller chunks.

    Parameters:
    wave: The waveform
    target_size: size of the chunks
    hop_length: distance between the beginning of chunks.

    Returns:
    A list of waveforms of size target_size taken from wave.
    """
    this_back_batch = []

    file_size = len(wave)
    frames = int((file_size-target_size)/hop_length)+1

    for i in range(frames):

        start_ind = i*hop_length
        stop_ind = start_ind+target_size
        this_wave = wave[start_ind:stop_ind]
        this_back_batch.append(this_wave)

    return np.array(this_back_batch)


def partition_directory(path_lib,direct,sr,file_len):
    """ Obtain partitions for all the examples in a directory into several examples
    with a fixed length.

    Parameters:
    path_lib: The path to the dataset.
    direct: The directory to collect the examples from
    sr: Sample Rate
    file_len: Length of the chunks

    Returns:
    List with partitioned waveforms
    """

    examples = np.zeros((1,file_len))

    files = load_direct(path_lib,direct,sr)

    for i, wave in enumerate(files):
        this_batch = divide_file(wave,file_len,file_len)
        examples = np.concatenate((this_batch,examples),axis = 0)

    examples = np.delete(examples,len(examples)-1,axis = 0)
    return examples



def append_examples(x,y,values,label):
    """Append a set of examples to a previously loded set.
    Parameters:
    x: examples of set to expand
    y: labels of the set to expand
    values: the values to append
    label: the label of the values to append
    """
    x = np.append(x,values = values,axis = 0)
    y = np.append(y,values = np.ones((len(values)))*label,axis = 0)

    return x,y
