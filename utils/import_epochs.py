# required packages
import os
import pandas as pd
import numpy as np
import h5py

def import_session_imgs(dir, subject, session, task, run):
    """ Returns pixel values and categories for all images in a session.

    params
    -----------------------
    dir : string
        indicates root directory (to save figure)
    subject : string
        indicates subject
    session : string
        session type (e.g. '_ses-nyuecog04_task-')
    task : string
        type of trials that were shown ('sixcatloctemporal' or 'sixcatlocisidiff')
    run : str
        run number

    returns
    -----------------------
    imgs : float
        array containing the pixel values for all images in one session
    cat : int
        array containing integers corresponding to image categories (e.g. 'FACES', 'BUILDINGS')

    """

    # extract pixel values
    path = dir + 'Documents/code/fMRI_adaptation/stimuli/' + subject+ session + task + '_acq-clinical_run-0' + run + '.mat'
    stim_file = h5py.File(path, 'r')
    stimulus = stim_file['stimulus']
    imgs = stimulus['images']
    imgs = np.array(imgs)
    imgs = imgs.T

    # extract categories
    cat = np.array(stimulus['cat'])

    return imgs, cat


def import_info(subject, dir):
    """ Returns info of the data.

    params
    -----------------------
    subject : string
        indicates subject
    dir : string
        indicates root directory (to save figure)

    returns
    -----------------------
    t : DataFrame
        contains timepoints
    events : array dim(n , m)
        contains the events (n) with additional info (m)
    channels : DataFrame dim(n, m)
        contains electrodes (n) with additional info (m).
    electrodes : DataFrame dim(n, m)
        contains electrodes (n) with additional info (m).

        """

    # read files
    t = pd.read_csv(dir+'Documents/ECoG_MATLAB/' + subject + '/t.txt', header=None)
    events = pd.read_csv(dir+'Documents/ECoG_MATLAB/' + subject + '/events.txt', header=0)
    channels = pd.read_csv(dir+'Documents/ECoG_MATLAB/' + subject + '/channels.txt', header=0)
    electrodes = pd.read_csv(dir+'Documents/ECoG_MATLAB/' + subject + '/electrodes.txt', header=0)

    return t, events, channels, electrodes

def import_epochs_b(subject, electrode_idx, dir):
    """ Returns broadband data.

    params
    -----------------------
    subject : string
        indicates subject
    electrode_idx : DataFrame dim(n) (optional)
        contains index number of specified electrode (n)
    dir : string
        indicates root directory (to save figure)

    returns
    -----------------------
    epochs_b : array dim(n, T)
        contains the broadband data (n) for each event

        """

    # os.chdir(dir+'Documents/ECoG_MATLAB/' + subject + '/epochs_b/')
    epochs_b = []
    if type(electrode_idx) == int:                                              # import data single electrode
        epochs_b = pd.read_csv(dir+'Documents/ECoG_MATLAB/' + subject + '/epochs_b/epochs_b_channel' + str(electrode_idx + 1) + '.txt', header=None)
    else:                                                                       # import data multiple electrodes
        for i in range(len(electrode_idx)):
            temp = pd.read_csv(dir+'Documents/ECoG_MATLAB/' + subject + '/epochs_b/epochs_b_channel' + str(electrode_idx[i] + 1) + '.txt', header=None)
            epochs_b.append(temp)

    return epochs_b

def import_epochs_v(subject, electrode_idx, dir):
    """ Returns raw data.

    params
    -----------------------
    subject : string
        indicates subject
    electrode_idx : DataFrame dim(1)
        contains index number of specified electrode
    dir : string
        indicates root directory (to save figure)

    returns
    -----------------------
    epochs_v : DataFrame dim(n, T)
        contains raw data for each event (n)(in voltage)

        """

    os.chdir(dir+'Documents/ECoG_MATLAB/' + subject + '/epochs_v/')
    epochs_v = []

    if type(electrode_idx) == int:
        epochs_v = pd.read_csv(dir+'Documents/ECoG_MATLAB/' + subject + '/epochs_v/epochs_v_channel' + str(electrode_idx + 1) + '.txt', header=None)
    else:
        for i in range(len(electrode_idx)):
            temp = pd.read_csv(dir+'Documents/ECoG_MATLAB/' + subject + '/epochs_v/epochs_v_channel' + str(i + 1) + '.txt', header=None)
            epochs_v.append(temp)

    return epochs_v
