""" File created to use and plot the results from the csv file """
import csv

import matplotlib.pyplot as plt
import numpy as np

plots = 'mean_std'

# opening the CSV file
with open('testjustinv1/mushra_backup15h11.csv', mode='r') as file:
    # reading the CSV file
    csvFile = csv.DictReader(file)

    participants = []
    stimuli = []

    threshold_ratings = []
    anchor_ratings = []
    low_ratings = []
    gain_ratings = []
    reference_ratings = []

    for line in csvFile:
        participant = line['name']
        stimulus = line['rating_stimulus']
        # print(line)


        if not participant in participants:
            participants.append(line['name'])
        if not stimulus in stimuli:
            stimuli.append(stimulus)

        '''Basic plot without looking at conditions'''
        if line['rating_stimulus']=='LOW':
            low_ratings.append(int(line['rating_score']))
        elif line['rating_stimulus']=='ANCHOR':
            anchor_ratings.append(int(line['rating_score']))
        elif line['rating_stimulus'] == 'reference':
            reference_ratings.append(int(line['rating_score']))
        elif line['rating_stimulus'] == 'THRESHOLD':
            threshold_ratings.append(int(line['rating_score']))
        elif line['rating_stimulus'] == 'GAIN':
            gain_ratings.append(int(line['rating_score']))

'''Means and std calculations'''
mean_gain = np.mean(gain_ratings)
mean_ref = np.mean(reference_ratings)
mean_anchor = np.mean(anchor_ratings)
mean_low = np.mean(low_ratings)
mean_threshold = np.mean(threshold_ratings)

std_gain = np.std(gain_ratings)
std_ref = np.std(reference_ratings)
std_anchor = np.std(anchor_ratings)
std_low = np.std(low_ratings)
std_threshold = np.std(threshold_ratings)

'''Plots'''
plt.figure()
if plots == 'raw':

    plt.plot(np.zeros(len(gain_ratings)), gain_ratings, '*', label='GAIN')
    plt.plot(np.ones(len(reference_ratings)), reference_ratings, '*', label='REF')
    plt.plot(np.ones(len(anchor_ratings))*2, anchor_ratings, '*', label='ANCHOR')
    plt.plot(np.ones(len(low_ratings))*3, low_ratings, '*', label='LOWPASS')
    plt.plot(np.ones(len(threshold_ratings))*4, threshold_ratings, '*', label='THRESHOLD')

    plt.legend()

elif plots == 'mean_std':

    x = [0,1,2,3,4]
    y = [mean_gain, mean_ref, mean_anchor, mean_low, mean_threshold]
    yerr = [std_gain, std_ref, std_anchor, std_low, std_threshold]
    plt.errorbar(x,y,yerr = yerr)
    plt.xticks(x,['GAIN','REF','ANCHOR','LOWPASS','THRESHOLD'])


    # plt.errorbar(np.zeros(1), mean_gain,  yerr=std_gain,  label='GAIN')
    # plt.errorbar(np.ones(1), mean_ref,  yerr=std_ref, label='REF')
    # plt.errorbar(np.ones(1)*2, mean_anchor, yerr=std_anchor, label='ANCHOR')
    # plt.errorbar(np.ones(1)*3, mean_low,  yerr=std_low, label='LOWPASS')
    # plt.errorbar(np.ones(1)*4, mean_threshold,  yerr=std_threshold, label='THRESHOLD')

plt.grid()
plt.show()


