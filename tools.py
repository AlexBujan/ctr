import os
import time as timeLib
import numpy as np
import nest
import pylab as pl
import matplotlib.pyplot as plt


def get_senders_times_from_file(gdf):
    '''
    Loads neuron ids and spike times from a gdf file
    '''
    senders = times = []
    try:
        senders,times = pl.loadtxt(gdf,unpack=True)
    except:
        print '\nError while loading data. Checking for corrupted lines ...'
        fi = open(gdf,'r')
        lines = fi.readlines()
        fi.close()
        if len(lines)==0:
            print 'Gdf %s is empty!'%gdf
        else:
            os.rename(gdf,'%s-tmp'%gdf)
            fo = open(gdf,'a')
            for idx,line in enumerate(lines):
                if len(line.split('\t'))!=3:
                    print '\n\t-Corrupted line :\t%s'%line
                    continue
                try:
                    tmp_i = int(line.split('\t')[0])
                    tmp_f = float(line.split('\t')[1])
                except:
                    print '\n\t-Corrupted line :\t%s'%line
                    continue
                fo.write(line)
            fo.close()
            os.remove('%s-tmp'%gdf)
            try:
                senders,times = pl.loadtxt(gdf,unpack=True)
            except:
                senders = times = []
                print '\nFile %s could not be loaded!'%gdf
    return senders,times


def get_senders_times_potentials_from_file(dat):
    '''
    Loads neuron ids and measurement times and Vm values from a dat file
    '''
    senders = times = potentials =[]
    try:
        senders,times,potentials = pl.loadtxt(dat,unpack=True)
    except:
        print '\nError while loading data. Checking for corrupted lines ...'
        fi = open(dat,'r')
        lines = fi.readlines()
        fi.close()
        if len(lines)==0:
            print 'Dat %s is empty!'%dat
        else:
            os.rename(dat,'%s-tmp'%dat)
            fo = open(dat,'a')
            for idx,line in enumerate(lines):
                if len(line.split('\t'))!=4:
                    print '\n\t-Corrupted line :\t%s'%line
                    continue
                try:
                    tmp_i  = int(line.split('\t')[0])
                    tmp_f1 = float(line.split('\t')[1])
                    tmp_f2 = float(line.split('\t')[2])
                except:
                    print '\n\t-Corrupted line :\t%s'%line
                    continue
                fo.write(line)
            fo.close()
            os.remove('%s-tmp'%dat)
            try:
                senders,times,potentials = pl.loadtxt(dat,unpack=True)
            except:
                senders = times = potentials =[]
                print '\nFile %s could not be loaded!'%dat
    return senders,times,potentials

def get_senders_times_potentials_gex_gin_from_file(dat):
    '''
    Loads neuron ids and measurement times and Vm values from a dat file
    '''
    senders = times = potentials = gex = gin = []
    try:
        senders,times,potentials,gex,gin = pl.loadtxt(dat,unpack=True)
    except:
        print '\nError while loading data. Checking for corrupted lines ...'
        fi = open(dat,'r')
        lines = fi.readlines()
        fi.close()
        if len(lines)==0:
            print 'Dat %s is empty!'%dat
        else:
            os.rename(dat,'%s-tmp'%dat)
            fo = open(dat,'a')
            for idx,line in enumerate(lines):
                if len(line.split('\t'))!=6:
                    print '\n\t-Corrupted line :\t%s'%line
                    continue
                try:
                    tmp_i  = int(line.split('\t')[0])
                    tmp_f1 = float(line.split('\t')[1])
                    tmp_f2 = float(line.split('\t')[2])
                    tmp_f3 = float(line.split('\t')[3])
                    tmp_f4 = float(line.split('\t')[4])
                except:
                    print '\n\t-Corrupted line :\t%s'%line
                    continue
                fo.write(line)
            fo.close()
            os.remove('%s-tmp'%dat)
            try:
                senders,times,potentials,gex,gin = pl.loadtxt(dat,unpack=True)
            except:
                enders = times = potentials = gex = gin = []
                print '\nFile %s could not be loaded!'%dat
    return senders,times,potentials,gex,gin

def get_senders_times_potentials_gex_gin(voltmeter,verbose=False):
    '''
    Load senders, times, Vm G_ex and G_in values from a file or NEST device
    '''
    senders = times = potentials = gex = gin = []
    tic = timeLib.time()
    if pl.is_string_like(voltmeter):
        senders,times,potentials,gex,gin = get_senders_times_potentials_gex_gin_from_file(voltmeter)
    else:
        data = nest.GetStatus(voltmeter,'events')[0]
        senders,times,potentials,gex,gin = data['senders'],data['times'],data['V_m'],data['g_ex'],data['g_in']
    if type(senders)==list:
        senders,times,potentials,gex,gin = np.array(senders),np.array(times),np.array(potentials),np.array(gex),np.array(gin)
    gids = np.unique(senders)
    times = times[senders==gids[0]]
    tmp_vm = np.zeros((len(times),len(gids)))
    tmp_ge = np.zeros((len(times),len(gids)))
    tmp_gi = np.zeros((len(times),len(gids)))
    for idx1 in xrange(len(gids)):
        try:
            tmp_vm[:,idx1]= potentials[senders==gids[idx1]]
            tmp_ge[:,idx1]= gex[senders==gids[idx1]]
            tmp_gi[:,idx1]= gin[senders==gids[idx1]]
        except:
            pdb.set_trace()
    if verbose:
        print "\nMultimeter statistics:"
        print "-----------------------"
        print "\t* mean Vm          :\t%.2f\tmV"%np.mean(potentials)
        print "\t* s.d. Vm          :\t%.2f\tmV"%np.std(potentials)
        print "\t* skew Vm          :\t%.2f\tmV"%np.mean(skew(potentials,0))
        print "\t* mean gex         :\t%.2f\tnS"%np.mean(gex)
        print "\t* s.d. gex         :\t%.2f\tnS"%np.std(gex)
        print "\t* mean gin         :\t%.2f\tnS"%np.mean(gin)
        print "\t* s.d. gin         :\t%.2f\tnS"%np.std(gin)
        print "\nNeeded %.2f sec to load the potentials."%(timeLib.time()-tic)
    return gids,times,tmp_vm,tmp_ge,tmp_gi


def get_senders_times(spike_detector):
    '''
    Load senders and times from file or NEST device
    '''
    senders = times = []
    if pl.is_string_like(spike_detector):
        senders,times = get_senders_times_from_file(spike_detector)
    else:
        data = nest.GetStatus(spike_detector,'events')[0]
        senders,times = data['senders'],data['times']
    if type(senders)==list:
        senders,times = np.array(senders),np.array(times)
    return senders,times





