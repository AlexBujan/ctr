import os,socket,pdb,getpass,glob
import time as timeLib
import numpy as np
import nest
import pylab as pl
import random as rand
import cPickle as cp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import skew


def findTauExp(w,n,dtau=1e4):
    dist = 0
    tmp_tau = np.linspace(5e-6,5e-2,dtau)
    for idx1 in xrange(len(tmp_tau)):
        f = np.exp(-tmp_tau[idx1]*n)
        pdf = f/np.sum(f,0)
        tmp_w = ((np.sum(n**2*pdf,0)/np.sum(n*pdf,0))-1)*1./(len(n)-1)
        if idx1==0:
            dist = np.abs(w-tmp_w)
        else:
            tmp_dist = np.abs(w-tmp_w)
            if tmp_dist>dist:
                break
            else:
                dist=tmp_dist
    return tmp_tau[idx1]

def findEpsilonBinom(w,eta,N,dw=1e4):
    dist = 0
    tmp_epsilon = np.linspace(w,1.,dw)
    for idx1 in xrange(len(tmp_epsilon)):
        tmp_w = (N*tmp_epsilon[idx1]**2*(1-eta))/(N*tmp_epsilon[idx1]*(1-eta)+eta)
        if idx1==0:
            dist = np.abs(w-tmp_w)
        else:
            tmp_dist = np.abs(w-tmp_w)
            if tmp_dist>dist:
                break
            else:
                dist=tmp_dist
    return tmp_epsilon[idx1]

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

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


def merge(path,ext,n_threads):
    '''
    Merges gdf or dat files from different threads
    '''
    devices = np.sort(glob.glob(path+'/*-[0-%i].%s'%((n_threads-1),ext)))
    print "Merging files %s ..."%path
    for dev in devices:
        tmp_name = dev.split('-')
        dev_name = tmp_name[0]
        for i in np.arange(1,len(tmp_name[:-2])):
            dev_name+='-%s'%(tmp_name[i])
        dev_number = tmp_name[-1].split('.')[0]
        if dev_number=='m':
            continue
        if dev_number=='0':
            mode='w'
        else:
            mode='a'
        file_merged = open(dev_name+'-m.%s'%ext,mode)
        file_part   = open(dev,'r')
        data = file_part.readlines()
        file_part.close()
        file_merged.writelines(data)
        file_merged.close()


def get_senders_times_potentials(voltmeter,verbose=False):
    '''
    Load senders, times and Vm values from a file or NEST device
    '''
    senders = times = potentials = []
    tic = timeLib.time()
    if pl.is_string_like(voltmeter):
        senders,times,potentials = get_senders_times_potentials_from_file(voltmeter)
    else:
        data = nest.GetStatus(voltmeter,'events')[0]
        senders,times,potentials = data['senders'],data['times'],data['V_m']
    if type(senders)==list:
        senders,times,potentials = np.array(senders),np.array(times),np.array(potentials)
#    senders = np.unique(senders)
#    n = len(senders)
#    times = times[::n]
#    potentials = np.reshape(potentials,[len(potentials)/n,n])
    gids = np.unique(senders)
    times = times[senders==gids[0]]
    tmp_vm = np.zeros((len(times),len(gids)))
    for idx1 in xrange(len(gids)):
        tmp_vm[:,idx1]= potentials[senders==gids[idx1]]
    if verbose:
        print "\nMembrane statistics:"
        print "--------------------"
        print "\t* mean Vm      :\t%.2f mV"%np.mean(potentials)
        print "\t* s.d. Vm      :\t%.2f mV"%np.std(potentials)
        print "\t* skew Vm      :\t%.2f mV"%np.mean(skew(potentials,0))
        print "\nNeeded %.2f sec to load the potentials."%(timeLib.time()-tic)
#    return senders,times,potentials
    return gids,times,tmp_vm

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
#    senders = np.unique(senders)
#    n = len(senders)
#    times = times[::n]
#    potentials  = np.reshape(potentials ,[len(potentials)/n,n])
#    gex         = np.reshape(gex        ,[len(gex)/n,n])
#    gin         = np.reshape(gin        ,[len(gin)/n,n])
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
#    return senders,times,potentials,gex,gin
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



def get_spiking_stats(senders,times,wup_time,sim_time,n_pairs=1000,binSize=200.,verbose=False):
    '''
    Computes single neuron spiking statistics from a gdf file or a spike detector
    '''
    fr = cv = cc = ff = []
    if len(senders)!=0:
        ids = np.unique(senders)
        fr = np.zeros(len(ids))
        cv = np.zeros(len(ids))
        ff = np.zeros(len(ids))
        cc = np.zeros(n_pairs)
        count_bins = np.arange(wup_time,wup_time+sim_time+binSize,binSize)
        if len(ids)**2-len(ids)==n_pairs:
            count=0
            for s in ids:
                for ss in ids:
                    if s==ss:
                        continue
                    psth1 = np.histogram(times[senders==s],count_bins)[0]
                    psth2 = np.histogram(times[senders==ss],count_bins)[0]
                    cc[count] = np.corrcoef(psth1,psth2)[0][1]
                    count+=1
        else:
            for j in xrange(n_pairs):
                gotit = False
                while not gotit:
                    sp = rand.sample(ids,2)
                    if sp[0]==sp[1]:
                        continue
                    gotit = True
                psth1 = np.histogram(times[senders==sp[0]],count_bins)[0]
                psth2 = np.histogram(times[senders==sp[1]],count_bins)[0]
                cc[j] = np.corrcoef(psth1,psth2)[0][1]
        for i,n in enumerate(ids):
            spikes = times[senders==n]
            fr[i] = len(spikes)*1e3/sim_time
            psth = np.histogram(spikes,count_bins)[0]
            ff[i] = np.var(psth)*1./np.mean(psth)
            if len(spikes)>1:
                isis = np.diff(spikes)
                cv[i] = np.std(isis)*1./np.mean(isis)
            else:
                cv[i] = np.nan
        cv = np.ma.masked_array(cv,np.isnan(cv))
        if verbose:
            print "\nSpiking statistics:"
            print "-------------------"
            print "\t* mean FR  :\t%.2f Hz"%np.mean(fr)
            print "\t* mean CV  :\t%.2f"%np.mean(cv)
            print "\t* mean CC  :\t%.2f"%np.mean(cc)
            print "\t* mean FF  :\t%.2f"%np.mean(ff)
    else:
        print 'Error while loading the spike data. Maybe there were no spikes.'
    return fr,cv,cc,ff


def compute_pop_power_spectrum(spike_train,sim_time,max_freq=1e3,df=1.,verbose=True):
    """
    Computes the spike train's power spectrum
    """
    st_T = spike_train-sim_time/2.
    j = np.complex(0,1)
    powerspect_freqs = np.arange(0.,max_freq+df,df)
    ps = np.zeros_like(powerspect_freqs)
    for i,f in enumerate(powerspect_freqs):
        om=2*np.pi*f
        contribs = np.exp(-j*om*st_T)
        ps[i] = np.abs(np.sum(contribs))**2/sim_time
    if verbose:
        print "\nFrequency analysis:"
        print "-------------------"
        print "\t* max. freq. :\t%.2f Hz"%powerspect_freqs[np.where(ps==np.max(ps[1:]))] 
        print "\t* max. power :\t%.2f*10^6"%(np.max(ps[1:])*1./1e6)
    return ps


def get_crosscovariance(xcorr,sim_time,lag,tau_max):
    xhist = nest.GetStatus(xcorr ,'count_histogram')[0]
    rate0 = nest.GetStatus(xcorr ,'n_events')[0][0]/sim_time
    rate1 = nest.GetStatus(xcorr ,'n_events')[0][1]/sim_time
    x=xhist/(sim_time*lag)-rate0*rate1
    tau_vector = np.arange(-tau_max,tau_max+lag,lag)
    return x,tau_vector

