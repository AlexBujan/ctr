import os,socket,pdb,time,getpass,random,glob
import pylab as pl
import numpy as np
import nest
import cPickle as cp
from optparse import OptionParser
import matplotlib.pyplot as plt

import tools as to
reload(to)
import nest_functions as nf
reload(nf)


"""
##########
Simulation
##########
"""
def simulate():
    global pulse_times
    #configure and reset Nest kernel
    nf.reset_kernel(nest,dic=nest_dic,n_threads=n_threads)

    if verbose:
        print '\nSimulating ctr_net with:'
        print '-------------------------'

        print '\t*delay within      :\t%.1f'%del_within
        print '\t*delay between     :\t%.1f'%del_between
        print '\t*Exc syn. tau      :\t%.1f'%neuron_param['tau_syn_ex']
        print '\t*Inh syn. tau      :\t%.1f'%neuron_param['tau_syn_in']
        print '\t*weight j_exc_exc  :\t%.1f'%j_exc_exc
        print '\t*weight j_exc_inh  :\t%.1f'%j_exc_inh
        print '\t*weight j_inh_exc  :\t%.1f'%j_inh_exc
        print '\t*weight j_inh_inh  :\t%.1f'%j_inh_inh
        print '\t*weight j_ext_high :\t%.1f'%j_ext_high
        print '\t*weight j_ext_low  :\t%.1f'%j_ext_low
        print '\t*Poi. backg. E     :\t%.1f'%poi_rate_bkg_exc
        print '\t*Poi. backg. I     :\t%.1f'%poi_rate_bkg_inh
        print '\t*Num. of layers    :\t%i'%n_layers
        print '\t*Num. of E neurons :\t%i'%Nexc
        print '\t*Ext inh-inh       :\t%.1f'%poi_rate_bkg_II
        print '\t*weight j_ext_II   :\t%.1f'%j_ext_II
        print '\t*Interlayer conn.  :\t%i'%LL_conn
        print '\t*E drive           :\t%s\n'%str(E_drive)

        if not reset_delays:
            print '\n## Warning: delays have not been reset and they may be wrong! ##'

        print '\nSimulation parameters:'
        print '----------------------'
        print '\t*sim_time              :\t%.0f'%sim_time
        print '\t*stim_start            :\t%.0f'%stim_start
        print '\t*stim_stop             :\t%.0f'%stim_stop
        print '\t*record_Vm             :\t%s'%str(record_Vm)

    #synapse models
    nest.CopyModel("static_synapse","EI",{"weight":j_exc_inh,"delay":del_exc_inh})
    nest.CopyModel("static_synapse","EE",{"weight":j_exc_exc,"delay":del_exc_exc})
    nest.CopyModel("static_synapse","IE",{"weight":j_inh_exc,"delay":del_inh_exc})
    nest.CopyModel("static_synapse","II",{"weight":j_inh_inh,"delay":del_inh_inh})
    nest.CopyModel("static_synapse","CH",{"weight":j_chain,"delay":del_chain})
    nest.CopyModel("static_synapse","PP",{"weight":j_pulse_packet,"delay":del_pp})

    #backgtround input
    pg_gen_inh = nest.Create('poisson_generator',params={'rate':poi_rate_bkg_inh*1e3})
    if pulses:
        pg_gen_exc = nest.Create('poisson_generator',params={'rate':poi_rate_bkg_exc*1e3})
    else:
        if n_layers>1:
            pg_gen_exc = nest.Create('poisson_generator',params={'rate':poi_rate_bkg_exc*1e3})
        pg_gen_exc_pre  = nest.Create('poisson_generator',params={'rate':poi_rate_bkg_exc*1e3,'stop':stim_start})
        pg_gen_exc_post = nest.Create('poisson_generator',params={'rate':poi_rate_bkg_exc*1e3,'start':stim_stop})

    #reset recorder parameters
    if record_toFile:
        recorder_params.update({'to_file':True})

    sd_exc_list     = []
    sd_inh_list     = []
    sd_group_list   = []
    layers_ex   = []
    layers_in   = []
    group       = []
    #reset threshold for spiking neurons
    if neuron_param['V_th']>0:
        neuron_param.update({'V_th':-54.})
    #create neurons and connect
    for i in xrange(n_layers):
        #create EI layers
        layers_ex.append(nest.Create(neuron_model,Nexc,neuron_param))
        layers_in.append(nest.Create(neuron_model,Ninh,neuron_param))
        #select group neurons
        group.append(layers_ex[i][:group_size])
        #connect external drive
        w = np.random.uniform(j_ext_low,j_ext_high,Nexc+Ninh)
        nest.DivergentConnect(pg_gen_inh,layers_in[i],w[Nexc:].tolist(),Ninh*[1.])
        if i==0:
            if pulses:
                nest.DivergentConnect(pg_gen_exc,layers_ex[0][group_size:Nexc],w[group_size:Nexc].tolist(),(Nexc-group_size)*[1.])
            else:
                nest.DivergentConnect(pg_gen_exc_pre,layers_ex[0][group_size:Nexc],w[group_size:Nexc].tolist(),(Nexc-group_size)*[1.])
                nest.DivergentConnect(pg_gen_exc_post,layers_ex[0][group_size:Nexc],w[group_size:Nexc].tolist(),(Nexc-group_size)*[1.])
        else:
            nest.DivergentConnect(pg_gen_exc,layers_ex[i],w[:Nexc].tolist(),Nexc*[1.])

        #create feedforward connections
        if i!=0:
            if LL_conn==0:
                nest.RandomConvergentConnect(group[i-1],group[i],syn_chain,model='CH')
            elif LL_conn==1:
                nest.RandomConvergentConnect(layers_ex[i-1],group[i],100,model='CH')
            elif LL_conn==2:
                nest.RandomConvergentConnect(layers_ex[i-1],layers_ex[i],30,model='CH')

        #create recurrent connections within each layer
        nest.RandomDivergentConnect(layers_ex[i],layers_ex[i],syn_exc_exc,model='EE')
        nest.RandomDivergentConnect(layers_ex[i],layers_in[i],syn_exc_inh,model='EI')
        nest.RandomDivergentConnect(layers_in[i],layers_ex[i],syn_inh_exc,model='IE')
        nest.RandomDivergentConnect(layers_in[i],layers_in[i],syn_inh_inh,model='II')

        #create and connect inhibitory input to I neruons 
        if poi_rate_bkg_II>0.:
            pg_gen_II = nest.Create('poisson_generator',params={'rate':poi_rate_bkg_II})
            w_II = np.random.uniform(j_ext_II+.5,j_ext_II-.5,Ninh)
            nest.DivergentConnect(pg_gen_II,layers_in[i],w_II.tolist(),Ninh*[1.])

        #create and connect spike detectors
        if recorder_params.has_key('record_from'):
            recorder_params.pop('record_from')
        if record_toFile:
            recorder_params.update({'label':recorder_label+'_grp_layer_%i_Sd'%(i+1)})
        sd_group_list.append(nest.Create("spike_detector",params=recorder_params))
        nest.ConvergentConnect(group[i],sd_group_list[i])
        if record_toFile:
            recorder_params.update({'label':recorder_label+'_exc_layer_%i_Sd'%(i+1)})
        sd_exc_list.append(nest.Create("spike_detector",params=recorder_params))
        nest.ConvergentConnect(layers_ex[i],sd_exc_list[i])
        if record_toFile:
            recorder_params.update({'label':recorder_label+'_inh_layer_%i_Sd'%(i+1)})
        sd_inh_list.append(nest.Create("spike_detector",params=recorder_params))
        nest.ConvergentConnect(layers_in[i],sd_inh_list[i])

    #create and connect pulse packet generator
    if pulses:
        if len(pulse_times)==0:
            pulse_times = np.arange(stim_start,stim_stop,interval)
            pulse_times+=np.random.uniform(-jitter,jitter,len(pulse_times))
        if verbose:
            print '\nPulse Packet generator stimulation:'
            print '-----------------------------------'
            print '\t*interval          :\t%.1f'%interval
            print '\t*activity          :\t%i'%activity
            print '\t*sigma             :\t%.1f'%sigma
            print '\t*jitter            :\t%.1f'%jitter
            print '\t*number of pulses  :\t%i'%len(pulse_times)
        pp = nest.Create('pulsepacket_generator',params={'activity':activity,'sdev':sigma,'pulse_times':pulse_times})
    else:
        ac_params = {'start': stim_start,
                     'stop' : stim_stop,
                     'dc'   : dc*1e3,
                     'ac'   : ac*1e3,
                     'freq' : freq,
                     'phi'  : phi}
        pp = nest.Create('sinusoidal_poisson_generator',params=ac_params)
        if verbose:
            print '\nSinusoidal Poisson generator stimulation:'
            print '----------------------------------------'
            print '\t*dc              :\t%.1f\tkHz'%dc
            print '\t*ac              :\t%.1f\tkHz'%ac
            print '\t*freq            :\t%.1f\tHz'%freq
            print '\t*phi             :\t%.1f'%phi

        if E_drive:
            #All E_{0} neurons get the constant drive not only L_{0} neurons. Here we first connect the poisson
            #generator to the non-L_{0} neurons, the connections to L_{0} neurons go through the parrot (see below) 
            w = np.random.uniform(j_ext_low,j_ext_high,Nexc-group_size)
            nest.DivergentConnect(pp,layers_ex[0][group_size:Nexc],w.tolist(),(Nexc-group_size)*[1.])

    #connect parrot input and connect to first layer
    pa = nest.Create('parrot_neuron',group_size*2)
    if pulses:
        nest.DivergentConnect(pg_gen_exc,pa[:group_size],1.,.5)
    else:
        nest.DivergentConnect(pg_gen_exc_pre,pa[:group_size],1.,.5)
        nest.DivergentConnect(pg_gen_exc_post,pa[:group_size],1.,.5)
    nest.DivergentConnect(pp,pa[group_size:],1.,.5)
    nest.Connect(pa[group_size:],group[0],model='PP')
    w = np.random.uniform(j_ext_low,j_ext_high,group_size)
    nest.Connect(pa[:group_size],group[0],w.tolist(),group_size*[.5])

    #record input
    if record_toFile:
        recorder_params.update({'label':recorder_label+'_inp_Sd'})
    input_sd = nest.Create("spike_detector",params=recorder_params)
    nest.ConvergentConnect(pa,input_sd,1.,.5)

    #record membrane potential
    if record_Vm:
        recorder_params.update({'record_from':['V_m','g_ex','g_in'],'label':recorder_label+'_exc_Vm'})
        mm_exc = nest.Create('multimeter',params=recorder_params)
        recorder_params.update({'label':recorder_label+'_inh_Vm'})
        mm_inh = nest.Create('multimeter',params=recorder_params)
        recorder_params.update({'label':recorder_label+'_grp_Vm'})
        mm_grp = nest.Create('multimeter',params=recorder_params)

        nest.DivergentConnect(mm_exc,layers_ex[0][:100])
        nest.DivergentConnect(mm_grp,group[0][:100])
        nest.DivergentConnect(mm_inh,layers_in[0][:100])

    #simulate
    nest.Simulate(wup_time+sim_time)

    #network activity
    mean_rate_exc =  np.zeros(n_layers)
    mean_rate_inh =  np.zeros(n_layers)
    for i in xrange(n_layers):
        mean_rate_exc[i] = len(nest.GetStatus(sd_exc_list[i],'events')[0]['times'])*1e3/Nexc/sim_time
        mean_rate_inh[i] = len(nest.GetStatus(sd_inh_list[i],'events')[0]['times'])*1e3/Ninh/sim_time
    if verbose:
        print '\nNetwork activity:'
        print '-----------------'
        print '\t*Mean E mean rate                  :\t%.2f'%np.mean(mean_rate_exc)
        print '\t*Mean I mean rate                  :\t%.2f\n'%np.mean(mean_rate_inh)
        if record_Vm:
            print '\t*Mean exc. conductance (E)         :\t%.2f'%np.mean(nest.GetStatus(mm_exc,'events')[0]["g_ex"])
            print '\t*Mean exc. conductance (I)         :\t%.2f'%np.mean(nest.GetStatus(mm_inh,'events')[0]["g_ex"])
            print '\t*Mean inh. conductance (E)         :\t%.2f'%np.mean(nest.GetStatus(mm_exc,'events')[0]["g_in"])
            print '\t*Mean inh. conductance (I)         :\t%.2f\n'%np.mean(nest.GetStatus(mm_inh,'events')[0]["g_in"])
            print '\t*Mean Vm (E)                       :\t%.2f'%np.mean(nest.GetStatus(mm_exc,'events')[0]["V_m"])
            print '\t*Mean Vm (I)                       :\t%.2f'%np.mean(nest.GetStatus(mm_inh,'events')[0]["V_m"])
            print '\t*S.d. Vm (E)                       :\t%.2f'%np.std(nest.GetStatus(mm_exc,'events')[0]["V_m"])
            print '\t*S.d. Vm (I)                       :\t%.2f'%np.std(nest.GetStatus(mm_inh,'events')[0]["V_m"])

    #collect recorders and return
    if return_recorders:
        recorders = []
        sd_list = [sd_exc_list,sd_group_list,sd_inh_list,input_sd]
        recorders.append(sd_list)
        if record_Vm:
            mm_list = [mm_exc,mm_grp,mm_inh]
            recorders.append(mm_list)
        return recorders

"""
################
Helper functions
################
"""
def set_delays():
    global del_exc_exc,del_inh_inh,del_exc_inh,del_inh_exc,reset_delays
    del_exc_exc = del_within
    del_inh_inh = del_within
    del_exc_inh = del_between
    del_inh_exc = del_between
    reset_delays = True
def set_weights():
    global j_inh_inh,j_inh_exc,j_exc_inh,j_exc_exc
    j_inh_inh = jinh*g
    j_inh_exc = jinh*g
    j_exc_inh = jexc*2
    j_exc_exc = jexc

"""
###########
Experiments
###########
"""

def figure3a():
    global record_Vm,record_toFile,return_recorders,recorder_params
    global pulses,activity,sigma,interval,jitter,pulse_times
    '''
    parameters
    '''
    #stimulus parameters
    pulses          = True
    activity        = 20
    sigma           = 3.
    interval        = 45.
    jitter          = 0.
    pulse_times = []
    #simulation parameters
    record_toFile   = False
    return_recorders= True
    record_Vm       = True
    #recorders
    recorder_params.update({'to_file':record_toFile})
    '''
    simulate
    '''
    recorders   = simulate()
    devices     = [recorders[0][0][0],recorders[0][1][0],recorders[0][2][0],recorders[0][-1]]
    plot_act(sd_list=devices,vm_list =[recorders[1][0]],n_pop=[Nexc,group_size,Ninh],wupTime=stim_start-250.,simTime=stim_stop+250.,binSize=10.,figureName='figure3a')


def figureS6a():
    global del_within,del_between,record_Vm,record_toFile,return_recorders,recorder_params
    global pulses,ac,phi,freq,dc,E_drive
    '''
    parameters
    '''
    #stimulus parameters
    pulses          = False
    ac              = 1.
    dc              = 1.
    E_drive          = True
    phi             = 0.
    freq            = 15.
    #network parameters
    del_within      = 1.
    del_between     = 2.5
    set_delays()
    #simulation parameters
    return_recorders= True
    record_toFile   = False
    record_Vm       = True
    recorder_params.update({'to_file':record_toFile})
    '''
    simulate
    '''
    recorders = simulate()
    devices = [recorders[0][0][0],recorders[0][1][0],recorders[0][2][0],recorders[0][-1]]
    plot_act(sd_list=devices,vm_list =[recorders[1][0]],n_pop=[Nexc,group_size,Ninh],wupTime=stim_start-250.,simTime=stim_stop+250.,binSize=10.,figureName='figureS6a')


def figure4a():
    global del_within,del_between,record_Vm,record_toFile,return_recorders,recorder_params,n_layers
    global pulses,activity,sigma,interval,jitter,pulse_times
    '''
    parameters
    '''
    #stimulus parameters
    pulses          = True
    activity        = 20
    sigma           = 0.
    jitter          = 0.
    pulse_times = []
    #network parameters
    del_within      = 2.
    del_between     = 5.
    set_delays()
    n_layers  = 5
    #simulation parameters
    record_toFile   = False
    recorder_params.update({'to_file':record_toFile})
    return_recorders= True
    record_Vm       = False
    '''
    simulate
    '''
    data55 = {'exc':{'times':[],'senders':[]},'grp':{'times':[],'senders':[]},'inh':{'times':[],'senders':[]},'input':{'times':[],'senders':[]}}
    data35 = {'exc':{'times':[],'senders':[]},'grp':{'times':[],'senders':[]},'inh':{'times':[],'senders':[]},'input':{'times':[],'senders':[]}}
    data1  = {'exc':{'times':[],'senders':[]},'grp':{'times':[],'senders':[]},'inh':{'times':[],'senders':[]},'input':{'times':[],'senders':[]}}
    data = {'55':data55,'35':data35,'1e3':data1}
    for k0 in data.keys():
        if recorder_params.has_key('record_from'):
            recorder_params.pop('record_from')
        pulse_times = []
        interval=float(k0)
        r = simulate()
        for i in xrange(n_layers):
            for j,k1 in enumerate(['exc','grp','inh']):
                for k2 in ['times','senders']:
                    data[k0][k1][k2].append(nest.GetStatus(r[0][j][i],'events')[0][k2])
        for k2 in ['times','senders']:
            data[k0]['input'][k2].append(nest.GetStatus(r[0][3],'events')[0][k2])
    f = open('figure4a.npy','w')
    cp.dump(data,f)
    f.close()
    '''
    plot
    '''
    col=['r','b','k']
    keys0 = ['55','35','1e3']
    bins = np.arange(1000.,3000.+10.,10.)
    fig = plt.figure('figure4a',figsize=(3,6))
    plt.clf()
    raster_pos  = [[.1,.165,.875,.2],[.1,.47,.875,.2],[.1,.775,.875,.2]]
    psth_pos    = [[.1,.075,.875,.075],[.1,.38,.875,.075],[.1,.685,.875,.075]]
    axList = [[],[]]
    for h,k0 in enumerate(keys0):
        axList[0].append(fig.add_axes(raster_pos[h]))
        axList[1].append(fig.add_axes(psth_pos[h]))
    for h,k0 in enumerate(keys0):
        for j,k1 in enumerate(data[k0].keys()[1:]):
            for i in xrange(5):
                if i!=1 and i!=3:
                    axList[0][h].add_patch(plt.Rectangle((1400,1500*i),1400,1500,facecolor=".8",edgecolor='.8'))
                try:
                    for sdr in np.unique(data[k0][k1]['senders'][i])[::10]:
                        spikes = data[k0][k1]['times'][i][data[k0][k1]['senders'][i]==sdr]
                        axList[0][h].plot(spikes,len(spikes)*[sdr],'.',markersize=1.35,color=col[j])
                except:
                    continue
        axList[0][h].set_yticks(np.arange(0,7501,1500))
        axList[0][h].set_ylim(0,7500)
        axList[0][h].set_yticklabels([])
        axList[0][h].set_xlim(1400,2800)
        axList[0][h].set_xticklabels([])
        psth = np.histogram(data[k0]['input']['times'],bins)[0]*1e3/10./300.
        axList[1][h].plot(bins[:-1],psth,drawstyle='steps-mid',linewidth=1,color='b')
        if k0=='55':
            axList[0][h].set_ylabel('Layers',fontsize=12,labelpad=5)
            axList[1][h].set_yticks(np.arange(0,3001,1500))
            axList[1][h].set_ylabel('kHz',fontsize=12,labelpad=3)
            axList[1][h].set_yticklabels([0,'',3],fontsize=8,stretch='ultra-condensed')
            axList[1][h].set_xticks(np.arange(1400,2801,700))
            axList[1][h].set_xticklabels([1.4,2.1,2.8],fontsize=8,stretch='ultra-condensed')
            axList[1][h].set_xlabel('Time [s]',fontsize=12,labelpad=0)
        else:
            axList[1][h].set_yticks(np.arange(0,3001,1500))
            axList[1][h].set_xticklabels([])
            axList[1][h].set_yticklabels([])
        axList[1][h].set_ylim(0,3000)
        axList[1][h].set_xlim(1400,2800)
    plt.show()

def figure5b():
    global del_within,del_between,record_Vm,record_toFile,return_recorders,recorder_params,n_layers
    global pulses,ac,phi,freq,dc,E_drive
    '''
    parameters
    '''
    #stimulus parameters
    pulses          = False
    ac              = 0.
    dc              = 1.8
    phi             = 0.
    freq            = 0.
    E_drive         = True
    #network parameters
    del_within      = 2.
    del_between     = 5.
    set_delays()
    n_layers  = 5
    #simulation parameters
    record_toFile   = False
    recorder_params.update({'to_file':record_toFile})
    return_recorders= True
    record_Vm       = False
    '''
    simulate
    '''
    r = simulate()
    '''
    plot
    '''
    bins = np.arange(1000.,3000.+10.,10.)
    col=['r','b','k']
    ax = []
    fig = plt.figure('figure5b',figsize=(5,3.5))
    plt.clf()
    ax.append(fig.add_axes([.1,.4,.85,.55]))
    ax.append(fig.add_axes([.1,.15,.85,.2]))

    data  = {'exc':{'times':[],'senders':[]},'grp':{'times':[],'senders':[]},'inh':{'times':[],'senders':[]},'input':{'times':[],'senders':[]}}
    for i in xrange(n_layers):
        for j,k1 in enumerate(['exc','grp','inh']):
            for k2 in ['times','senders']:
                data[k1][k2].append(nest.GetStatus(r[0][j][i],'events')[0][k2])
    for k2 in ['times','senders']:
        data['input'][k2].append(nest.GetStatus(r[0][3],'events')[0][k2])

    for j,k1 in enumerate(data.keys()[1:]):
        for i in xrange(5):
            if i!=1 and i!=3:
                ax[0].add_patch(plt.Rectangle((1400,1500*i),1400,1500,facecolor=".8",edgecolor='.8'))
            for sdr in np.unique(data[k1]['senders'][i])[::10]:
                spikes = data[k1]['times'][i][data[k1]['senders'][i]==sdr]
                ax[0].plot(spikes,len(spikes)*[sdr],'.',markersize=3,color=col[j])

    ax[0].set_yticks(np.arange(0,7501,1500))
    ax[0].set_ylim(0,7500)
    ax[0].set_yticklabels([])
    ax[0].set_xlim(1400,2800)
    ax[0].set_xticklabels([])
    ax[0].set_ylabel('Layers',fontsize=12,labelpad=5)

    psth = np.histogram(data['input']['times'],bins)[0]*1e3/10./300.
    ax[1].plot(bins[:-1],psth,drawstyle='steps-mid',linewidth=1.35,color='b')

    ax[1].set_yticks(np.arange(0,3001,1500))
    ax[1].set_ylabel('kHz',fontsize=12,labelpad=3)
    ax[1].set_yticklabels([0,'',3],fontsize=8,stretch='ultra-condensed')
    ax[1].set_xticks(np.arange(1400,2801,700))
    ax[1].set_xticklabels([1.4,2.1,2.8],fontsize=8,stretch='ultra-condensed')
    ax[1].set_xlabel('Time [s]',fontsize=12,labelpad=0)
    ax[1].set_ylim(0,3000)
    ax[1].set_xlim(1400,2800)

    plt.show()

def plot_act(axList=[],sd_list=[],vm_list=[],n_pop=[100],binSize=5.,wupTime=200.,simTime=1200.,figureName='figure'):
    #plotting parameters
    fs  = 12
    tfs = 8
    if len(axList)==0:
        #figure parameters
        fh = 7
        fw = 3
        #create figure
        fig = pl.figure(figureName,figsize=(fw,fh))
        fig.clf()
        axList.append(fig.add_axes([.22,.09,.750,.09]))#stim
        axList.append(fig.add_axes([.22,.21,.750,.17]))#psth
        axList.append(fig.add_axes([.22,.41,.750,.17]))#mem
        axList.append(fig.add_axes([.22,.61,.750,.38]))#raster
        #labels
        fig.text(0.050,0.825,'Neuron #'     ,fontsize=12,rotation='vertical')
        fig.text(0.000,0.525,'Membrane'     ,fontsize=12,rotation='vertical')
        fig.text(0.070,0.555,'potential [mV]',fontsize=12,rotation='vertical')
        fig.text(0.000,0.300,'Firing'       ,fontsize=12,rotation='vertical')
        fig.text(0.070,0.325,'rate [Hz]'    ,fontsize=12,rotation='vertical')
        fig.text(0.000,0.160,'Stimulus'     ,fontsize=12,rotation='vertical')
        fig.text(0.070,0.175,'rate [kHz]'   ,fontsize=12,rotation='vertical')
        fig.text(0.500,0.015,'Time [s]'     ,fontsize=12)

    #analysis parameters
    bins=np.arange(wupTime,wupTime+simTime+binSize,binSize)
    ylim = 0
    maxY = 30
    count1 = 1
    color_list = ['b','k','r']
    for idx1 in xrange(3):
        senders,times = to.get_senders_times(sd_list[idx1])

        #plot psth
        h,b = np.histogram(times,bins)
        axList[1].plot(b[:-1],h*1e3/binSize/n_pop[idx1],drawstyle='steps-mid',linewidth=1,color=color_list[idx1])

        #plot raster
        gids = np.unique(senders)
        if idx1==0:
            gids = gids[np.where(gids>302)]
        if len(gids)<100:
            ylim+=len(gids)
        else:
            ylim+=100
        for idx2 in xrange(len(gids)):
            if gids[idx2] in gids:
                spikes = times[senders==gids[idx2]]
                axList[3].plot(spikes,len(spikes)*[count1],'.',markersize=3,color=color_list[idx1])
                count1+=1
            if count1>ylim:
                break

        #plot membrane traces
        if idx1==1:
            senders_vm,times_vm,potentials= to.get_senders_times_potentials_gex_gin(vm_list[0])[:3]
            minLSS = 0
            for idx3 in xrange(2):
                for idx2 in np.arange(minLSS,len(senders_vm)):
                    n_spikes = len(times[senders==senders_vm[idx2]])
                    if n_spikes>1 and n_spikes<10:
                        selectedId = np.where(senders_vm==senders_vm[idx2])[0][0]
                        break
                times_vm = times_vm[np.argsort(times_vm)]
                for sp in times[senders==senders_vm[selectedId]]:
                    axList[2].plot(10*[sp],np.linspace(-54.,-50.,10),color=color_list[idx3],linewidth=1.35)
                vm = potentials[:,selectedId]
                refIdx = np.where(vm==-70)[0]
                for idx2 in xrange(len(refIdx)):
                    if vm[refIdx[idx2]-1]==-70:
                        continue
                    vm[refIdx[idx2]-1]=-54
                axList[2].plot(times_vm,vm,linewidth=1,color=color_list[idx3])
                minLSS=idx2+1
    axList[2].plot(np.linspace(wupTime,simTime+wupTime,100),100*[-70],':',color='gray',linewidth=1.5,alpha=.4)
    axList[2].plot(np.linspace(wupTime,simTime+wupTime,100),100*[-54],':',color='gray',linewidth=1.5,alpha=.4)

    #plot input histogram
    senders,times = to.get_senders_times(sd_list[-1])
    h,b = np.histogram(times,bins)
    axList[0].plot(b[:-1],h*1e3/binSize/n_pop[1],linewidth=1,color='gray',drawstyle='steps-mid')

    axList[3].set_xlim(wupTime,simTime+wupTime)
    axList[3].set_xticklabels([])
    axList[3].set_yticks(np.arange(0,ylim+1,ylim/2))
    axList[3].set_yticklabels(np.arange(0,ylim+1,ylim/2),fontsize=tfs,stretch='ultra-condensed')
    axList[3].set_ylim(0,ylim)
    axList[3].set_xlim(wupTime,simTime)

    axList[2].set_xticklabels([])
    axList[2].set_xlim(wupTime,simTime+wupTime)
    axList[2].set_yticks([-80,-65,-50])
    axList[2].set_yticklabels([-80,-65,-50],fontsize=tfs,stretch='ultra-condensed')
    axList[2].set_ylim(-80,-50)
    axList[2].set_xlim(wupTime,simTime)

    axList[1].set_xticklabels([])
    axList[1].set_xlim(wupTime,simTime+wupTime)
    axList[1].set_yticks([0,50,100])
    axList[1].set_yticklabels([0,50,100],fontsize=tfs,stretch='ultra-condensed')
    axList[1].set_ylim(0,100)
    axList[1].set_xlim(wupTime,simTime)

    axList[0].set_yticks([0,3000])
    axList[0].set_yticklabels([0,3],fontsize=tfs,stretch='ultra-condensed')
    axList[0].set_ylim(0,3000)
    axList[0].set_xticks(np.arange(wupTime,simTime+1.,500.))
    axList[0].set_xticklabels(np.arange(1,2.6,.5),fontsize=tfs)
    axList[0].set_xlim(wupTime,simTime)

    plt.show()

"""
#############
Main function
#############
"""

def main():
    global data_path,nest_dic,recorder_label
    global sim_time,wup_time,stim_start,stim_stop
    global neuron_model,neuron_param
    global del_within,del_between,del_chain,del_pp
    global g,j_chain,j_ext_high,j_ext_low,jexc,jinh,j_pulse_packet,j_inh_inh,j_ext_II
    global cp_exc_exc,cp_exc_inh,cp_inh_exc,cp_inh_inh,cp_chain
    global syn_exc_exc,syn_exc_inh,syn_inh_inh,syn_inh_exc,syn_chain
    global record_toFile,n_threads,record_Vm,LL_conn,verbose,recorder_params
    global activity,sigma,jitter,pulse_times,pulses
    global poi_rate_bkg_exc,poi_rate_bkg_inh,poi_rate_bkg_II
    global Nexc,Ninh,n_layers,group_size
    global ac,dc,freq,phi,E_drive

    usage = '%prog [options]'
    parser = OptionParser(usage)
    parser.add_option("--activity"   ,type="int"          ,default = 20       ,help="number spikes per pulse packet")
    parser.add_option("--sigma"      ,type="float"        ,default = 3.5      ,help="width of the pulse packet in ms")
    parser.add_option("--interval"   ,type="float"        ,default = 45.      ,help="interval between pulse packets in ms")
    parser.add_option("--delWit"     ,type="float"        ,default = 1.       ,help="delay within populations")
    parser.add_option("--delBet"     ,type="float"        ,default = 2.5      ,help="delay between populations")
    parser.add_option("--jitter"     ,type="float"        ,default = 0.       ,help="size of the jitter--> j>1: value in ms; otherwise: interval fraction")
    parser.add_option("--ext"        ,type="float"        ,default = 1.       ,help="External exc. input rate to E populations in kHz")
    parser.add_option("--extI"       ,type="float"        ,default = 1.       ,help="External exc. input rate to I neurons in kHz")
    parser.add_option("--g"          ,type="float"        ,default = 4.       ,help="Inhibition factor")
    parser.add_option("--synE"       ,type="float"        ,default = 5.       ,help="Time constant of excitatory synapses")
    parser.add_option("--synI"       ,type="float"        ,default = 10.      ,help="Time constant of inhibitory synapses")
    parser.add_option("--g_L"        ,type="float"        ,default = 10.      ,help="Leak conductance")
    parser.add_option("--jChain"     ,type="float"        ,default = .668     ,help="connection weight between groups")
    parser.add_option("--jPp"        ,type="float"        ,default = .668     ,help="connection weight pulse-packet generator")
    parser.add_option("--jexc"       ,type="float"        ,default = .668     ,help="Excitatory connection weight")
    parser.add_option("--jinh"       ,type="float"        ,default = -4.7     ,help="Inhibitory connection weight")
    parser.add_option("--jexthi"     ,type="float"        ,default = .8       ,help="Highest external connection weight")
    parser.add_option("--jextlo"     ,type="float"        ,default = .5       ,help="Lowest external connection weight")
    parser.add_option("--exp_name"   ,type="string"       ,default = "sim"    ,help="name of the experiment")
    parser.add_option("--with-vm"    ,action ="store_true",dest = "record_Vm" ,help="add jitter to the interval")
    parser.add_option("--mem_only"   ,action ="store_true",dest = "mem_only"  ,help="Don't save spikes into disk'")
    parser.add_option("--no_pulses"  ,action ="store_true",dest = "no_pulses" ,help="Stimulus is a Poisson spike train")
    parser.add_option("--no_verbose" ,action ="store_true",dest = "no_verbose",help="Print to terminal")
    parser.add_option("--E_drive"    ,action ="store_true",dest = "E_drive"   ,help="With E_drive")
    parser.add_option("--threads"    ,type="int"          ,default = 1        ,help="number of threads for the simulation")
    parser.add_option("--nExc"       ,type="int"          ,default = 1000     ,help="number of E neurons per layer")
    parser.add_option("--nLayers"    ,type="int"          ,default = 1        ,help="number of layers")
    parser.add_option("--j_II"       ,type="float"        ,default = -18.8    ,help="II weight")
    parser.add_option("--j_ext_II"   ,type="float"        ,default = -18.8    ,help="Weight of external connections replacing II")
    parser.add_option("--extII"      ,type="float"        ,default = 0.       ,help="External inh. input rate to I neurons in Hz")
    parser.add_option("--LLconn"     ,type="int"          ,default = 0        ,help="type 0: L-L; type 1: E-L; type 2: E-E")
    parser.add_option("--dc"         ,type="float"        ,default = 1.       ,help="Stimulus DC amplitude in kHz")
    parser.add_option("--ac"         ,type="float"        ,default = 0.       ,help="Stimulus AC amplitude in kHz")
    parser.add_option("--freq"       ,type="float"        ,default = 0.       ,help="Stimulus AC frequency in Hz")
    parser.add_option("--phi"        ,type="float"        ,default = 0.       ,help="Stimulus AC phase in rad")

    (options, args) = parser.parse_args()

    """
    #Set parameters
    """

    #data path for Nest data
    data_path       = os.getcwd()
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    #Nest configuration dictionary
    nest_dic = {'resolution':.1,'print_time':False,'overwrite_files':True,'data_path':data_path}
    recorder_label  = 'ctr_net'

    #neuron params
    neuron_model = 'iaf_cond_exp'
    neuron_param = {'V_th'      :-54.,
                    'V_reset'   :-70.,
                    't_ref'     :  5.,
                    'g_L'       : options.g_L,
                    'C_m'       :200.,
                    'E_ex'      :  0.,
                    'E_in'      :-80.,
                    'tau_syn_ex': options.synE,
                    'tau_syn_in': options.synI,
                    'tau_minus' : 20.}

    #simulation parameters
    n_threads       = options.threads
    sim_time        = 3e3
    wup_time        = 1e3
    stim_start      = wup_time+500.
    stim_stop       = sim_time-500.
    if options.no_verbose:
        verbose = False
    else:
        verbose = True

    #stimulus parameters
    if options.no_pulses:
        pulses  = False
        ac      = options.ac*1e3
        freq    = options.freq
        phi     = options.phi
        dc      = options.dc
    else:
        pulses  = True
        activity= options.activity
        sigma   = options.sigma
        jitter  = options.jitter

    #network parameters
    Nexc = options.nExc
    n_layers = options.nLayers
    Ninh = 500
    group_size = 300

    cp_exc_exc = .05
    cp_exc_inh = .1
    cp_inh_exc = .1
    cp_inh_inh = .1
    cp_chain   = .1

    syn_exc_exc = int(Nexc*cp_exc_exc)
    syn_exc_inh = int(Ninh*cp_exc_inh)
    syn_inh_inh = int(Ninh*cp_inh_inh)
    syn_inh_exc = int(Nexc*cp_inh_exc)
    syn_chain   = int(group_size*cp_chain)

    LL_conn = options.LLconn

    #delays
    del_chain   = 1.
    del_pp      = .5
    del_within  = options.delWit
    del_between = options.delBet
    set_delays()

    #weights
    g = options.g
    jexc = options.jexc
    jinh = options.jinh
    j_ext_high = options.jexthi
    j_ext_low = options.jextlo
    j_chain = options.jChain
    j_pulse_packet = options.jPp
    set_weights()
    j_inh_inh = options.j_II
    j_ext_II = options.j_ext_II

    #Poisson input rates
    poi_rate_bkg_exc    = options.ext
    poi_rate_bkg_inh    = options.extI
    poi_rate_bkg_II     = options.extII
    if options.E_drive:
        E_drive = True
    else:
        E_drive = False

    #recorder paramaters
    if options.mem_only:
        record_toFile = False
    else:
        record_toFile = True
    recorder_params = {'start':wup_time,'stop':sim_time,'to_file':record_toFile,'to_memory':True}

    if options.record_Vm:
        record_Vm = True
    else:
        record_Vm = False

    """
    #select the experiment
    """
    experiment_name=options.exp_name
    if experiment_name=='figure3a':
        figure3a()
    elif experiment_name=='figure4a':
        figure4a()
    elif experiment_name=='figure5b':
        figure5b()
    elif experiment_name=='figureS6a':
        figureS6a()


if __name__=='__main__':
    main()

