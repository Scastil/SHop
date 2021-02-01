# PAQUETES PARA CORRER OP.
import pandas as pd
import numpy as np
import datetime as dt
import json
import wmf.wmf as wmf
import hydroeval
import glob
import MySQLdb
#modulo pa correr modelo
import SHop
import hidrologia

#FORMATO
# fuente
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.dates as mdates
import matplotlib.font_manager as font_manager
font_dirs = ['/home/hidrologia/jupyter/SH_op/fuentes/AvenirLTStd-Book']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
matplotlib.rcParams['font.family'] = 'Avenir LT Std'
matplotlib.rcParams['font.size']=12
import pylab as pl 
#axes
pl.rc('axes',labelcolor='#4f4f4f')
pl.rc('axes',linewidth=1.5)
pl.rc('axes',edgecolor='#bdb9b6')
pl.rc('text',color= '#4f4f4f')
# pl.rc('xtick',color='#4f4f4f')
# pl.rc('ytick',color='#4f4f4f')
# pl.locator_params(axis='y', nbins=5)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)

print (dt.datetime.now())

print (dt.datetime.now())
############################################################################################  ARGUMENTOS
ruta_proj = '/home/hidrologia/jupyter/SH_op/SHop_E260_90m_1d/SHop/project_files/'
configfile=ruta_proj+'inputs/configfile_SHop_E260_90m_1d.md'
save_hist = True #####################################################False for first times
date = '2020-10-28 22:13'#dt.datetime.now().strftime('%Y-%m-%d %H:%M')

############################################################################################ EJECUCION

ConfigList= SHop.get_rutesList(configfile)
# abrir simubasin
path_ncbasin = SHop.get_ruta(ConfigList,'ruta_proj')+SHop.get_ruta(ConfigList,'ruta_nc')
cu = wmf.SimuBasin(rute=path_ncbasin)

#sets para correr modelo.
SHop.set_modelsettings(ConfigList)
warming_steps =  0#pasos de simulacion, dependen del dt.
warming_window ='%ss'%int(wmf.models.dt * warming_steps) #siempre en seg
dateformat_starts = '%Y-%m-%d'

starts  = ['%ss'%(90*24*60*60)]#,'%ss'%(90*24*60*60)] #60d back
starts_names = ['90d']#,'1d'] #starts y starts_names deben ser del mismo len.
window_end = '0s' #none

print ('######')
print ('Start DAILY execution: %s'%dt.datetime.now()) 

#dates
date = (pd.to_datetime(pd.to_datetime(date).strftime('%Y-%m-%d')) - pd.Timedelta('1 day')) #llega al dia antes del evento.

starts_w = [date - pd.Timedelta(start) for start in starts]
starts_m_d = [start_w - pd.Timedelta(warming_window) for start_w in starts_w]
end_d = date + pd.Timedelta(window_end)


# df execution
df_executionprops_d = pd.DataFrame([starts,
                                  starts_names,
                                  ['%s-p01-ci1-90d.StOhdr'%(SHop.get_ruta(ConfigList,'ruta_proj')+SHop.get_ruta(ConfigList,'ruta_sto_op'))], #first run: 0.0,
                                  ['ci1'],
                                  [[1.0 , 5.9 , 5.7 , 0.0 , 1.0 , 1.0 , 10.8 , 1.0 , 1.0 , 1.0, 1.0 ]],
                                  ['-p01'],
                                  [0]], #wup_stets:pasos de sim, depende de dt #int((end - starts_w[0]).total_seconds()/wmf.models.dt)
                                 columns = [1],
                                 index = ['starts','start_names','CIs','CI_names','pars','pars_names','wup_steps']).T

#rainfall
pseries,ruta_out_rain_d = SHop.get_rainfall2sim(ConfigList,cu,path_ncbasin,[starts_m_d[0]],end_d, #se corre el bin mas largo.
                                             Dt= float(wmf.models.dt),include_escenarios=None,
                                             evs_hist= False,
                                             check_file=True,stepback_start = '%ss'%int(wmf.models.dt *1),
                                             complete_naninaccum=True,verbose=False)

print (ruta_out_rain_d)

# set of executions
ListEjecs_d = SHop.get_executionlists_fromdf(ConfigList,ruta_out_rain_d,cu,starts_m_d,end_d,df_executionprops_d,
                                             warming_steps=warming_steps, dateformat_starts = dateformat_starts)

#execution
print ('Start simulations: %s'%dt.datetime.now())
print ('start: %s - end: %s'%(starts_m_d[0], end_d))
res = SHop.get_qsim(ListEjecs_d,set_CI=True,save_hist=save_hist,verbose = True)
print ('End simulations: %s'%dt.datetime.now())