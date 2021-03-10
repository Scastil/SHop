## Parametros de calibracion

Cada una se compone de 11 parámetros escalares, los cuales son:

- R[1] : Evaporación.
- R[2] : Infiltración.
- R[3] : Percolación.
- R[4] : Pérdidas.
- R[5] : Vel Superficial.
- R[6] : Vel Sub-superficial.
- R[7] : Vel Subterranea.
- R[8] : Vel Cauces.
- R[9] : Alm capilar maximo.
- R[10] : Alm gravitacional maximo.
- R[11] : Alm maximo del acuifero.

Los valores de calibración varían de acuerdo a la escala temporal y 
espacial de ejecución del modelo.  Cada uno de estos parámetros es 
multiplicado como un escalar por el mapa que componga una variable **X**
del modelo. 

################################################################################################
|Nombre | id| evp | ks_v | kp_v | Kpp_v | v_sup | v_sub | v_supt | v_cau | Hu | Hg | Hq |
|--------:|----:|:---:|:----:|:----:|:-----:|:-----:|:-----:|:------:|:-----:|:--:|:--:|:--:|
|    | -p01 | 1.0 | 5.9 | 5.7 | 0.0 | 1.0 | 1.0 | 10.8 | 1.0 | 1.0 | 1.0 | 1.0 |
| -c   | -p02 | 0.8 | 10 | 17.7 | 0.0 | 9.0 | 2.0 | 15 | 0.9 | 1.0 | 1.0 | 1.0 |



## C.I.

Condiciones iniciales en caso de que no exista un binario establecido
para alguno de los casos presentados en la tabla:

- **Inicial Capilar**:
- **Inicial Escorrentia**:
- **Inicial Subsup**:
- **Inicial Subterraneo**:
- **Inicial Corriente**:


### SHop_E260_90m_1h

Este proj corre con una CI inicial en 3 ventanas: -30d, -3d+CI1d -3d+CIpant.


## Rutas generales

Estas rutas son basicas para la ejecucion del modelo.
Para garantizar la lectura de las rutas, estas deben escribirse sin comillas y respetando
los espacios, deben tener la sgte. estructura: - **nombreruta**: ruta.

- **name_proj**: SHop_E260_90m_1h
    > Nombre del proyecto.
- **ruta_proj**: /home/socastillogi/jupyter/git/SHop/project_files/
    > Ruta donde se almacena el proyecto, donde se guardan todos los archivos.
- **ruta_md_d**: /home/socastillogi/jupyter/SH_op/SHop_E260_90m_1d/SHop/project_files/inputs/configfile_SHop_E260_90m_1d.md
    > Ruta del configfile de ejecucion diaria.

    
## Rutas de ejecucion

Estas cosas solo se leen.

- **ruta_nc**: inputs/E260_90m_py3_v111.nc
    > Ruta del .nc de la cuenca para la simulación.
- **ruta_nc_metadatos**: inputs/metadatosdeests_2validar.csv
    > Ruta del .csv con metadatos del .nc: asignacion de tramos que corresponden a estaciones de nivel y curvas de estimación de caudales.
- **ruta_csv_subbasinmask**: inputs/df_E260_90m_py2_subbasins_posmasks.csv
    > Ruta del .csv con la asignacion de posiciones del nc que corresponden a la subcuenca de cada estacion de nivel.
- **ruta_modelset**: inputs/model_settings_h.json
    > Ruta de model_settings para configurar ejecución del modelo.
- **ruta_CI_reglaspant**: inputs/CI/reglas_pant/
    > Ruta de archivo con reglas de condiciones iniciales para intervalos de lluvia acumulada.
- **ruta_radardbz**: /var/radar/operacional/
    > Ruta de carpeta con barridos de reflectividad de radar
- **ruta_credenciales**: inputs/credenciales.csv
    > Ruta de credenciales de acceso a bd.
   

## Rutas resultados de ejecución.

- **ruta_rain**: results/results_op/rain_op/
    > Ruta en donde se generan los binarios de precipitación para ejecutar simulación.
    
- **ruta_sto_op**: results/results_op/Sto_op
    > Ruta en donde se encuentran los binarios con mapas de almacenamiento de cada tanque hidrológico.
- **ruta_qsim_op**: results/results_op/Qsim_op
    > Ruta del .csv con resultados de caudal simulado para todos los nodos.
- **ruta_HSsim_ests_op**: results/results_op/HSsim_ests_op
    > Ruta del .csv con resultados del humedad del suelo simulada en los puntos donde hay estaciones (% saturación)
    
- **ruta_nobs_op**: results/results_op/df_nobs.csv
    > Ruta del .csv con consulta de ultimo mes de nobs para graficar.
- **ruta_qobs_op**: results/results_op/df_qobs.csv
    > Ruta del .csv con consulta de ultimo mes de qobs para graficar.
- **ruta_HSobs_op**: results/results_op/estH_obs/
    > Ruta de la carpeta donde se guarda los .csv con la consulta de ultimo mes de HSobs para graficar.
   
- **ruta_MS_hist**: results/results_h/MSto_hist
    > Ruta del .csv con resultados del almacenamiento simulado - histórico.
- **ruta_qsim_hist**: results/results_h/Qsim_hist
    > Ruta del .csv con resultados de caudal simulado - histórico.
- **ruta_HSsim_ests_hist**: results/results_h/HSsim_ests_hist
    > Ruta del .csv con resultados del humedad del suelo simulada en los puntos donde hay estaciones (% saturación) -histórico


## Rutas resultados a desplegar

- **ruta_graficas_resultados**: results/graficas/
    > Ruta donde se almacenan todas las graficas de resultados.
- **ruta_qsim_png**: results/graficas/caudal/
    > Ruta donde se almacenan las imágenes con cuadal sim.
- **ruta_nsim_png**: results/graficas/nivel/
    > Ruta donde se almacenan las imágenes con nivel sim.
- **ruta_hs_sim_png**: results/graficas/humedaddelsuelo/estaciones/
    > Ruta donde se almacenan las imágenes con humedad del suelo sim.
- **ruta_hglog_sim_png**: results/graficas/humedaddelsuelo/mapas/
    > Ruta donde se almacenan gráficas del mapa con humedad del suelo sim.: Hg en escala log.
- **ruta_psat_sim_png**: results/graficas/humedaddelsuelo/mapas/
    > Ruta donde se almacenan gráficas del mapa con humedad del suelo sim.: Hg+Hg en (porcentaje de saturación).
    
    
<!-- > **ruta_nsim_op**: /media/nicolas/maso/Soraya/SHOp_files/SHop_SM_E260_90m_1h/results_op/Nsim_op
> **ruta_qsim_ns_hist**: /media/nicolas/maso/Soraya/SHOp_files/SHop_SM_E260_90m_1h/results_H/performance/Qsim_NS_hist
    > Ruta del .csv con resultados del desempeno del modelo - histórico. Criterio: Nash-Sutcliffe.
> **ruta_qsim_kge_hist**: /media/nicolas/maso/Soraya/SHOp_files/SHop_SM_E260_90m_1h/results_H/performance/Qsim_KGE_hist
    > Ruta del .csv con resultados del desempeno del modelo - histórico. Criterio: Kling-Gupta.
> **ruta_nsim_ns_cco_hist**: /media/nicolas/maso/Soraya/SHOp_files/SHop_SM_E260_90m_1h/results_H/performance/Nsim_NS_cco_hist
    > Ruta del .csv con resultados del desempeno del modelo - N - histórico. Criterio: Nash-Sutcliffe.
> **ruta_nsim_kge_cco_hist**: /media/nicolas/maso/Soraya/SHOp_files/SHop_SM_E260_90m_1h/results_H/performance/Nsim_KGE_cco_hist
    > Ruta del .csv con resultados del desempeno del modelo - histórico. Criterio: Kling-Gupta.  
> **ruta_performance_op**: /media/nicolas/maso/Soraya/SHOp_files/SHop_SM_E260_90m_1h/results_op/performance/
    > Ruta en donde se guarda resumen de desempeno operacional
> **ruta_performance_hist**: /media/nicolas/maso/Soraya/SHOp_files/SHop_SM_E260_90m_1h/results_H/performance/
    > Ruta del .csv con resultados del desempeno del modeloo - histórico. Se usa para escribir resumenes de par escogidas por cada criterio para cada estacion para cada paso de tiempo. -->

