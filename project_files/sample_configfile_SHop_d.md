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
- R[11] : Alm maximo en acuifero.

Los valores de calibración varían de acuerdo a la escala temporal y 
espacial de ejecución del modelo.  Cada uno de estos parámetros es 
multiplicado como un escalar por el mapa que componga una variable **X**
del modelo. 

################################################################################################
|Nombre | id| evp | ks_v | kp_v | Kpp_v | v_sup | v_sub | v_supt | v_cau | Hu | Hg | Hq |
|--------:|----:|:---:|:----:|:----:|:-----:|:-----:|:-----:|:------:|:-----:|:--:|:--:|:--:|
| -c   | -p01 | 1.0 | 5.9 | 5.7 | 0.0 | 1.0 | 1.0 | 10.8 | 1.0 | 1.0 | 1.0 | 1.0 |


## C.I.

Condiciones iniciales en caso de que no exista un binario establecido
para alguno de los casos presentados en la tabla:

- **Inicial Capilar**:
- **Inicial Escorrentia**:
- **Inicial Subsup**:
- **Inicial Subterraneo**:
- **Inicial Corriente**:


### SHop_E260_90m_1d

Este proj corre con una CI inicial en una ventanas: -1d + -90d de warming.


## Rutas universales

Estas rutas son basicas para la ejecucion del modelo, toda la ejecucion y los resultados dependen de y se almacenan en esta rutas.
Para garantizar la lectura de las rutas, estas deben escribirse sin comillas y respetando
los espacios, deben tener la sgte. estructura: - **nombreruta**: ruta.

- **name_proj**: SHop_E260_90m_1d
    > Nombre del proyecto.
- **ruta_proj**: /home/hidrologia/jupyter/SH_op/SHop_E260_90m_1d/SHop/project_files/
    > Ruta donde se almacena el proyecto, donde se guardan todos los archivos.

    
## Rutas de ejecucion

- **ruta_nc**: inputs/E260_90m_py3_v111.nc
    > Ruta del .nc de la cuenca para la simulación.
- **ruta_nc_tramos**: inputs/E260_90m_py2_tramos.csv
    > Ruta del .csv con la asignacion de tramos que corresponden a estaciones de nivel.
- **ruta_csv_subbasinmask**: inputs/df_E260_90m_py2_subbasins_posmasks.csv
    > Ruta del .csv con la asignacion de posiciones del nc que corresponden a la subcuenca de cada estacion de nivel.
- **ruta_curvascalob3**: inputs/df_curvascal_frankenstein_20200915.csv
    > Ruta en donde se encuentran el dataframe con coeficientes de curvas de calibracion observada
- **ruta_modelset**: inputs/model_settings_d.json
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
    > Ruta en donde se encuentran las copias de almacenamiento que pueden remplazar a las operacionales
- **ruta_qsim_op**: results/results_op/Qsim_op
    > Ruta del .csv con resultados de caudal simulado.
- **ruta_MS_hist**: results/results_h/MSto_hist
    > Ruta del .csv con resultados del almacenamiento simulado - histórico.
- **ruta_qsim_hist**: results/results_h/Qsim_hist
    > Ruta del .csv con resultados de caudal simulado - histórico.


## Rutas resultados a desplegar

- **ruta_qsim_png**: results/graficas/
    > Ruta donde se almacenan las imágenes con resultados de simulación.
