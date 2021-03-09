Esta carpeta es para almacenar archivos que se generar y sobreescriben con la ejecucion de cada paso de tiempo, por eso se llaman archivos operacionales. Se genera el mismo grupo de archivos por cada "versión" de CIs que corre el modelo.

Hasta ahora, en ese grupo de archivos op. se generan:

- Binarios de almacenamiento: .St0bin y .St0hdr
- Df con caudal simulado en todos los nodos: Qsim_blabla.csv
- Df con humedad del suelo simulada(porc. de saturación) en todas las estaciones: HSsim_blabla.csv
- Df con caudal observado de las estaciones a comparar: df_qobs.csv
- Df con nivel observado de las estaciones a comparar: df_nobs.csv
- .csv's con humedad del suelo observada en las estaciones: estH_obs/codigoesth.csv
