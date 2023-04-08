
# Esquelets dels programas

<!---
![EsqueletDeteccions](./DATA/readme_images/EsqueletDeteccions.png)
-->

| generar deteccions | generar tracks |
| --- | --- |
| <img src="./DATA/readme_images/EsqueletDeteccions.png" alt="EsqueletDeteccions" width="50%" title="EsqueletDeteccions" /> | <img src="./DATA/readme_images/EsqueletTracking.png" alt="EsqueletTracking" width="69%" title="EsqueletTracking" /> |

Tots els programes (scripts) tenen una interficie docopt amb l'ajuda activada.


---
<br>


# ant_detection_pca.py

Aplica un model de detecció basat en extracció del fons (background), estimat de manera dinamica, i, posteriorment, detecció de components conexes.

A partir d'aquestes deteccions i un valor umbral basat en el nivell de gris mitja de la imatge, es segmenta la formiga per cada detecció, es calcula el PCA i s'afageix l'angle respecte a l'horitzontal les coaracteristiques de la detecció.

python3 ant_detection_pca.py ./DATA/output_4_gt.mp4 ./OUTPUT/output_4_2_pca.txt --varThreshold=20 --startWriteFrames=500

# ant_detection.py

Aplica un model de detecció basat en extracció del fons (background), estimat de manera dinamica, i, posteriorment, detecció de components conexes.

python3 ant_detection.py ./DATA/output_4_gt.mp4 ./OUTPUT/output_4_2.txt --varThreshold=20 --startWriteFrames=500

# associated_histograms notebooks

En aquest grup de notebooks, s'aplica l'algoritme d'assignació de tracks estimats amb els tracks reals (ground truth) definit per [MOTChallenge](https://arxiv.org/pdf/1603.00831.pdf).

## associated_histograms.ipynb

En aquest cas, no s'aplica cap correcció de fase basada en pca i s'espera que l'arxiu de tracks estimats contingui les prediccions del filtre de Kalman (perque el ground truth s'ha generat a partir de les mateixes deteccions).

Els resultats d'aquest notebook permeten estudiar graficament l'error en modul, fase (direccio i sentit) i intersection over union (IoU) entre les associacions.

## associated_histograms_pca.ipynb

En aquest cas, s'aplica una correcció de fase als desplaçaments de les formigues basada en la direcció del cos de la formiga; el sentit del canvi de fase es la mínima fase entre la fase de la predicció i la nova fase.

També s'espera que els tracks continguin les prediccions del filtre de Kalman i, adicionalment, l'angle amb l'eix X del cos de la formiga en la columna #10 (no usada en tracks 2D, previament reservada per les coordenades Z de tracks 3D).

Els resultats d'aquest notebook peremten estudiar graficament l'error en fase i  intersection over union (IoU) entre les associacions i comparar els errors del sistema amb PCA amb el sistema sense PCA.

# deepsort_detection.py

# deepsort_track.py

# hota_idf1.ipynb

# minimum_id.py

# ocsort_track.py

# pca_tracks.py

# plot_pca_directions.ipynb

# plot_rectangles_video.py

# pred_to_cvat.py

# sort_inference.py

# unassociated_histograms.ipynb
