
# AntTracking
Tracking Ants in a lab table

This code is intended to detect and track ants over a white table.

Els videos, datasets, outputs, etc estan a:
* [TSC Drive Ramon](https://drive.tsc.upc.edu/index.php/s/f545rLJCdqGyYye?path=%2F)
* [OneDrive Ignasi](https://upcbcntech-my.sharepoint.com/:f:/g/personal/ignasi_nogueiras_office365_estudiantat_upc_edu/EhuKd08N809EsRrZpJPiS9gBaIbdCgtlYNG79Z7MwsC4-A?e=EXthGK) (EL CONTINGUT DESAPAREIXERÀ AVIAT!!!)

# Instalació d'entorn

L'entorn habitual es "**ants**", també hi ha l'entorn "fastreid" que pot ser útil i l'entorn "deepsort" que permet usar el codi dels autors de "deepocsort" (el nom deepsort en lloc de deepocsort es un despiste).

Per instalar-los, hi ha una serie de comandes i comentaris amb instruccions als arxius f"{entorn}.install" (algunes instalacións no son facils).

**També hi ha informació útil en la carpeta tutorials, potser algún pas fundamental en la instalació**.

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

A partir d'aquestes deteccions i un valor umbral basat en el nivell de gris mitja de la imatge, es segmenta la formiga per cada detecció, es calcula el PCA i s'afageix l'angle respecte a l'horitzontal les caracteristiques de la detecció.

    python3 ant_detection_pca.py ./DATA/output_4_gt.mp4 ./OUTPUT/output_4_2_pca.txt --varThreshold=20 --startWriteFrames=500

ES POT OMETRE DE CARA A LA CONTINUACIÓ DE LA INVESTIGACIÓ.

# ant_detection_yolo_sahi.py

Aplica un model de detecció YOLOv8n pre-entrenat.

S'aplica multiples vegades per cada frame, utilitçant crops d'una certa mida i, posteriorment, s'aggregen els resultats per obtenir deteccions per tota la imatge (SAHI).

A més, permet processar segments i ampliar (o crear) els fitxer de resultats anterior per evitar problemes amb el temps disponible pels estudiants a CALCULA (SLURM queues).

    python3 ant_detection_yolo_sahi.py DATA/ant_subset_1-000.mp4 OUTPUT/subset_1-000_yolo_dets.txt runs/detect/ants_yolo_2_fancy-sweep-13_11wl20fa/weights/best.pt --imgsz=640

o, en dos parts de 1000 frames,

    python3 ant_detection_yolo_sahi.py DATA/ant_subset_1-000.mp4 OUTPUT/subset_1-000_yolo_dets.txt runs/detect/ants_yolo_2_fancy-sweep-13_11wl20fa/weights/best.pt --imgsz=640 --initialFrame=0 --stopFrame=1000

    python3 ant_detection_yolo_sahi.py DATA/ant_subset_1-000.mp4 OUTPUT/subset_1-000_yolo_dets.txt runs/detect/ants_yolo_2_fancy-sweep-13_11wl20fa/weights/best.pt --imgsz=640 --initialFrame=1000 --stopFrame=1000

# ant_detection_yolo.py

Aplica un model de detecció YOLOv8n pre-entrenat.

S'aplica a tota la imatge de cada frame.

    python3 ant_detection_yolo.py DATA/ant_subset_1-000.mp4 OUTPUT/subset_1-000_yolo_dets.txt runs/detect/ants_yolo_2_fancy-sweep-13_11wl20fa/weights/best.pt --imgsz=640

ES POT OMETRE DE CARA A LA CONTINUACIÓ DE LA INVESTIGACIÓ.

# ant_detection.py

Aplica un model de detecció basat en extracció del fons (background), estimat de manera dinamica, i, posteriorment, detecció de components conexes.

    python3 ant_detection.py ./DATA/output_4_gt.mp4 ./OUTPUT/output_4_2.txt --varThreshold=20 --startWriteFrames=500

ES POT OMETRE DE CARA A LA CONTINUACIÓ DE LA INVESTIGACIÓ.

# appearance_tracks.py

Aplica un model d'aparença FastReID a qualsevol arxiu en format MOT Challenge (deteccions o tracks). Posteriorment es pot usar per tracking o per descriure tracklets sencers i fusionar-los.

    python3 appearance_tracks.py DATA/all_ants_0-007.mp4 OUTPUT/dets/all_ants_0-007_yolo_dets_151.txt dets/all_ants_0-007_yolo_dets_fastreid_151.txt --config=runs/apparence/ants_fastreid_vocal-sweep-6_cee5t8ta/config.yaml --weights=runs/apparence/ants_fastreid_vocal-sweep-6_cee5t8ta/model_best.pth

S'HAURIA D'ARREGLAR UNA MICA PER UNA INVESTIGACIÓ MÉS CENTRADA EN APARENÇA (per exemple, diversos models o millorar algun element hardwired).

# associated_histograms notebooks

En aquest grup de notebooks, s'aplica l'algoritme d'assignació de tracks estimats amb els tracks reals (ground truth) definit per [MOTChallenge](https://arxiv.org/pdf/1603.00831.pdf). Amb les assignacions, s'estudien diverses característiques dels resultats.

## associated_histograms.ipynb

En aquest cas, no s'aplica cap correcció de fase basada en pca ni s'espera que l'arxiu de tracks estimats contingui les prediccions del filtre de Kalman (perque el ground truth s'ha generat a partir de les mateixes deteccions).

Els resultats d'aquest notebook permeten estudiar graficament l'error en modul, fase (direccio i sentit) i intersection over union (IoU) entre les associacions.

## associated_histograms_pca.ipynb

En aquest cas, s'aplica una correcció de fase als desplaçaments de les formigues basada en la direcció del cos de la formiga; el sentit del canvi de fase es la mínima fase entre la fase de la predicció i la nova fase.

També s'espera que els tracks continguin les prediccions del filtre de Kalman i, adicionalment, l'angle amb l'eix X del cos de la formiga en la columna #10 (no usada en tracks 2D, previament reservada per les coordenades Z de tracks 3D).

Els resultats d'aquest notebook peremten estudiar graficament l'error en fase i  intersection over union (IoU) entre les associacions i comparar els errors del sistema amb PCA amb el sistema sense PCA.

ES POT OMETRE DE CARA A LA CONTINUACIÓ DE LA INVESTIGACIÓ.

# cut_tracklets.py

Adaptat del codi de Sigrid Vila.

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Aplica un umbral a la Interseccio sobre la Unio (IoU) per dividir tracks en parelles de tacklets en els punts on varies formigues son properes.

Es pot usar per dividir el problema en tracking de tracklets i, posteriorment, reidentificació. També es útil per estudiar la resolució d'encreuaments.

    python3 cut_tracklets.py OUTPUT/all_ants_0-007_gt_tcks_fastreid_151.txt OUTPUT/all_ants_0-007_splitgt_tcks_fastreid_151.txt

# cvat_ds_mot_to_mot.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Petit script per transformar les annotacions obtingudes per CVAT (en format MOT 1.1 i per els videos submostrejats a 1/2 horitzontals i verticals) al format usat durant el projecte (format MOT i resolucio completa).

    python3 cvat_ds_mot_to_mot.py gt/gt.txt all_ants_0-007_gt_tcks.txt

# deepsort_track.py

Aplica el model de tracking definit en [DeepSORT](https://arxiv.org/pdf/1703.07402.pdf). Requereix un arxiu amb les deteccions i els descriptors d'aparença en format MOT Challenge ampliat (resultat de appearance_tracks.py).

    python3 deepsort_track.py OUTPUT/dets/all_ants_0-007_yolo_dets_fastreid_151.txt all_ants_0-007_yolo_deepsort02iou30age_fastreid_151.txt --max_age=30 --iouThreshold=0.2

# detectors_map_test.py

Adaptat del codi del usuari de Kaggle [CYC](https://www.kaggle.com/code/chenyc15/mean-average-precision-metric).

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Petit script per evaluar els resultats d'un detector mitjaçant mean Average Precision (mAP@50 i mAP@50:95), falsos positius i falsos negatius, i precision i recall.

    python3 detectors_map_test.py OUTPUT/dets/all_ants_0-007_yolo_dets_151.txt DATA/all_ants_0-007_gt_tcks.txt

# hota_idf1.ipynb

Notebook on s'aplica la biblioteca [evaldet](https://github.com/tadejsv/EvalDeT) per calcular les metriques habituals en problemes de tracking (CLEARMOT, IDs i HOTA).

Els resultats del notebook permeten observar els valors numerics en forma tabular. També permet observar graficament la evolució de les components del HOTA en funció del parametre $\alpha$.

# interpolate_tracks.py

Adaptat del codi de Du Yunhao, Gaussian-smoothed interpolation (GSI from StrongSORT).

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

A partir d'un arxiu de tracking en format MOT, genera un altre arxiu de tracking en format MOT on els tracks tallats durant un periode inferior a "interval" s'han generat mitkançant una interpolacio gaussiana.

    python3 interpolate_tracks.py OUTPUT/all_ants_0-007_yolo_ocsort04th08ciou_151.txt OUTPUT/all_ants_0-007_yolo_ocsort04th08ciou_gsi_151.txt

# join_tracklets.py

**TODO**: AFLink, ReID for splitted tracklets, etc

# minimum_id.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Script per modificar els IDs discontinuus dels tracks detectats en IDs continuus desde 1 fins al nombre de tracks.

    python3 minimum_id.py DATA/output_4_gt.txt DATA/output_4_gt_min.txt

# ocsort_track.py

Aplica el model de tracking definit en [OCSort](https://arxiv.org/pdf/2203.14360.pdf). Requereix un arxiu amb les deteccions en format MOT Challenge (resultat de ant_detection.py).

    python3 ocsort_track.py DATA/detections.txt test.txt --iouThreshold=0.1 --associationFunc=ciou

# pca_tracks.py

A partir d'un arxiu de tracking en format MOTChallenge i un valor umbral basat en el nivell de gris mitja de la imatge, es segmenta la formiga per cada detecció, es calcula el PCA i s'afageix l'angle respecte a l'horitzontal les caracteristiques de la linea del arxiu.

    python3 pca_tracks.py DATA/output_4_gt.mp4 OUTPUT/ocsort_tracking_output_4.txt output_4_dets_pca.txt

ES POT OMETRE DE CARA A LA CONTINUACIÓ DE LA INVESTIGACIÓ.

# plot_pca_directions.ipynb

Notebook per comprobar el correcte funcionament del sistema amb PCA. Es pot observar graficament la segmentació de les formigues i els vectors de desplaçament que s'aplicarien usant Kalman o PCA.

Nomes s'observa un frame a la vegada i el numero de frames a avançar l'escull l'usuari al moment (cap numero o Esc avança 1 frame, un nombre negatiu o caracter no numeric finalitça el bloc).

ES POT OMETRE DE CARA A LA CONTINUACIÓ DE LA INVESTIGACIÓ.

# plot_rectangles_video.py

Codi de Ramon Morros. 

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST AL PRINCIPI.

No he mirat el funcionament del codi, permet generar videos amb els tracks dibuixats en diferents colors.

    python3 plot_rectangles_video.py DATA/output_4_gt.txt DATA/output_4_gt.mp4 output_4_gt_rectangles.mp4 --downsampleVideo=True

# plot_tracks.py

Codi basat en el codi d'algú.

Representa gràficament el ground truth amb les identitats assignades a cada identitat real (o fals positiu).

Requereix una estructura de directoris específica (la de MOT Challenge, mirar el directori plot_tracks_data).

    python3 plot_tracks.py plot_tracks_data/gt/mot_challenge/ plot_tracks_data/trackers/mot_challenge/ --trackerList=ocsort

# postprocess_track.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Actualment, nomes filtra bboxes més grans que un valor umbral de pixels per costat.

    python3 OUTPUT/all_ants_0-007_yolo_ocsort04th08ciou_151.txt OUTPUT/all_ants_0-007_yolo_ocsort04th08ciou_post_151.txt --max_size=200

ES POT OMETRE DE CARA A LA CONTINUACIÓ DE LA INVESTIGACIÓ.

# preanotate_dets_from_two.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Combina dos arxius de tracks, coneixement a priori i euristicas per clssificar cada deteccio en possibles true positives, possibles false negatives i possibles false positives.

El producte d'aquest script son 3 carpetes que inclouen una carpeta de crops que es pot usar per filtrar les deteccions incorrectes mitjançant la eliminació.

Les anotacions segueixen un format YOLO.

Posteriorment, amb preanotated_to_mot.py es pot generar un arxiu MOT amb les deteccions verificades.

Aquest script serveix per generar datasets per reidentificació o millorar la preanotacio abans d'importar el MOT a CVAT.

També es útil l'ús amb els dos arxius de tracks identics, sobretot per l'ús de "millorar la preanotacio abans d'importar el MOT a CVAT".

    python3 preanotated_to_mot.py OUTPUT/colonia_B0_clean/ OUTPUT/colonia_B0.txt 4000 3000 DATA/colonia_B_0_frames.txt

Cas de 2 tracks identics:

    python3 preanotate_dets_from_two.py DATA/ant_subset_1-000.mp4 OUTPUT/subset_1-000_yolo_dets.txt OUTPUT/subset_1-000_yolo_dets.txt ant_subset_1-000_from_two --onePerFrame=False

# preanotated_to_mot.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

A partir de la carpeta obtniguda per preanotate_dets_from_two.py (posteriorment revisada i modificada), genera un arxiu MOT.

Requereix la resolucio del video per passar de YOLO a MOT.

    python3 preanotated_to_mot.py ant_subset_1-000_from_two/ OUTPUT/subset_1-000_yolo_dets_corrected.txt 4000 3000

# pred_to_cvat.py

A partir d'un arxiu de tracking en format MOTChallenge amb IDs minimes (resultat de minimum_id.py) genera un ZIP i un JSON.

EL JSON conté el text necessari per definir les etiquetes d'una tasca de CVAT, s'aplicarà copiant i enganxant el contingut.

El ZIP conté una carpeta amb un arxiu de text amb tracks en format MOT1.1 i un altre arxiu de text amb noms per cada etiqueta. Aquest ZIP es pot usar per carregar els resultats del model a la tasca de CVAT configurada amb el JSON anterior.

    python3 pred_to_cvat.py DATA/output_4_gt_min.txt gt.zip

# reid_clustering_test.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Està pensat per ser usat en conjunt amb cut_tracklets.py i appearance_tracks.py. L'input ha de ser ground truths o tracklets basats en ground truth. Es pot millorar aplicant algoritmes d'associacio d'un input qualsevol amb el ground truth.

Mitjançant cut_tracklets.py es genera un arxiu on els tracks del ground truth estan dividits (mitjançant un espai de fals negatius i, posteriorment, canvi d'identitat).

Mitjaçant appearance_tracks.py es generen descriptors de cada deteccio del ground truth i dels ground truth alterat.

Aquest script analitza la capacitat d'utilitzar aparença per juntar els tracks en els casos on hi ha oclusions i genera una carpeta amb diverses gràfiques.

    python3 reid_clustering_test.py OUTPUT/all_ants_0-007_splitgt_tcks_fastreid_151_v2.txt OUTPUT/all_ants_0-007_gt_tcks_fastreid_151_v2.txt OUTPUT/REID_TEST_all_ants_0-007_gt_v2/

# reid_rotation_test.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Mitjançant un model d'apparença (Bag of Tricks de FastReid), un video i un ground truth, genera una imatge (amb dos grafiques polars) on es pot observar la capacitat que te el model entrenat per conservar identitats al simplement aplicar rotació a una detecció. Per fer-ho, es compara amb la capacitat de diferenciar identitats aplicant rotació en una detecció, mantenint la orientació de l'altre identitat.

    python3 -u reid_rotation_test.py DATA/all_ants_0-007.mp4 DATA/all_ants_0-007_gt_tcks.txt all_ants_0-007.png --num_imgs=20 --num_steps=45 --max_ids=10

S'HAURIA D'AFEGIR NOUS MODELS DE CARA A CONTINUAR INVESTIGANT APARENÇA.

# segment_detections.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Per generar el dataset d'aparença, cal descartar els intervals annotats manualment. 

Aquest script elimina les deteccions en els intervals on no es vol que hi hagi del arxiu MOT generat per un script de detecció.

    python3 segment_detections.py OUTPUT/dets/OLD/colonia_A_0104_dets.txt DATA/COLONIAS/colonia_A_0104_frames.txt OUTPUT/dets/OLD/colonia_A_0104_dets_seg.txt

# segment_tracks.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Per mantenir les identitats i filtrar falsos positius, a les deteccions filtrades per segment_detections.py se'ls hi aplica un tracker. 

Aquest script aplica els intervals annotats manualment per filtrar els frames on ha quedat més d'una formiga i assigna una id unica a totes les deteccions remanents dins de cada interval annotat.

    python3 segment_tracks.py OUTPUT/COLONIAS/colonia_A_0104_trck.txt DATA/COLONIAS/colonia_A_0104_frames.txt OUTPUT/COLONIAS/colonia_A_0104_trck_seg.txt

# sort_inference.py

Aplica el model de tracking definit en [SORT](https://arxiv.org/pdf/1602.00763.pdf). Requereix un arxiu amb les deteccions en format MOT Challenge (resultat de ant_detection.py).

    python3 sort_inference.py DATA/detections.txt

ES POT OMETRE DE CARA A LA CONTINUACIÓ DE LA INVESTIGACIÓ.

# ultralytics_to_validable.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

A partir d'una carpeta amb un dataset en format YOLO (object detection), genera una subcarpeta amb els crops anotats.

S'espera que les deteccions hagin sigut generades per un detector, un anotador ha d'eliminar els crops que consideri mals exemples. Posteriorment validable_to_ultralytics.py generarà un nou dataset omitint les imatges amb crops incorrectes.

    python3 ultralytics_to_validable.py ant_subset_1-000_det_dataset/

# unassociated_histograms.ipynb

Aquest notebook serveix per observar i comparar les distribuicions de 2 arxius de tracking, considerant una el _ground truth_.

Els resultats son histogrames de:
* velocitat (desplaçament en pixels / frame), 
* IoU entre dos frmaes consecutius d'un mateix track (per tots els tracks).
* cIoU entre dos frmaes consecutius d'un mateix track (per tots els tracks).

# validable_to_ultralytics.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

A partir de la carpeta obtinguda per ultralytics_to_validable.py i posteriorment repasada per un anotador, genera un daaset de detecció en format YOLO.

    python3 validable_to_ultralytics.py ant_subset_1-000_det_dataset ant_subset_1-000_det_dataset_corrected

# video_color_mean_std.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Calcula la mitja i la desviació estandard de tots els pixels d'un video.

    python3 video_color_mean_std.py DATA/ant_subset_1-000.mp4

# video_to_apparences.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Genera un dataset d'aparença compatible amb FastReid.

Els resultats son cuadrats.

L'input permet varios arxius de video i coresponent MOT per tal de generar un dataset amb moltes identitats sense colisio en les IDs.

    video_to_apparences.py DATA/COLONIAS/colonia_A_0104.avi OUTPUT/COLONIAS/colonia_A_0104_trck_seg.txt --pad_reshape

# video_to_crop_ultralytics.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Genera un dataset per deteccio d'objectes amb anotacions en format YOLO a partir de video i anotacions en format MOT.

Les imatges estan retallades en posicions aleatories amb un mínim de 1 objecte per retall, minimitzant el nombre d'objectes tallats, maximitzant el nombre d'objectes per retall i assegurant que cada objecte apareix sencer en algun retall.

Permet usar varios videos i arxius de detecció. També es pot seleccionar la fracció de test, el submosterig aplicat al video i les dimensions dels retalls.

    python3 video_to_crop_ultralytics.py colonias_640x640_crops DATA/VIDEOS/COLONIA_A/colonia_A_0.mp4 OUTPUT/colonia_A0.txt DATA/VIDEOS/COLONIA_B/colonia_B_1.mp4 OUTPUT/colonia_B1.txt

# video_to_oriented_apparences.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Genera un dataset d'aparença compatible amb FastReid.

Els resultats poden tenir dimensions personalitzables ja que l'algoritme intenta que la orientació de la formiga sigui sempre igual.

L'input permet varios arxius de video i coresponent MOT per tal de generar un dataset amb moltes identitats sense colisio en les IDs.

    python -u video_to_oriented_apparences.py DATA/VIDEOS/COLONIA_A/colonia_A_0.mp4 OUTPUT/colonia_A0.txt DATA/VIDEOS/COLONIA_B/colonia_B_1.mp4 OUTPUT/colonia_B1.txt --sampling_rate=1 --th=0.4 --pad_reshape

# video_to_ultralytics.py

EL DOCOPT D'AQUEST SCRIPT ESTÀ EN EL PROPI SCRIPT, JUST ABANS DEL MAIN.

Genera un dataset per deteccio d'objectes amb anotacions en format YOLO a partir de video i anotacions en format MOT.

Els frames son usats directament com a imatges completes.

    python3 video_to_ultralytics.py DATA/VIDEOS/COLONIA_A/colonia_A_0.mp4 OUTPUT/colonia_A0.txt colonia_A

ES POT OMETRE DE CARA A LA CONTINUACIÓ DE LA INVESTIGACIÓ.

# wandb_fast_reid.py

Script alterat de train_net.py de fastReid.

LES OPCIONS SON GESTIONADES PEL CODI DE FASTREID.

Permet aprofitar els sweeps de WandB i entrenar un model Bag of Tricks.

NO ESTA PENSAT PER SER USAT DIRECTAMENT. Inclus usant-lo a través de l'script correcte, es problematic (encara que els resultats que dona son vàlids).

ESTÀ FET MOLT PROVISIONALMENT I SERÍA MILLOR DESENVOLUPAR ALGO MILLOR.

# wandb_fast_reid.sh

Junta wandb_fast_reid.py i wandb_log_fast_reid.py per aprofitar correctement els sweeps de WandB.

POT GENERAR PROBLEMES D'ESPAI DISPONIBLE EN DISC DUR.

POT GENERAR PROBLEMES DE TIMEOUT (hi ha una mig solució) O DESINCRONITZACIÓ (la primera execució sempre fallarà) DE WANDB.

    WANDB__SERVICE_WAIT=86400 nohup wandb agent --count=12 ignasi00/ants_fastreid/ff031o0e > nohup_reid.out &

on wandb agent --count=12 ignasi00/ants_fastreid/ff031o0e farà 12 execuccions del script amb les opcions definides en un arxiu YAML que defineix com s'executarà aquest script.

# wandb_log_fast_reid.py

Permet aprofitar els sweeps de WandB i pujar els resultats i evolució de l'entrenament d'un model Bag of Tricks de fastReid.

NO ESTA PENSAT PER SER USAT DIRECTAMENT.

ESTÀ FET MOLT PROVISIONALMENT I SERÍA MILLOR DESENVOLUPAR ALGO MILLOR.

---

# YAML de WandB

Exemples de YAMLs que permeten configurar sweeps de WandB per l'entranament de la YOLOv8 i del Bag of Tricks de fastReid.
