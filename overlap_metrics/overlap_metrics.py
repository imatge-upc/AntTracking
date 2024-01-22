"""overlap_metrics.py
    Entrada: name to json archive

Usage:
  overlap_metrics.py (-h | --help)
  overlap_metrics.py <input_file> <gt_file> [--isGT=<gt>]

Options:
  -h --help       Muestra esta ayuda.
  --opcional=<valor>  Valor opcional.
"""

import pandas as pd
import os
import cv2
from docopt import docopt
import re
import copy

def read_annotations_from_txt(txt_path):
    """
    Lee anotaciones desde un archivo de texto con el formato especificado.

    Parameters:
    - txt_path (str): Ruta al archivo de texto.

    Returns:
    - pandas.DataFrame: DataFrame de pandas con las anotaciones.
    """
    annotations = []

    with open(txt_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 5:
                annotation = {'id': int(parts[0]),
                              'x': float(parts[1]),
                              'y': float(parts[2]),
                              'w': float(parts[3]),
                              'h': float(parts[4])}
                annotations.append(annotation)

    return pd.DataFrame(annotations)


def box_overlap(box1, box2):
    """
    Determines if two bbox are overlapping

    Parameters:
    - box1 (tuple): Coordenadas de la primera bbox en formato (x, y, w, h).
    - box2 (tuple): Coordenadas de la segunda bbox en formato (x, y, w, h).

    Returns:
    - bool: True if bbox are overlapping, else False
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    left1, top1, right1, bottom1 = x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2
    left2, top2, right2, bottom2 = x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2

    return not (right1 < left2 or left1 > right2 or bottom1 < top2 or top1 > bottom2)

def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) score for two bounding boxes.

    Parameters:
    - box1 (tuple): Coordinates of the first bbox in the format (x, y, w, h).
    - box2 (tuple): Coordinates of the second bbox in the format (x, y, w, h).

    Returns:
    - float: IoU score.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    intersection_left = max(x1 - w1 / 2, x2 - w2 / 2)
    intersection_top = max(y1 - h1 / 2, y2 - h2 / 2)
    intersection_right = min(x1 + w1 / 2, x2 + w2 / 2)
    intersection_bottom = min(y1 + h1 / 2, y2 + h2 / 2)

    intersection_area = max(0, intersection_right - intersection_left) * max(0, intersection_bottom - intersection_top)

    box1_area = w1 * h1
    box2_area = w2 * h2

    union_area = box1_area + box2_area - intersection_area

    iou_score = intersection_area / union_area if union_area > 0 else 0.0

    return iou_score


def txt_creation(txt_overlaps,val_dir,add_ovlps):
    with open(txt_overlaps, 'w') as txt:
        for path in sorted(os.listdir(val_dir)):
            df = read_annotations_from_txt(os.path.join(val_dir,path))
            num_overlaps = 0
            for i,row_i in df.iterrows():
                for j, row_j in df.iloc[i+1:].iterrows():
                    if calculate_iou((row_i['x'], row_i['y'], row_i['w'], row_i['h']),
                                (row_j['x'], row_j['y'], row_j['w'], row_j['h']))>=0.10:
                        num_overlaps=num_overlaps+1
                        if num_overlaps==1: txt.write(f"{path.replace('.txt','')} / "); overlaps = f" / [{row_i['x'], row_i['y'], row_i['w'], row_i['h']} - {row_j['x'], row_j['y'], row_j['w'], row_j['h']}]" 
                        else: overlaps += f" [{row_i['x'], row_i['y'], row_i['w'], row_i['h']} - {row_j['x'], row_j['y'], row_j['w'], row_j['h']}]"
                        #print(f"OVerlap beetween annotations {i} y {j} en:\t {path.replace('.txt','')}")
            if num_overlaps!=0:
                if add_ovlps: txt.write(str(num_overlaps)+overlaps+"\n")
                else: txt.write(str(num_overlaps)+"\n")

def is_same_ovlp(bbox_pred, bbox_gt, threshold=0.8):
    if isinstance(bbox_pred, (list, tuple)) and len(bbox_pred) == 4 and isinstance(bbox_gt, (list, tuple)) and len(bbox_gt) == 4:
        x1, y1, w1, h1 = bbox_pred
        x2, y2, w2, h2 = bbox_gt
        iou_score = calculate_iou(bbox_pred, bbox_gt)
        print(f"\t\t\t{iou_score}")
        return iou_score >= threshold
    else:
        # Manejar el caso cuando bbox_pred o bbox_gt no son tuplas válidas
        return False

def are_same_ovlps(ovlp_pred_pair, ovlp_gt_pair, threshold=0.5):
    ovlp_pred1, ovlp_pred2 = ovlp_pred_pair
    ovlp_gt1, ovlp_gt2 = ovlp_gt_pair

    # Check if both overlaps in the pair are considered the same
    same_ovlp_1 = is_same_ovlp(ovlp_pred_pair[0], ovlp_gt_pair[0], threshold=threshold)
    same_ovlp_2 = is_same_ovlp(ovlp_pred_pair[1], ovlp_gt_pair[1], threshold=threshold)
    same_ovlp_3 = is_same_ovlp(ovlp_pred_pair[0], ovlp_gt_pair[1], threshold=threshold)
    same_ovlp_4 = is_same_ovlp(ovlp_pred_pair[1], ovlp_gt_pair[0], threshold=threshold)
    # Return True if both pairs are considered the same, else False
    return (same_ovlp_1 and same_ovlp_2) or (same_ovlp_3 and same_ovlp_4)

def extract_positions(input_string):
    # Eliminar caracteres no deseados y dividir la cadena por '] ['
    overlaps_match = re.findall(r'\[([^\]]+)\]', input_string)
    overlaps = []
    for match in overlaps_match:
        overlaps_str = match
        overlap = [tuple(map(float, box.replace('(','').replace(')','').split(', '))) for box in overlaps_str.split(') - (')]
        overlaps.append(overlap)
    return overlaps

def compare_overlaps(txt_gt,txt_preds):
    true_ovlp=[]
    num_true_ovlp = 0
    false_ovlp=[]
    num_false_ovlp = 0
    not_det_ovlp= []
    num_not_det_ovlp = 0
    dif_num_ovlp=[]
    total_ovlp_seen = 0
    with open(txt_gt,'r') as gt_txt, open(txt_preds,'r') as preds_txt:
        gts = gt_txt.readlines()
        data_gt = [gt.strip().split('/') for gt in sorted(gts)]
        preds = preds_txt.readlines()
        data_preds = [pred.strip().split('/') for pred in sorted(preds)]
    df_gt = pd.DataFrame(data_gt, columns=['Img ID', 'Num Overlaps', 'Positions'])
    df_preds = pd.DataFrame(data_preds, columns=['Img ID', 'Num Overlaps', 'Positions'])
    all_gt=df_gt['Num Overlaps'].astype(int).sum()
    for i, pred_row in df_preds.iterrows():
            df_aux=df_gt[df_gt['Img ID']==(pred_row['Img ID'])]
            true_ovlps=False
            if df_aux.empty:
                false_ovlp.append(pred_row['Img ID'])
                num_false_ovlp += int(pred_row['Num Overlaps'])

            elif not (df_aux['Num Overlaps'].str.strip() == pred_row['Num Overlaps']).all():
                ovlp_gt = int(df_aux['Num Overlaps'])
                ovlp_pred = int(pred_row['Num Overlaps'])
                dif_num_ovlp.append(pred_row['Img ID'])
                #Comprovar si quins overlaps coincideixen amb gt i quins no coincideixen
                pred_pos_list=extract_positions(pred_row['Positions'])
                #gt_pos_list = extract_positions(df_aux['Positions']) #.apply(extract_positions)
                gt_pos_list = extract_positions(df_aux['Positions'].iloc[0])
                pred_pos_list_=copy.copy(sorted(pred_pos_list))
                for j in range(len(pred_pos_list)):
                    pos_pred = pred_pos_list[j]
                    #for index, pos_gt_ in gt_pos_list.items():
                    for pos_gt in gt_pos_list:
                        if are_same_ovlps(pos_pred,pos_gt):
                            num_true_ovlp += 1
                            gt_pos_list.remove(pos_gt)  # Eliminar pos_gt de gt_pos_list
                            #break
                            pred_pos_list_.remove(pos_pred)
                num_false_ovlp += len(pred_pos_list_)
                num_not_det_ovlp += len(gt_pos_list)

            else:
                true_ovlp.append(pred_row['Img ID'])
                # Comprovar si tots el overlaps concideixen
                pred_pos_list=extract_positions(pred_row['Positions'])
                gt_pos_list = extract_positions(df_aux['Positions'].iloc[0]) #.apply(extract_positions
                pred_pos_list_=copy.copy(sorted(pred_pos_list))
                for j in range(len(pred_pos_list)):
                    pos_pred = pred_pos_list[j]
                    #for index, pos_gt_ in gt_pos_list.items():
                    for pos_gt in gt_pos_list:
                        if are_same_ovlps(pos_pred,pos_gt):
                            print("\t\t\tCorrect OVLP\n")
                            num_true_ovlp += 1
                            gt_pos_list.remove(pos_gt)  # Eliminar pos_gt de gt_pos_list
                            pred_pos_list_.remove(pos_pred)
                num_false_ovlp += len(pred_pos_list_)
                num_not_det_ovlp += len(gt_pos_list)

    for _,gt_row in df_gt.iterrows():
        if gt_row['Img ID'] not in df_preds['Img ID'].values: 
            not_det_ovlp.append(gt_row['Img ID'])
            num_not_det_ovlp += int(gt_row['Num Overlaps'])

    print(f"OVLP preds = {len(df_preds)} SUM: {len(false_ovlp)+len(dif_num_ovlp)+len(true_ovlp)}")
    print(f"True OVLP:\t Num: {len(true_ovlp)}\n{true_ovlp}\nFalse OVLP:\t Num: {len(false_ovlp)}\n{false_ovlp}\nDiff Num OVLP:\t Num: {len(dif_num_ovlp)}\n{dif_num_ovlp}\nNot Det OVLP: \t Num: {len(not_det_ovlp)}\n{not_det_ovlp}\n\n")
    print(f"TP: {num_true_ovlp}\tFP: {num_false_ovlp}\tFN: {num_not_det_ovlp}\tPrecision: {(num_true_ovlp/(num_true_ovlp+num_not_det_ovlp))}\tRecall: {(num_true_ovlp/(num_true_ovlp+num_false_ovlp))}")


        
if __name__ == "__main__":

    arguments = docopt(__doc__)
    root_dir_results = '/home/usuaris/imatge/pol.serra.i.montes/TFG/AntTracking/overlap_metrics/results'
    dir_preds = arguments['<input_file>']
    dir_gt = arguments['<gt_file>']
    txt_ovlp_preds = os.path.join(root_dir_results,"ovlp_metrics_"+dir_gt.split('/')[-3]+".txt")
    txt_ovlp_gt = os.path.join(root_dir_results,"ovlp_metrics_gt_"+dir_gt.split('/')[-3]+".txt")
    
    if not os.path.isfile(txt_ovlp_preds) :
        txt_creation(txt_ovlp_preds,dir_preds,True)
    if not os.path.isfile(txt_ovlp_gt):
        txt_creation(txt_ovlp_gt,dir_gt,True)

    compare_overlaps(txt_ovlp_gt,txt_ovlp_preds)
