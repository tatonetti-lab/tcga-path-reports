#Post-Process AWS Response Files: Identify and Remove TCGA QC Tables, Remove handwriting as annotated by AWS Output
import glob
import pickle
import pandas as pd
import os, shutil
from collections import Counter
import matplotlib.pyplot as plt 
import random
import numpy as np 
import regex
from PIL import Image, ImageDraw
from IPython.display import display

response_list = glob.glob('data/textract_response/*.p')
random.shuffle(response_list)

new_dir = 'post_table_and_handwriting_removal/' 
if not os.path.exists(new_dir):
    os.makedirs(new_dir)

large_area_dir = new_dir+'/tables_drawn_area_geq_point05/' 
if not os.path.exists(large_area_dir):
    os.makedirs(large_area_dir)


def detect_qc_table(img):

    match_list = []
    
    image_file = 'imgs_for_aws/' + img
    response =  pickle.load( open('textract_response/'+img.replace('.jpg','')+'_response.p', "rb" ))
    blocks = response['Blocks']
    image=Image.open(image_file)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    keyword_dict = {'agnosis Discrep':{'left1':0,'top1':-.01,'left2':.3,'top2':.075}, 
                   'imary Tumor Site Discrep':{'left1':0,'top1':-.02,'left2':.3,'top2':.065}, 
                   'IPAA':{'left1':0,'top1':-.03,'left2':.3,'top2':.055},
                   'rior Malignancy History':{'left1':0,'top1':-.04,'left2':.3,'top2':.045},
                   'ual/Syn':{'left1':0,'top1':-.05,'left2':.3,'top2':.035}, 
                   'ase is \(cir':{'left1':0,'top1':-.06,'left2':.3,'top2':.025}, 
                   'eviewer Initials':{'left1':0,'top1':-.07,'left2':.3,'top2':.015},
                   'ate Reviewed':{'left1':-.085,'top1':-.065,'left2':.22,'top2':.015}, 
                   'ISQUALI':{'left1':-.16,'top1':-.06,'left2':.13,'top2':.015}}

    error_dict = {'agnosis Discrep':1,
                   'imary Tumor Site Discrep':2,
                   'IPAA':0,
                   'rior Malignancy History':2,
                   'ual/Syn':1,
                   'ase is \(cir':1,
                   'eviewer Initials':2,
                   'ate Reviewed':1, 
                   'ISQUALI':1}

    coord_dict = {'left':[],'top':[],'right':[],'bottom':[]}
    
    for block in blocks:
        if (block['BlockType'] == "LINE"):
            for keyword in list(keyword_dict.keys()):
                matches = regex.findall('('+keyword+'){e<='+str(error_dict[keyword])+'}',block['Text'])
                if len(matches) > 0:
                    match_list += matches
                    box=block['Geometry']['BoundingBox']
                    left = width * box['Left'] 
                    top = height * box['Top'] 
                    coord_dict['left'].append(left+width*keyword_dict[keyword]['left1'])
                    coord_dict['top'].append(top+height*keyword_dict[keyword]['top1'])
                    coord_dict['right'].append(left +width*keyword_dict[keyword]['left2'])
                    coord_dict['bottom'].append(top +height*keyword_dict[keyword]['top2'])
    
    #If QC table detected, take max of all bounding boxes for QC table 
    if len(coord_dict['left'])>0:
        left_qc = min(coord_dict['left'])
        top_qc = min(coord_dict['top'])
        right_qc = max(coord_dict['right'])
        bottom_qc = max(coord_dict['bottom'])
                
        qc_box_area = (right_qc-left_qc)/width*(bottom_qc-top_qc)/height #area in units of proportion of pixels

        draw = ImageDraw.Draw(image)
        draw.rectangle([left_qc, top_qc, right_qc, bottom_qc], outline='black') 
        
        #Save image with drawn rectangle 
        if len(match_list) == 1:
            image.save(new_img_dir + img.replace('.jpg','')+'_tables_drawn.png')
        #display(image) #inline
        
        if qc_box_area >= .05:
            image.save(large_area_dir + img.replace('.jpg','')+'_tables_drawn.png')
        
    else:
        qc_box_area = 0
        left_qc, top_qc, right_qc, bottom_qc = 0,0,0,0
    return qc_box_area, {'left':left_qc, 'right':right_qc,'top':top_qc,'bottom':bottom_qc}, match_list



#Check if a contains b
def check_subset(a,b):
    #Note: (0,0) is the upper left hand corner of the coordinates
    if (a['left'] <= b['left'] and a['right'] >= b['right']):
         x_overlap = True
    else:
        x_overlap = False
    if (a['top'] <= b['top'] and a['bottom'] >= b['bottom']):
        y_overlap = True
    else:
        y_overlap = False
    if x_overlap and y_overlap:
        return True
    else:
        return False

#Check whether Two Rectangles Overlap
def rect_overlap(a,b):
    if ((a['right'] >= b['left']) and (a['right'] <= b['right'])) or ((a['left'] >= b['left']) and (a['left'] <= b['right'])):
         x_overlap = True
    else:
        x_overlap = False
    if ((a['top'] <= b['bottom']) and (a['top'] >= b['top'])) or ((a['bottom'] >= b['bottom']) and (a['bottom'] <= b['top'])):
        y_overlap = True
    else:
        y_overlap = False
    if x_overlap and y_overlap:
        return True
    else:
        return False
    
#Compute Area between Two Rectangles (Bounding Boxes) - as a proportion of the area of the line
def get_overlap_area(a,b): #(qc_box, line_box)
    mid_x = sorted([a['left'],a['right'],b['left'],b['right']])[1:3]
    width = abs(mid_x[0]-mid_x[1])
    mid_y = sorted([a['top'],a['bottom'],b['top'],b['bottom']])[1:3]
    height = abs(mid_y[0]-mid_y[1])
    return width*height/(abs(b['left']-b['right'])*abs(b['top']-b['bottom'])) 

#Check whether a line contains Printed or Handwritten Text (as detected by Textract)
def check_line_printed(line, words):
    
    type_list = []
    id_list = line['Relationships'][0]['Ids']
    
    for word_id in id_list:
        match = [a for a in words if a['Id'] == word_id][0] #unique id
        if len(match) > 0: 
            type_list.append(match['TextType'])
    if set(type_list) != {'HANDWRITING'}: 
        return True
    else:
        return False

#Remove TCGA QC Table and Handwritten Lines
def process_img(img, qc_box_coord):
    
    final_line_list = []
    
    image_file = 'imgs_for_aws/' + img #Image = Page from Pathology Report (Downloaded from TCGA Portal)
    response =  pickle.load( open('aws_response/'+img.replace('.jpg','')+'_response.p', "rb" ))
    blocks = response['Blocks']
    image=Image.open(image_file)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    words = [a for a in blocks if a['BlockType']=='WORD']
    orig_n_lines = len([a for a in blocks if a['BlockType']=='LINE'])
    
    for block in blocks:
        
        if (block['BlockType'] == "LINE"):
            
            #remove the line if overlap >= 75% of its area with QC table bounding box 
            
            line_coord = {}
            box=block['Geometry']['BoundingBox']
            left = width * box['Left'] 
            top = height * box['Top'] 
            line_coord['left']=left
            line_coord['top']= top
            line_coord['right']= left + (width * box['Width'])
            line_coord['bottom']=top +(height * box['Height'])
            
            #If line is subset of QC table, do not include in final list
            if check_subset(qc_box_coord,line_coord):
                pass
                #draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline='black')
            
            #If line overlaps with QC table > 25% of line's area, do not include in final list
            elif rect_overlap(qc_box_coord,line_coord) or rect_overlap(line_coord,qc_box_coord):
                overlap_area = get_overlap_area(qc_box_coord,line_coord)
                if overlap_area <=.25:
                   
                    #If line contains only handwritten words, do not include in final list
                    if check_line_printed(block, words): 
                        final_line_list.append(block['Text'])
                        
                    #draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline='black')
            else:
                #If line contains only handwritten words, do not include in final list
                if check_line_printed(block, words):
                    final_line_list.append(block['Text'])
                    
    if len(final_line_list)>0: #Automatically exclude empty pages
        pickle.dump(final_line_list, open(new_dir + img.replace('.jpg','')+'_lines.p', "wb") )
    return orig_n_lines - len(final_line_list)
    
def display_all(img, qc_box_coord):
    image_file = 'imgs_for_aws/' + img
    response =  pickle.load( open('textract_response/'+img.replace('.jpg','')+'_response.p', "rb" ))
    blocks = response['Blocks']
    image=Image.open(image_file)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    draw.rectangle([qc_box_coord['left'], qc_box_coord['top'], 
                    qc_box_coord['right'], qc_box_coord['bottom']], outline='black') 
    for block in blocks:
        if (block['BlockType'] == "LINE"):
            line_coord = {}
            box=block['Geometry']['BoundingBox']
            left = width * box['Left'] 
            top = height * box['Top'] 
            line_coord['left']=left
            line_coord['top']=top
            line_coord['right']= left + (width * box['Width'])
            line_coord['bottom']=top +(height * box['Height'])
            draw.rectangle([left,top, left + (width * box['Width']), top +(height * box['Height'])],outline='black')
    
    display(image) 
    
#Run QC Removal 
lines_removed_dict = {}
med_conf_dict = {}
final_match_list = []
area_dict = {}
for i in range(len(response_list)):
    if i % 200 == 0:
        print(i, 'pages processed.')
    img = response_list[i].split('/')[1].replace('_response.p','.jpg')
    qc_area, qc_box_coord, interim_match_list = detect_qc_table(img)
    area_dict[img] = qc_area
    final_match_list += interim_match_list
    n_lines_removed = process_img(img, qc_box_coord) 
    lines_removed_dict[img] = n_lines_removed

print('Number of pages with non-zero lines:', len(glob.glob(new_dir +'*.p')))
print('Total number of lines removed:',sum(list(lines_removed_dict.values())))

