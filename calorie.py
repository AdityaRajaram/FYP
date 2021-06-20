import cv2
import numpy as np
from image_segment import *

#density - gram / cm^3
density_dict = { 'apple':0.808, 'orange':0.814, 'onion':0.95, 'tomato':0.47, 'banana':1.09 , 'carrot':1.04, 'cucumber':0.94}
#kcal
calorie_dict = { 'apple':77, 'orange':62,  'onion': 40, 'tomato':18 , 'banana':72, 'carrot':30, 'cucumber':25 }
#skin of photo to real multiplier
skin_multiplier = 5*2.3

def getCalorie(label, volume): #volume in cm^3
    calorie = calorie_dict[(label)]
    density = density_dict[(label)]
    mass = volume*density*1.0
    calorie_tot = (calorie/100.0)*mass
    return mass, calorie_tot, calorie #calorie per 100 grams

def getVolume(label, area, skin_area, pix_to_cm_multiplier, fruit_contour):
    area_fruit = (area/skin_area)*skin_multiplier #area in cm^2
    volume = 100
    if label in ['apple', 'tomato', 'orange', 'onion'] : #sphere-apple,tomato,orange,kiwi,onion
        radius = np.sqrt(area_fruit/np.pi)
        volume = (4/3)*np.pi*radius*radius*radius
        #print (area_fruit, radius, volume, skin_area)
    if label in ['banana', 'cucumber '] or (label == 'carrot' and area_fruit > 30): #cylinder like banana, cucumber, carrot
        fruit_rect = cv2.minAreaRect(fruit_contour)
        height = max(fruit_rect[1])*pix_to_cm_multiplier
        radius = area_fruit/(2.0*height)
        volume = np.pi*radius*radius*height
    if (label=='carrot' and area_fruit < 30) : # carrot
        volume = area_fruit*0.5 #assuming width = 0.5 cm
    return volume

def calories(result,img, segment_dir):
    img_path =img
    fruit_areas,final_f,areaod,skin_areas, fruit_contours, pix_cm = getAreaOfFood(img_path, segment_dir)
    volume = getVolume(result, fruit_areas, skin_areas, pix_cm, fruit_contours)
    mass, cal, cal_100 = getCalorie(result, volume)
    fruit_volumes=volume
    fruit_calories=cal
    fruit_calories_100grams=cal_100
    fruit_mass=mass
    return fruit_calories