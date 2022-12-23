from os import listdir
from os.path import isfile, join
import shutil

path = "/home/xenono/AI/AI_Coursework/dataset/raw_datasets/"
folders = ["cars", "trucks", "buses", "bikes"]

for folder in folders:
    src = path + folder + "/"
    dst = path + "3779_80Train_20Test/training_set/" + folder + "/"
    for file in listdir(src)[:3000]:
        shutil.move(src + file, dst + file)
