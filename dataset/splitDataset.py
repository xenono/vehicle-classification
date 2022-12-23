from os import listdir
from os.path import isfile, join
import shutil

path = "/home/xenono/AI/AI_Coursework/dataset/"
folders = ["cars", "trucks", "buses", "bikes"]

for folder in folders:
    src = path + "training_set/" + folder + "/"
    dst = path + "test_set/" + folder + "/"
    for file in listdir(src)[:400]:
        shutil.move(src + file, dst + file)
