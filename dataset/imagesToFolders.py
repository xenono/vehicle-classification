import pandas as pd
import shutil

csv = pd.read_csv("trucks and buses/df.csv")
print(csv)
vehiclesCount = {
    "Bus": 0,
    "Truck": 0
}

for index, row in csv.iterrows():
    src = "/home/xenono/AI/AI_Coursework/dataset/trucks and buses/images/" + row["ImageID"] + ".jpg"
    vT = row["LabelName"]
    dst = "/home/xenono/AI/AI_Coursework/dataset/trucks and buses/"
    print(vT)
    if vT == "Bus":
        dst += "buses/" + row["ImageID"] + ".jpg"
    elif vT == "Truck":
        dst += "trucks/" + row["ImageID"] + ".jpg"
    try:
        shutil.copy(src, dst)
        vehiclesCount[vT] += 1
    except FileNotFoundError:
        print("File has been already moved: ", row["ImageID"])

print("Trucks: ", vehiclesCount["Truck"])
print("Buses: ", vehiclesCount["Bus"])
