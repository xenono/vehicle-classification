import pandas as pd
import shutil

csv = pd.read_csv("df.csv")

vehiclesCount = {
    "Bus": 0,
    "Truck": 0
}

for index, row in csv.iterrows():
    src = "/home/xenono/AI/AI_Coursework/buses and trucks/images/" + row["ImageID"] + ".jpg"
    vT = row["LabelName"]
    dst = "/home/xenono/AI/AI_Coursework/buses and trucks/"
    if vT == "Bus":
        dst += "buses/" + row["ImageID"] + ".jpg"
    elif vT == "Truck":
        dst += "trucks/" + row["ImageID"] + ".jpg"
    try:
        shutil.move(src, dst)
        vehiclesCount[vT] += 1
    except FileNotFoundError:
        print("File has been already moved: ", row["ImageID"])
    if vehiclesCount["Bus"] > 2000 and vehiclesCount["Truck"] > 2000:
        break
print("Trucks: ", vehiclesCount["Truck"])
print("Buses: ", vehiclesCount["Bus"])
