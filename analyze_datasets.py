import os
directory = "/home/xbajger/BAKPR/AutoML/data/datasets/"

with open("datasets_features_records.csv", mode="w", encoding="UTF-8") as csv:
    csv.write("dataset,features,records\n")
    for filename in os.listdir(directory):
        data_switch = False
        records_counter = 0
        features_counter = 0
        with open(directory + filename, mode="r") as dataset:
            for line in dataset:
                if data_switch:
                    records_counter += 1
                elif line.startswith("@ATTR"):
                    features_counter += 1
                elif line.startswith("@DATA"):
                    data_switch = True
        csv.write(filename.split(".")[0] + "," + str(features_counter) + "," + str(records_counter) + "\n")