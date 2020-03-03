from Utils import locate_string_in_arr
from Utils import extract_parameter_value_as_int

with open("gains-nn-merged.csv", "r", encoding="UTF-8") as f_gains,\
        open("prolog-k-nearest-neighbours.txt", "w", encoding="UTF-8") as f_prolog:
    c = 0
    data = []

    colname_dataset = 0
    colname_clf = 1
    colname_clf_family = 2
    colname_od_name = 3
    colname_removed = 4
    colname_accuracy = 5
    colname_base_acc = 6
    colname_gain = 7
    colname_od_params = 8

    for line in f_gains:
        data = line.strip().split(',')
        if c == 0:
            colname_dataset = locate_string_in_arr(arr=data, string="dataset")
            colname_clf = locate_string_in_arr(arr=data, string="clf")
            colname_clf_family = locate_string_in_arr(arr=data, string="clf_family")
            colname_od_name = locate_string_in_arr(arr=data, string="od_name")
            colname_od_params = locate_string_in_arr(arr=data, string="od_params")
            colname_removed = locate_string_in_arr(arr=data, string="removed")
            colname_accuracy = locate_string_in_arr(arr=data, string="accuracy")
            colname_base_acc = locate_string_in_arr(arr=data, string="accuracy_old")
            colname_gain = locate_string_in_arr(arr=data, string="gain")
            f_prolog.write(
                'd(id,DS,Classifier,Clf_family,Od_method,Od_params,Removed,With_OD_acc,Base_acc,Gain,Rand_acc)\n')
            c += 1
            continue

        # d(id,DS,Classifier,Clf_family,Od_method,Removed,With_OD_acc,Base_acc,Gain,Rand_acc)

        f_prolog.write('d(' + str(c) + ',' +
                       data[colname_dataset] + ',' +
                       data[colname_clf] + ',' +
                       data[colname_clf_family] + ',' +
                       data[colname_od_name] + ',' +
                       str(extract_parameter_value_as_int(data[colname_od_params], parameter="n_neighbors")) + ',' +
                       data[colname_removed] + ',' +
                       "{:0.5f}".format(float(data[colname_accuracy])) + ',' +
                       "{:0.5f}".format(float(data[colname_base_acc])) + ',' +
                       "{:0.5f}".format(float(data[colname_gain])) + ',' +
                       '0' +
                       ')\n')

        c += 1
        # if c > 10:
        #    break
