
import order_classifier as o_model
import order_preprocess as o_pre






if __name__ == "__main__":
    print()
    data = o_pre.read_data("data/training_set.csv")
    #print(data)

    clf = o_model.readModel('model/rf_duplicate_total.model')
    o_model.evalModel(clf, data)
