#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""




import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit




def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)

### Calculate min/max exercised stock options
def get_exercised_stock_options_min_max(data_dict):
    exercised_stock_options_list = []

    for key, value  in data_dict.iteritems():
        name = key
        exercised_stock_options = value['exercised_stock_options']
        exercised_stock_options_list.append((name, exercised_stock_options))

    sorted_list = sorted(exercised_stock_options_list, key=lambda x: x[1])
    filtered_list = [x for x in sorted_list if x[1] != 'NaN']
    print "Most exercised stock options:" + str(filtered_list[-1])
    print "Least exercised stock options:" + str(filtered_list[0])

get_exercised_stock_options_min_max(data_dict)



### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list )
poi, finance_features = targetFeatureSplit( data )

### Scaling
from sklearn import preprocessing
salary_data = featureFormat(data_dict, ["salary"])
eso_data = featureFormat(data_dict, ["exercised_stock_options"])

salary_scaler = preprocessing.MinMaxScaler()
salary_scaler.fit(salary_data)
print "Scaled salary of 200,000: " + str(salary_scaler.transform([[200000]]))

eso_scalar = preprocessing.MinMaxScaler()
eso_scalar.fit(eso_data)
print "Scaled EOS of 1,000,000: " + str(eso_scalar.transform([[1000000]]))

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2, _ in finance_features:
    plt.scatter( f1, f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2).fit(finance_features)
pred = kmeans.predict(finance_features)


### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters-3.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print "no predictions object named pred found, no clusters to plot"
