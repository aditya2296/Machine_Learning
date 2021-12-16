import pandas
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore');

d = pandas.read_csv(r"C:\Users\Pc\Desktop\Materials\Datas.csv");

d = d.drop("ids",axis=1);
d["nucleuses"] = d["nucleuses"].replace(0,d["nucleuses"].mean());

X = d.drop("Classes",axis=1);
y = d["Classes"].copy();
c_m = d.corr();
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2);

def display(a,b):
    print("confusion_matrix: ",confusion_matrix(a,b),".");
    print("accuracy_score: ",accuracy_score(a,b),".");
    print("precision_score: ",precision_score(a,b,pos_label=2),".");
    print("recall_score: ",recall_score(a,b,pos_label=2),".");
    print("f1_score: ",f1_score(a,b,pos_label=2),".");
    
    
print();
print("knearestsneighborsclassifier.");
k_n_n_c = KNeighborsClassifier().fit(X_train,y_train);
print();
print("training data.");
display(y_train, k_n_n_c.predict(X_train));
print();
print("test data.");
display(y_test,k_n_n_c.predict(X_test));

print();
print("linearsupportvectormachineclassifier.");
l_s_v_m_c = LinearSVC().fit(X_train,y_train);
print();
print("training data.");
display(y_train,l_s_v_m_c.predict(X_train));
print();
print("test data.");
display(y_test,l_s_v_m_c.predict(X_test));

print();
print("randomforestclassifier.");
r_f_c = RandomForestClassifier(n_estimators=500,max_leaf_nodes=8,n_jobs=-1).fit(X_train,y_train);
print();
print("training data.");
display(y_train,r_f_c.predict(X_train));
print();
print("test data.");
display(y_test,r_f_c.predict(X_test));
