import sys
import re
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import numpy as np
import math

def load_input_data(path_of_file):
    input_data = []
    input_tokens = []
    labels =[]
    with open(path_of_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 3:
                token = parts[1] 
                class_label = parts[2]
                input_tokens.append(token)
                labels.append(class_label)

    # print(len(input_tokens)-1)

    for i in range(1, len(input_tokens)-1):
        if input_tokens[i].endswith('.'): 
            L = input_tokens[i].split(".")[0]
            R = input_tokens[i+1]
            label = labels[i]
            input_data.append((L,R,label))
            
    return input_data

def feature_extraction(input_data,feature_no):

    features = []
    labels = []
    # print(feature_no == 8)
    if feature_no == "5":
        for L, R, label in input_data:
            feature_vector = []
            # Core features
            feature_vector.append(L)      
            feature_vector.append(R)  
            feature_vector.append(len(L) < 4) 
            feature_vector.append(L.isdigit())  
            feature_vector.append(R[0].isupper())


            features.append(feature_vector)
            if label == "EOS":
                s = 1
            else:
                s = 0
            labels.append(s)

        features = np.array(features)
        label_enc= preprocessing.LabelEncoder()
        for i in range(5):
            features[:,i] = label_enc.fit_transform(features[:,i])
    elif feature_no == "8":
        for L, R, label in input_data:
            feature_vector = []
            # Core features
            feature_vector.append(L)      
            feature_vector.append(R)  
            feature_vector.append(len(L) < 4) 
            feature_vector.append(L.isdigit())  
            feature_vector.append(R[0].isupper())
            d= len(L)
            if d==0:
                n=0
            else:
                n= math.log(len(L))
            feature_vector.append(n)
            feature_vector.append(len(R))
            b= sum(1 for char in R if char.islower())
            if b==0:
                m=0
            else:
                m=math.log(sum(1 for char in R if char.islower()))
            feature_vector.append(m)
            
            features.append(feature_vector)
            if label == "EOS":
                s = 1
            else:
                s = 0
            labels.append(s)

        features = np.array(features)
        label_enc= preprocessing.LabelEncoder()
        for i in range(8):
            features[:,i] = label_enc.fit_transform(features[:,i])
    elif feature_no == "3":
        for L, R, label in input_data:
            feature_vector = []
            # Core features
            d= len(L)
            if d==0:
                n=0
            else:
                # print("print_d_value",d)
                n= math.log(len(L))
            feature_vector.append(n)
            feature_vector.append(len(R))
            b= sum(1 for char in R if char.islower())
            if b==0:
                m=0
            else:
                m=math.log(sum(1 for char in R if char.islower()))
            feature_vector.append(m)
            
            features.append(feature_vector)
            if label == "EOS":
                s = 1
            else:
                s = 0
            labels.append(s)

        features = np.array(features)
        label_enc= preprocessing.LabelEncoder()
        for i in range(3):
            features[:,i] = label_enc.fit_transform(features[:,i])
    else:
        print("give any of these feature numbers 3,5,8")

    return features, labels

def main(train_file, test_file, feature_no):
        train_input_data = load_input_data(train_file)
        test_input_data = load_input_data(test_file)

        X_train, y_train = feature_extraction(train_input_data,feature_no)
        X_test, y_test = feature_extraction(test_input_data,feature_no)

        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        print(f"Accuracy for {feature_no}_features: {accuracy * 100:.2f}%")

        if feature_no == "5":
            tree_text = export_text(clf, feature_names=['L', 'R', 'Len(L)<4', 'L_is_digit', 'R_is_cap'])
            print(tree_text)
        elif feature_no == "8":
            tree_text = export_text(clf, feature_names=['L', 'R', 'Len(L)<4', 'L_is_digit', 'R_is_cap', 'log_len_L', 'len_R', 'log_len_lower_R'])
            print(tree_text)
        elif feature_no == "3":
            tree_text = export_text(clf, feature_names=[ 'log_len_L', 'len_R', 'log_len_lower_R'])
            print(tree_text)

if __name__ == "__main__":
        main(sys.argv[1], sys.argv[2], sys.argv[3])
