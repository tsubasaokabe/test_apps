from sklearn import svm
from sklearn import datasets
from sklearn.externals import joblib

def main():
    #SVMを分類機にする
    clf = svm.SVC()
    #データセットの読み込み
    iris = datasets.load_iris()
    #従属変数と説明変数
    X,y = iris.data, iris.target
    #学習
    clf.fit(X,y)
    joblib.dump(clf,'./model/sample-model.pkl')

if __name__ =='__main__':
    main()
