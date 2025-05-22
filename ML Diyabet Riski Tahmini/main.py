#import kütüphaneleri

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler  #veriyi standart haline getirir
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

#klasik makine öğrenmesi algoritmaları
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier

#gereksiz uyarı mesajlarını bastırır
import warnings
warnings.filterwarnings("ignore")

#veriyi yükle
df = pd.read_csv("diabetes.csv")
df_name = df.columns
df.info() #sutun isimleri, sample sayisi ve kayip veri problemi, veri tipleri
describe = df.describe() #ortalam, std, min-max gibi temel istatistikler

sns.pairplot(df, hue="Outcome") #pairplot ve korelasyon analizi
plt.show()

def plot_correlation_heatmap(df):
    corr_matrix = df.corr()
    plt.figure(figsize=(10,8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", linewidths=0.5)
    plt.title("Correlation of Features")
    plt.show()

plot_correlation_heatmap(df)


#IQR yöntemini kullanarak aykırı değerleri tespit et ve temizle

def detect_outliers_iqr(df):
    outlier_indices = []  #aykırı değerlerin indexlerini tutacak boş bir liste
    outliers_df = pd.DataFrame() #aykırı değerleri saklamak için boş bir dataframe

    for col in df.select_dtypes(include=["float64", "int64"]).columns: #sadece sayısal veri tipine sahip tüm sütunlar
        Q1 = df[col].quantile(0.25) #verinin %25'lik kısmı
        Q3 = df[col].quantile(0.75) #verinin %75'lik kısmı
        IQR = Q3 -Q1 #ortadaki %50'lik verinin yayılımı
        lower_bound = Q1 - 1.5*IQR #2.5 de olabilir farketmez ama bu yöntemde standart olan 1.5
        upper_bound = Q3 + 1.5*IQR

        outliers_in_col = df[(df[col] < lower_bound) | (df[col] > upper_bound)] #alt veya üst sınırların dışında kalan veriler seçiliyor(aykırı değerler)
        outlier_indices.extend(outliers_in_col.index) #listeye aykırı değerlerin indekslerini ekledik
        outliers_df = pd.concat([outliers_df, outliers_in_col], axis=0) # aykırı değerlerin verilerini ekledik

    outlier_indices = list(set(outlier_indices)) #aynı satır satır birden fazla sütunda aykırı olabilir. bu nedenle aynı
    outliers_df = outliers_df.drop_duplicates() #indeks tekrarlarını ile veri tekrarını temizliyoruz
    return outliers_df, outlier_indices #tespit ettiğimiz aykırı değerlerin veri satırları ve indexleri döndürülüyor

outliers_df, outlier_indices = detect_outliers_iqr(df) #fonksiyonu çağırıyoruz
df_cleaned = df.drop(outlier_indices).reset_index(drop=True) #aykırı değerlerini temizledik


#testler

X = df_cleaned.drop(["Outcome"], axis=1)
y = df_cleaned["Outcome"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

 #standartizasyon
 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#modelleme oluştur

def getBasedModel():
    basedModels = []
    basedModels.append(("LR", LogisticRegression()))
    basedModels.append(("DT", DecisionTreeClassifier()))
    basedModels.append(("KNN", KNeighborsClassifier()))
    basedModels.append(("GNB", GaussianNB()))
    basedModels.append(("SVM", SVC()))
    basedModels.append(("AdaB", AdaBoostClassifier()))
    basedModels.append(("GBM", GradientBoostingClassifier()))
    basedModels.append(("RF", RandomForestClassifier()))
    return basedModels

def baseModelsTraning(X_train, y_train, models):
    results = []
    names = []
    for name, model in models:
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print(f"{name}: accuracy: {cv_results.mean()}, std: {cv_results.std()}")
    return names, results

def plot_box(names, results):
    df = pd.DataFrame(results, index=names).T  # Transpose ederek sütunları model isimleri yapıyoruz
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=df)
    plt.title("Model Accuracy Distribution")
    plt.ylabel("Accuracy")
    plt.xlabel("Models")
    plt.xticks(rotation=45)
    plt.show()
    
models = getBasedModel()
names, results = baseModelsTraning(X_train_scaled, y_train, models)
plot_box(names, results)

model_scores = pd.DataFrame({
    "Model": names,
    "Mean Accuracy": [r.mean() for r in results],
    "Std Dev": [r.std() for r in results]
}).sort_values("Mean Accuracy", ascending=False)

print(model_scores.head(3))  # en iyi 3 model


#decision tree için hiperparametre oluşturuyoruz

param_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}
dt = DecisionTreeClassifier()

grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring="accuracy")

grid_search.fit(X_train_scaled, y_train)

print("En iyi parametreleri: ", grid_search.best_params_)

best_dt_model = grid_search.best_estimator_
y_pred = best_dt_model.predict(X_test_scaled)
print("Confuusion Matris")
print(confusion_matrix(y_test, y_pred))

print("Classification Report")
print(classification_report(y_test, y_pred))


#yeni veriyle model testler yap

new_data = np.array([[6,149,72,35,0,34.6,0.627,51]])  # çift köşeli!
new_data_scaled = scaler.transform(new_data)         # unutma: modeli scale edilmiş veriye eğittik
new_prediction = best_dt_model.predict(new_data_scaled)
print("New prediction:", new_prediction)
