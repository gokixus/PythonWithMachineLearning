#gerekli import kütüphaneleri
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

iris = load_iris() #veri setini yükle
X = iris.data #bağımsız değişkenler(özellikleri). Çanak yaprak uzunluk-genişlik, taç yaprak uzunluk-genişlik. Yanı sayısal değerleri içeriyor
y = iris.target #bağımlı değişkenler(etiketleri). 3 farklı çiçek türü içeriyor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk oranı : %{accuracy*100:0.2f}")

