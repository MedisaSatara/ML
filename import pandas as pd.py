
import pandas as pd
import numpy as np

# Učitavanje podataka iz CSV fajla
data = pd.read_csv('IB170012_MedisaSatara_podaci_ML.csv', encoding='latin1')

# Definisanje grupa predmeta (ovo je samo primer, prilagodite prema stvarnim podacima)
predmeti_grupe = {
    'Matematika': ['P-219', 'P-221'],
    'Programiranje': ['P-175', 'P-176', 'P-149'],
    'Engleski jezik': ['P-12', 'P-246'],
    'Ekonomija': ['P-222', 'P-220'],
    'Informatika': ['P-218'],
    'Mreze': ['P-142', 'P-58']
}

# Računanje prosečnih ocena po grupama
for grupa, predmeti in predmeti_grupe.items():
    ocene_kolone = [f'ProsjecnaOcjena_{predmet}' for predmet in predmeti]
    
    # Osiguravanje da su sve vrednosti numeričke
    for kolona in ocene_kolone:
        data[kolona] = pd.to_numeric(data[kolona], errors='coerce')
    
    data[f'ProsjecnaOcjena_{grupa}'] = data[ocene_kolone].mean(axis=1)

# Definisanje interesovanja kao oblast sa najvišom prosečnom ocenom
# Zamenjujemo NaN vrednosti pre nego što primenimo idxmax
prosjecne_ocene_grupe = [f'ProsjecnaOcjena_{grupa}' for grupa in predmeti_grupe.keys()]
data[prosjecne_ocene_grupe] = data[prosjecne_ocene_grupe].fillna(-1)
definirano_interesovanje = data[prosjecne_ocene_grupe].idxmax(axis=1)
data['Interesovanje'] = definirano_interesovanje.apply(lambda x: x.split('_')[-1])

# Sada imamo kolonu 'Interesovanje' koja se može koristiti kao ciljna varijabla
print(data[['StudentId', 'Interesovanje']].head(40))

# Kodiranje kategorijalnih varijabli
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

# Selektovanje karakteristika i ciljne varijable
X = data.drop(columns=['Interesovanje'])  # Navesti sve kolone osim ciljne varijable
y = data['Interesovanje']

# Podela podataka na trening i test skupove
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treniranje multinomijalne logističke regresije
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# Predikcija na test skupu
y_pred = model.predict(X_test)

# Evaluacija modela
print(classification_report(y_test, y_pred))
print("Tačnost modela:", accuracy_score(y_test, y_pred))

# Dodatno: Prikaz koeficijenata modela
coefficients = pd.DataFrame(model.coef_, columns=X.columns)
print(coefficients)
