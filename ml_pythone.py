import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Učitavanje podataka
data = pd.read_csv('IB170012_MedisaSatara_podaci_ML.csv', encoding='latin1')

# Konverzija datuma u numerički format
data['DatumUpisa'] = pd.to_datetime(data['DatumUpisa']).dt.strftime('%Y%m%d')

# Pretvaranje kolona PolozioDatum_P-* u odgovarajući datumski format
for predmet in ['P-12', 'P-149', 'P-175', 'P-176', 'P-218', 'P-219', 'P-220', 'P-221', 'P-222', 'P-246', 'P-224', 'P-58', 'P-223', 'P-8', 'P-142', 'P-143', 'P-104', 'P-140', 'P-141', 'P-178']:
    data[f'PolozioDatum_{predmet}'] = pd.to_datetime(data[f'PolozioDatum_{predmet}'])
# Odabir karakteristika
features = data[['P-12', 'PolozioDatum_P-12', 
                 'P-149', 'PolozioDatum_P-149',
                 'P-175', 'PolozioDatum_P-175',
                 'P-176', 'PolozioDatum_P-176',
                 'P-218', 'PolozioDatum_P-218',
                 'P-219', 'PolozioDatum_P-219',
                 'P-220', 'PolozioDatum_P-220',
                 'P-221', 'PolozioDatum_P-221',
                 'P-222', 'PolozioDatum_P-222',
                 'P-246', 'PolozioDatum_P-246',
                 'P-224', 'PolozioDatum_P-224',
                 'P-58', 'PolozioDatum_P-58',
                 'P-223', 'PolozioDatum_P-223',
                 'P-8', 'PolozioDatum_P-8',
                 'P-142', 'PolozioDatum_P-142',
                 'P-143', 'PolozioDatum_P-143',
                 'P-104', 'PolozioDatum_P-104',
                 'P-140', 'PolozioDatum_P-140',
                 'P-141', 'PolozioDatum_P-141',
                 'P-178', 'PolozioDatum_P-178',
                 'Godina', 'DatumUpisa', 'SrednjaSkola', 'GodinaZavrsetka', 'NacinStudiranja', 'Diplomirao', 'Spol']]

# Normalizacija podataka
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Klasterska analiza
kmeans = KMeans(n_clusters=5, random_state=42)
cluster_labels = kmeans.fit_predict(features_scaled)

# Dodavanje rezultata klasterske analize u podatke
data['Interesovanje'] = cluster_labels

# Čuvanje rezultata
data.to_csv('predvidjena_interesovanja.csv', index=False)

print("Predvidjena interesovanja su sačuvana u 'predvidjena_interesovanja.csv' fajlu.")


