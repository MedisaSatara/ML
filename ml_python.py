import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer 

# Učitavanje podataka
data = pd.read_csv('IB170012_MedisaSatara_podaci_ML.csv', encoding='latin1')

# Uklonite stupce s nedostajućim vrijednostima ili ih obradite na drugi način
data = data.drop(['PolozioDatum_P-224', 'SklonostKaPredmetu'], axis=1)

imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Pretprocesiranje datuma (pretpostavimo da su datumi već u odgovarajućem formatu)
data['GodinaPolaganja_P-12'] = pd.to_datetime(data['PolozioDatum_P-12']).dt.year
data['GodinaPolaganja_P-149'] = pd.to_datetime(data['PolozioDatum_P-149']).dt.year
data['GodinaPolaganja_P-175'] = pd.to_datetime(data['PolozioDatum_P-175']).dt.year
data['GodinaPolaganja_P-176'] = pd.to_datetime(data['PolozioDatum_P-176']).dt.year
data['GodinaPolaganja_P-218'] = pd.to_datetime(data['PolozioDatum_P-218']).dt.year
data['GodinaPolaganja_P-219'] = pd.to_datetime(data['PolozioDatum_P-219']).dt.year
data['GodinaPolaganja_P-220'] = pd.to_datetime(data['PolozioDatum_P-220']).dt.year
data['GodinaPolaganja_P-221'] = pd.to_datetime(data['PolozioDatum_P-221']).dt.year
data['GodinaPolaganja_P-222'] = pd.to_datetime(data['PolozioDatum_P-222']).dt.year
data['GodinaPolaganja_P-246'] = pd.to_datetime(data['PolozioDatum_P-246']).dt.year
data['GodinaPolaganja_P-58'] = pd.to_datetime(data['PolozioDatum_P-58']).dt.year
data['GodinaPolaganja_P-223'] = pd.to_datetime(data['PolozioDatum_P-223']).dt.year
data['GodinaPolaganja_P-8'] = pd.to_datetime(data['PolozioDatum_P-8']).dt.year
data['GodinaPolaganja_P-142'] = pd.to_datetime(data['PolozioDatum_P-142']).dt.year
data['GodinaPolaganja_P-143'] = pd.to_datetime(data['PolozioDatum_P-143']).dt.year
data['GodinaPolaganja_P-104'] = pd.to_datetime(data['PolozioDatum_P-104']).dt.year
data['GodinaPolaganja_P-140'] = pd.to_datetime(data['PolozioDatum_P-140']).dt.year
data['GodinaPolaganja_P-141'] = pd.to_datetime(data['PolozioDatum_P-141']).dt.year
data['GodinaPolaganja_P-178'] = pd.to_datetime(data['PolozioDatum_P-178']).dt.year

# Odabir karakteristika i labela (primer za Matematiku)
features = data[['P-12', 'GodinaPolaganja_P-12',
                 'P-149', 'GodinaPolaganja_P-149',
                 'P-175', 'GodinaPolaganja_P-175',
                 'P-176', 'GodinaPolaganja_P-176',
                 'P-218', 'GodinaPolaganja_P-218',
                 'P-219', 'GodinaPolaganja_P-219',
                 'P-220', 'GodinaPolaganja_P-220',
                 'P-221', 'GodinaPolaganja_P-221',
                 'P-222', 'GodinaPolaganja_P-222',
                 'P-246', 'GodinaPolaganja_P-246',
                 'P-58', 'GodinaPolaganja_P-58',
                 'P-223', 'GodinaPolaganja_P-223',
                 'P-8', 'GodinaPolaganja_P-8',
                 'P-142', 'GodinaPolaganja_P-142',
                 'P-143', 'GodinaPolaganja_P-143',
                 'P-104', 'GodinaPolaganja_P-104',
                 'P-140', 'GodinaPolaganja_P-140',
                 'P-141', 'GodinaPolaganja_P-141',
                 'P-178', 'GodinaPolaganja_P-178', 
                 'Godina', 'DatumUpisa', 'SrednjaSkola', 
                 'GodinaZavrsetka', 'NacinStudiranja', 'Spol', 'Diplomirao'
                 ]]



# Preprocesiranje kategorijskih podataka
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['P-12', 'GodinaPolaganja_P-12',
                                   'P-149', 'GodinaPolaganja_P-149',
                                   'P-175', 'GodinaPolaganja_P-175',
                                   'P-176', 'GodinaPolaganja_P-176',
                                   'P-218', 'GodinaPolaganja_P-218',
                                   'P-219', 'GodinaPolaganja_P-219',
                                   'P-220', 'GodinaPolaganja_P-220',
                                   'P-221', 'GodinaPolaganja_P-221',
                                   'P-222', 'GodinaPolaganja_P-222',
                                   'P-246', 'GodinaPolaganja_P-246',
                                   'P-58', 'GodinaPolaganja_P-58',
                                   'P-223', 'GodinaPolaganja_P-223',
                                   'P-8', 'GodinaPolaganja_P-8',
                                   'P-142', 'GodinaPolaganja_P-142',
                                   'P-143', 'GodinaPolaganja_P-143',
                                   'P-104', 'GodinaPolaganja_P-104',
                                   'P-140', 'GodinaPolaganja_P-140',
                                   'P-141', 'GodinaPolaganja_P-141',
                                   'P-178', 'GodinaPolaganja_P-178',
                                   ]),
        ('cat', OneHotEncoder(), ['Godina', 'DatumUpisa', 'SrednjaSkola', 'GodinaZavrsetka', 'NacinStudiranja', 'Spol', 'Diplomirao'])
    ])

# Podela na trenirajući i testirajući skup
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Kreiranje pipeline-a za multinomijalnu logističku regresiju
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(multi_class='multinomial', max_iter=1000))
])

# Treniranje modela
model.fit(X_train, y_train)

# Evaluacija modela
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
