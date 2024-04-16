import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
from scipy.io import arff
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

# Διάβασμα του αρχείου CSV με επιλεγμένα πεδία
selected_columns = ["date_min", "date_max", "text", "date_circa", "id" ,"region_sub_id"]  # Προσθήκη του πεδίου date_circa
df = pd.read_csv("iphi2802.csv", usecols=selected_columns, encoding="ISO-8859-1")

# Μετατροπή του πεδίου κειμένου σε διανυσματική μορφή TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X_text_tfidf = vectorizer.fit_transform(df['text'])

# Εύρεση μέγιστου tf-idf αριθμού για κάθε γραμμή
max_tfidf = X_text_tfidf.max(axis=1).toarray()

# Ενημέρωση του X_text_tfidf με τον μέγιστο tf-idf αριθμό
X_text_tfidf = csr_matrix(max_tfidf)

# Αφαίρεση του πεδίου κειμένου από το DataFrame
df.drop(columns=['text'], inplace=True)

# Συνδυασμός των υπάρχοντων αριθμητικών πεδίων με το TF-IDF πεδίο κειμένου
X_combined = csr_matrix(df.drop(columns=['date_circa']).values)

# Εγγραφή στο αρχείο ARFF
with open("output11.arff", "w", encoding="utf-8") as f:
    f.write("@relation 'myweka'\n")

    f.write("@attribute id NUMERIC\n")

    f.write("@attribute region_sub_id NUMERIC\n")
    
    # Γράφουμε το attribute για το πεδίο date_min
    f.write("@attribute date_min NUMERIC\n")
    
    # Γράφουμε το attribute για το πεδίο date_max
    f.write("@attribute date_max NUMERIC\n")
    
    # Γράφουμε το attribute για το πεδίο text
    f.write("@attribute text NUMERIC\n")
    
    # Γράφουμε το attribute για το πεδίο date_circa
    f.write("@attribute date_circa NUMERIC\n")

    
    f.write("\n@data\n")
    
    for i in range(X_combined.shape[0]):
        row_data = ",".join(map(str, X_combined[i].toarray().flatten().tolist() + [X_text_tfidf[i].toarray().flatten().tolist()[0]] + [df['date_circa'].iloc[i]]))
        f.write(f"{row_data}\n")







































