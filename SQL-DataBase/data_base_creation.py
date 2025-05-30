# -*- coding: utf-8 -*-
"""Data Base-Creation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1-OtDUNQrUBKRVFaEFqI8xx7WuqWvQ2ZS

#### Import Librires
"""

import pickle
import sqlite3

"""#### Unpickling the saved data"""

# Load pickled data
with open('/content/documents2.pkl', 'rb') as f:
    documents = pickle.load(f)

# Optional: Check the first document
print(type(documents[0]))
print(documents[0])

"""#### Creating the data base"""

conn = sqlite3.connect('/content/itc_documents1.db')
cursor = conn.cursor()

# Create the table
cursor.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    content TEXT,
    source TEXT
)
''')

"""#### Insert each Document into the database"""

for doc in documents:
    content = doc.page_content
    source = doc.metadata.get("source")

    cursor.execute("""
        INSERT INTO documents (content, source)
        VALUES (?, ?)
    """, (content, source))

conn.commit()

"""#### Fetch and display a few rows to confirm"""

cursor.execute("SELECT id, source, substr(content, 1, 100) FROM documents LIMIT 4")
rows = cursor.fetchall()

for row in rows:
    print(f"ID: {row[0]}, Source: {row[1]}, Content Preview: {row[2]}")

# Close the connection
cursor.close()
conn.close()