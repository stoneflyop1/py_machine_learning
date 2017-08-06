import sqlite3
import os
dest = 'movieclassifier'
if not os.path.exists(dest):
    os.makedirs(dest)
data_file = os.path.join(dest, 'reviews.sqlite')
conn = sqlite3.connect(data_file)
c = conn.cursor()
if not os.path.exists(data_file):
    c.execute(
        'CREATE TABLE review_db (review TEXT, sentiment INTEGER, date TEXT)'
    )
    example1 = 'I love this movie'
    c.execute(
        "INSERT INTO review_db (review, sentiment, date) VALUES"\
        " (?, ?, DATETIME('now'))", (example1, 1)
    )
    example2 = 'I disliked this movie'
    c.execute(
        "INSERT INTO review_db (review, sentiment, date) VALUES"\
        " (?, ?, DATETIME('now'))", (example2, 0)
    )
    conn.commit()
else:
    c.execute(
        "SELECT * FROM review_db WHERE date"\
        " BETWEEN '2017-01-01 00:00:00' AND DATETIME('now')"
    )
    result = c.fetchall()
    print(result)
conn.close()

