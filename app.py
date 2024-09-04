from flask import Flask, render_template, request
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

books = pd.read_csv('../books2.csv')

# Drop null or unused coloumns 
books = books.drop(['asin','ISBN10','answered_questions','availability','currency','date_first_available','delivery','department','description','discount',
                  'domain','features','format','images_count','initial_price','item_weight','manufacturer','model_number','plus_content','product_dimensions',
                  'reviews_count','root_bs_rank','seller_id','seller_name','timestamp','upc','video','video_count','best_sellers_rank','buybox_seller',
                  'number_of_sellers','colors','image'],axis=1)

# Drop duplicates and null values
books = books.drop_duplicates()
books = books.dropna()
books = books.reset_index(drop=True)

# create a TF-IDF matrix for catigories 
tfid = TfidfVectorizer()
tfid_matrix = tfid.fit_transform(books['categories'])


# Get the cosine similarity matrix
similarity = cosine_similarity(tfid_matrix)

# A function for generate top five simialr books based on category
def recommender(title):

    # Find the index of the book 
    index = books[books['title'].str.contains(title, case=False)].index[0]
    
    # Get the cosine similarity scores for the book
    similarity_scores = sorted(list(enumerate(similarity[index])), key=lambda x: x[1], reverse=True)

    # Get the top five books 
    top_books = [i[0] for i in similarity_scores[:5]]
    
    # Return the most similar books
    return books.loc[top_books].drop_duplicates()

# A function to search the given book title or author
def search(title: str):
   return books[books['title'].str.contains(title, case=False)]

    
app = Flask(__name__)

@app.route("/", methods=["POST", "GET"])
def reco():
    if request.method == 'POST':
        if request.form.get("text1") !='':
            text1 = request.form.get("text1")
        return render_template('recommender.html',  books=recommender(text1).values.tolist(),text1 = text1)
    else:
        return render_template('recommender.html', books=books.values.tolist(),text1 = None)


@app.route("/search", methods=["POST", "GET"])
def search1():
    if request.method == "POST":
        return render_template('search.html', books=search(request.form.get("text2")).values.tolist())
    else:
        return render_template('search.html', books=books.values.tolist())
        

if __name__ == '__main__':
    app.run(debug=True)

