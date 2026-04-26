from flask import Flask, render_template, jsonify

dataset_categories = [
    "armed conflicts and attacks",
	"law and crime",
	"disasters and accidents",
	"politics and elections",
	"international relations",
	"business and economy",
	"sports",
	"science and technology",
	"health and environment",
	"arts and culture"
]

srv = Flask(__name__)

@srv.route('/')
def index():
    return render_template("index.html")

@srv.route('/chat.html')
def chat():
    return render_template("chat.html")

@srv.route('/api/categories')
def categories():
    return jsonify(dataset_categories)

if __name__ == "__main__":
    srv.run(debug=True)