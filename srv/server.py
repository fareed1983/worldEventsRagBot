from flask import Flask, render_template, jsonify, request
from rag import start_chat_job, fetch_chat_job
import sys
sys.stdout.reconfigure(line_buffering=True)

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

@srv.route('/api/events-chat-job', methods=['POST'])
def post_events_chat_job():
	data = request.get_json()
	
	if not data:
		return jsonify({"error": "Missing JSON body"}), 400
	
	mode = data.get("mode", "rag")

	query = data.get("query")

	if not query:
		return jsonify({"error": "Missing required parameter: query"}), 400

	if mode != "raw" and mode != "rag":
		return jsonify({"error": "Bad value for mode parameter"}), 400
	
	category = data.get("category") # can be posted without a category
	job_id = start_chat_job(mode, category, query)

	if job_id is None:
		return jsonify({"error": "Server busy"}), 429

	return jsonify({"jobId": job_id})

@srv.route('/api/events-chat-job/<job_id>')
def get_chat_job(job_id):

	if not job_id:
		return jsonify({"error": "Missing required parameter: jobId"}), 400

	job = fetch_chat_job(job_id)

	if job is None:
		return jsonify({"error": "Job not found"}), 404

	return jsonify(job)


if __name__ == "__main__":
	srv.run(debug=True)