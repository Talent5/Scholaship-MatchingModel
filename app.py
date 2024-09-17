import os
from flask import Flask, request, jsonify
from scholarship_recommender import ScholarshipRecommender
import firebase_admin
from firebase_admin import credentials, firestore, storage

app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate(os.environ.get('FIREBASE_CREDENTIALS'))
firebase_admin.initialize_app(cred, {
    'storageBucket': os.environ.get('FIREBASE_STORAGE_BUCKET')
})
db = firestore.client()
bucket = storage.bucket()

# Initialize ScholarshipRecommender
recommender = ScholarshipRecommender(db, bucket)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/recommendations', methods=['POST'])
def get_recommendations():
    user_id = request.json.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID is required"}), 400

    try:
        user_ref = db.collection('users').document(user_id)
        user = user_ref.get().to_dict()
        if not user:
            return jsonify({"error": "User not found"}), 404

        matches = recommender.find_matching_scholarships(user)
        recommendations = [
            {
                "title": scholarship['title'],
                "university": scholarship['university'],
                "score": float(score),
                "application_link": scholarship['application_link-href'],
                "deadline": str(scholarship['deadline']),
                "amount": scholarship['Grant'],
                "eligibility": scholarship['Eligibility'],
                "description": scholarship['Description']
            }
            for scholarship, score in matches
        ]

        return jsonify({"recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/process_all_users', methods=['POST'])
def process_all_users():
    try:
        recommender.process_users()
        return jsonify({"message": "All users processed successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
