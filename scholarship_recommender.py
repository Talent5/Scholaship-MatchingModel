import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import re
from datetime import datetime
import uuid
import logging
from typing import List, Dict, Tuple, Any
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ScholarshipRecommender:
    def __init__(self, db, bucket):
        self.db = db
        self.bucket = bucket
        self.scholarships = self._load_and_clean_scholarships()
        self.feature_matrix, self.tfidf = self._create_feature_matrix()
        self.kmeans = self._cluster_scholarships()
        self.rf_classifier = self._train_rf_classifier()

    def _load_and_clean_scholarships(self) -> pd.DataFrame:
        try:
            blob = self.bucket.blob('Scholarships/scholarships.csv')
            content = blob.download_as_string()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            df = self._clean_scholarships(df)
            df = self._remove_duplicates_and_similar(df)
            logging.info(f"Loaded, cleaned, and deduplicated. Remaining scholarships: {len(df)}")
            return df
        except Exception as e:
            logging.error(f"Failed to load and clean scholarships: {str(e)}")
            raise

    def _clean_scholarships(self, df: pd.DataFrame) -> pd.DataFrame:
        def clean_text(text: str) -> str:
            if pd.isna(text):
                return ''
            text = re.sub(r'[^\w\s]', ' ', str(text))
            return text.lower().strip()

        text_columns = ['title', 'field_of_study', 'benefits', 'location', 'university', 'About', 'Description', 'Applicable_programmes', 'Eligibility']
        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].apply(clean_text)

        if 'deadline' in df.columns:
            df['deadline'] = pd.to_datetime(df['deadline'], errors='coerce')

        df = df.fillna('')
        return df

    def _remove_duplicates_and_similar(self, df: pd.DataFrame, similarity_threshold: float = 0.95) -> pd.DataFrame:
        df = df.drop_duplicates()
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['title'] + ' ' + df['Description'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        distance_matrix = np.clip(1 - cosine_sim, 0, None)

        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=1-similarity_threshold, min_samples=2, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)

        unique_scholarships = df[labels == -1]
        for cluster in set(labels):
            if cluster != -1:
                cluster_scholarships = df[labels == cluster]
                unique_scholarships = pd.concat([unique_scholarships, cluster_scholarships.iloc[[0]]])

        return unique_scholarships.reset_index(drop=True)

    def _create_feature_matrix(self) -> Tuple[np.ndarray, TfidfVectorizer]:
        features = ['field_of_study', 'location', 'university', 'About', 'Description', 'Applicable_programmes', 'Eligibility']
        tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
        feature_matrix = tfidf.fit_transform(self.scholarships[features].apply(lambda x: ' '.join(x), axis=1))
        
        svd = TruncatedSVD(n_components=100, random_state=42)
        feature_matrix_reduced = svd.fit_transform(feature_matrix)
        
        scaler = StandardScaler()
        feature_matrix_normalized = scaler.fit_transform(feature_matrix_reduced)
        
        return feature_matrix_normalized, tfidf

    def _cluster_scholarships(self) -> KMeans:
        kmeans = KMeans(n_clusters=10, random_state=42)
        self.scholarships['cluster'] = kmeans.fit_predict(self.feature_matrix)
        return kmeans

    def _train_rf_classifier(self) -> RandomForestClassifier:
        y = np.random.randint(0, 2, size=len(self.scholarships))
        X_train, X_test, y_train, y_test = train_test_split(self.feature_matrix, y, test_size=0.2, random_state=42)
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        y_pred = rf.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        logging.info(f"Random Forest Classifier Metrics - Precision: {precision:.2f}, Recall: {recall:.2f}, F1: {f1:.2f}")
        
        return rf

    def calculate_similarity(self, user_profile: Dict[str, Any]) -> np.ndarray:
        user_text = ' '.join([
            user_profile.get('intendedFieldOfStudy', ''),
            user_profile.get('preferredLocation', ''),
            user_profile.get('educationLevel', ''),
            user_profile.get('courseOfStudy', ''),
            user_profile.get('degreeType', ''),
            user_profile.get('financialNeed', ''),
            user_profile.get('incomeBracket', '')
        ])
        user_vector = self.tfidf.transform([user_text])
        user_vector_reduced = TruncatedSVD(n_components=100, random_state=42).fit_transform(user_vector)
        user_vector_normalized = StandardScaler().fit_transform(user_vector_reduced)
        return cosine_similarity(user_vector_normalized, self.feature_matrix)[0]

    def find_matching_scholarships(self, user_profile: Dict[str, Any], min_score: float = 0.3) -> List[Tuple[pd.Series, float]]:
        similarities = self.calculate_similarity(user_profile)
        rf_predictions = self.rf_classifier.predict_proba(self.feature_matrix)[:, 1]
        
        combined_scores = 0.7 * similarities + 0.3 * rf_predictions
        
        profile_fields = ['intendedFieldOfStudy', 'preferredLocation', 'educationLevel', 'courseOfStudy', 'degreeType', 'financialNeed', 'incomeBracket']
        profile_strength = sum(1 for field in profile_fields if user_profile.get(field)) / len(profile_fields)
        
        min_scholarships, max_scholarships = 5, 30
        num_scholarships = int(min_scholarships + (max_scholarships - min_scholarships) * profile_strength)
        similarity_threshold = max(min_score, 0.5 - (0.4 * profile_strength))
        
        matches = []
        for idx, score in enumerate(combined_scores):
            if score >= similarity_threshold:
                scholarship = self.scholarships.iloc[idx]
                matches.append((scholarship, score))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:num_scholarships]

    def save_recommendations(self, user_id: str, matches: List[Tuple[pd.Series, float]]) -> None:
        try:
            recommendations = []
            for scholarship, score in matches:
                unique_id = str(uuid.uuid4())
                recommendations.append({
                    'id': unique_id,
                    'title': scholarship.get('title', ''),
                    'deadline': str(scholarship.get('deadline', '')),
                    'amount': scholarship.get('Grant', ''),
                    'application_link': scholarship.get('application_link-href', ''),
                    'eligibility': scholarship.get('Eligibility', ''),
                    'description': scholarship.get('Description', ''),
                    'application_process': scholarship.get('application_process', ''),
                    'score': float(score),
                    'cluster': int(scholarship.get('cluster', -1))
                })

            self.db.collection('scholarship_recommendations').document(user_id).set({
                'recommendations': recommendations,
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            logging.info(f"Saved recommendations for user: {user_id}")
        except Exception as e:
            logging.error(f"Failed to save recommendations for user {user_id}: {str(e)}")

    def process_users(self, min_score: float = 0.15) -> None:
        try:
            users = self.get_all_users()
            for user in users:
                matches = self.find_matching_scholarships(user, min_score)
                self.save_recommendations(user['userId'], matches)
                logging.info(f"Processed recommendations for user: {user.get('firstName', '')} {user.get('lastName', '')}")
                logging.info(f"Number of scholarships recommended: {len(matches)}")
        except Exception as e:
            logging.error(f"Error processing users: {str(e)}")

    def get_all_users(self) -> List[Dict[str, Any]]:
        try:
            users_ref = self.db.collection('users')
            return [doc.to_dict() for doc in users_ref.stream()]
        except Exception as e:
            logging.error(f"Failed to get users: {str(e)}")
            return []