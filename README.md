# Scholarship Recommendation System

This is a Flask-based API for a scholarship recommendation system. It uses machine learning techniques to match users with relevant scholarships based on their profiles. The scholarship data is stored in Firebase Storage.

## Setup

1. Clone this repository.
2. Install the required packages: `pip install -r requirements.txt`
3. Set up your Firebase credentials and storage bucket as environment variables:
   ```
   export FIREBASE_CREDENTIALS=/path/to/your/credentials.json
   export FIREBASE_STORAGE_BUCKET=your-firebase-storage-bucket-name
   ```
4. Ensure your `scholarships.csv` file is uploaded to Firebase Storage in the `Scholarships/` folder.
5. Run the application: `python app.py`

## API Endpoints

- `GET /health`: Health check endpoint
- `POST /recommendations`: Get scholarship recommendations for a user
- `POST /process_all_users`: Process recommendations for all users

## Deployment

This application is designed to be deployed on Render. To deploy:

1. Push your code to a GitHub repository.
2. Create a new Web Service on Render.
3. Connect your GitHub repository to Render.
4. Set the environment variables in Render:
   - `FIREBASE_CREDENTIALS`: Your Firebase credentials JSON (as a string)
   - `FIREBASE_STORAGE_BUCKET`: Your Firebase Storage bucket name
   - `PORT`: The port number (Render will set this automatically)
5. Deploy the application.

For more detailed instructions, refer to Render's documentation.

## Updating Scholarship Data

To update the scholarship data:

1. Prepare your new `scholarships.csv` file.
2. Upload it to Firebase Storage in the `Scholarships/` folder, replacing the existing file.
3. The application will automatically use the updated data on the next restart or when processing new recommendations.

Remember to secure your API endpoints in a production environment and implement proper error handling and rate limiting.