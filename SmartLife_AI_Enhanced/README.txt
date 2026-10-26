# SmartLife AI Enhanced — All-in-One Recommendation Agent

## What is included
- Four modules: Products, Movies, Career & Courses, and Travel
- Larger curated demo catalogs (not live or authoritative external datasets)
- Hybrid ranking: query relevance + saved-profile affinity + ratings + popularity
- Local user profiles saved to `data/profiles.json`
- Like/dislike feedback saved to `data/feedback.csv`
- Precision@K, Recall@K, and judged catalog coverage
- Optional generative AI explanations via the OpenAI Responses API
- Streamlit interface with CSV export

## Run on Windows
1. Extract this ZIP.
2. Open the extracted `SmartLife_AI_Enhanced` folder in VS Code or any editor.
3. Open a terminal in that folder.
4. Install packages:
   `py -m pip install -r requirements.txt`
5. Start the app:
   `py -m streamlit run app.py`
6. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Enable generative AI explanations (optional)
You need an OpenAI API key and API access; API usage may cost money.
1. Create an API key in your OpenAI API dashboard.
2. In Windows Command Prompt, run:
   `setx OPENAI_API_KEY "YOUR_API_KEY_HERE"`
3. Close and reopen the terminal (setx affects future terminals).
4. Start Streamlit again.
5. In a recommendation result, open "Why these recommendations?" and click "Generate AI explanation".
6. Optional model override: set the `OPENAI_MODEL` environment variable to a model available to your API account.

Never paste your API key into `app.py`, screenshots, or a public repository. The app reads it from an environment variable. If no key is configured, all recommendation and profile features still work; only the generative explanation feature is unavailable.

## Algorithm
A hybrid weighted ranking combines:
- TF-IDF/cosine similarity between current user request and item descriptions
- Profile affinity based on saved interests and previously liked items
- Normalized catalog rating
- Normalized popularity

The weights are changed when no profile information exists. The hybrid score is a ranking score, not a probability.

## Data note
The catalog CSV files are curated educational demonstration data. Some ratings, popularity values, prices, and budgets are illustrative. Before formal submission, document that clearly or replace the catalogs with well-sourced datasets whose licenses permit your use.

## Evaluation note
Precision@K and Recall@K use explicit likes as relevant judgments. Unrated items are not assumed to be irrelevant; metrics are preliminary and depend on how representative the feedback is.

UPGRADE NOTES (v2)
- Refreshed vibrant gradient interface, metric cards, tabs, sidebar and controls.
- Expanded recommendation count selector from 3-12 to 3-25.
- Added selectable ranking strategies: Balanced, Best match, Top rated, Popular picks, Personalized, and Budget friendly.
- Ranking strategy changes the hybrid scoring weights; budget-first ranking is applied to products and travel when cost fields exist.
- Existing catalogs remain educational demo catalogs (130 curated sample records), not live or independently verified datasets. Do not describe their values as real-time market prices, live availability, or authoritative ratings.
