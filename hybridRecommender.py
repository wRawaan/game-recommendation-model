import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

print("--- INITIALIZING HYBRID STEAM RECOMMENDATION ENGINE ---")

# 1. Load Datasets & Build Collaborative Filtering Matrix
print("[1/4] Loading full dataset in chunks...")
chunk_size = 2000000
chunks = []
for chunk in pd.read_csv("recommendations.csv", chunksize=chunk_size):
  filtered_chunk = chunk[chunk["is_recommended"] == True][
      ["app_id", "user_id", "is_recommended"]
  ]
  chunks.append(filtered_chunk)

rec_df = pd.concat(chunks, ignore_index=True)
games_df = pd.read_csv("games.csv").drop_duplicates(subset=["app_id"])

rec_df["user_code"] = rec_df["user_id"].astype("category").cat.codes
rec_df["game_code"] = rec_df["app_id"].astype("category").cat.codes

app_id_mapping = (
    rec_df[["game_code", "app_id"]]
    .drop_duplicates()
    .set_index("game_code")["app_id"]
    .to_dict()
)

print("Building sparse interaction matrix for 35M+ rows...")
interaction_matrix = csr_matrix(
    (
        rec_df["is_recommended"].astype(int),
        (rec_df["game_code"], rec_df["user_code"]),
    )
)
game_cf_similarity = cosine_similarity(interaction_matrix, dense_output=False)

# 2. Build Content Embeddings Model
print("[2/4] Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# prepare text feature for each game in metadata
games_df["text_feature"] = games_df["title"].fillna("")
print("Generating content embeddings for all games...")
game_embeddings = embed_model.encode(
    games_df["text_feature"].tolist(), show_progress_bar=False, batch_size=256
)
# Compute content similarity matrix
game_content_similarity = cosine_similarity(game_embeddings)

print(
    "[3/4] Crunching big-data metrics locally offline."
)


# 3. Hybrid Recommendation Function
def recommend_hybrid(game_title, top_n=5, cf_weight=0.7):
  emb_weight = 1.0 - cf_weight

  query_terms = game_title.strip().split()
  mask = True
  for term in query_terms:
    mask = mask & games_df["title"].str.contains(term, case=False, na=False)

  game_matches = games_df[
      mask
      & ~games_df["title"].str.contains(
          "Soundtrack|Costume|DLC|Pack|Unlock|Season Pass|BlackCell",
          case=False,
          na=False,
      )
  ]

  if len(game_matches) == 0:
    print(f"Game matching '{game_title}' not found in dataset.")
    return

  if "user_reviews" in game_matches.columns:
    game_matches = game_matches.sort_values(by="user_reviews", ascending=False)

  # Find the first match present in collaborative filtering matrix
  target_app_id = None
  matched_title = None
  cf_idx = None

  for _, row in game_matches.iterrows():
    candidate_id = row["app_id"]
    matching_codes = [
        c for c, aid in app_id_mapping.items() if aid == candidate_id
    ]
    if matching_codes:
      target_app_id = candidate_id
      matched_title = row["title"]
      cf_idx = matching_codes[0]
      break

  if cf_idx is None:
    print(
        f"'{game_matches.iloc[0]['title']}' exists in metadata, but lacks"
        " review history."
    )
    return

  # Find row index in the content dataframe
  content_idx = games_df[games_df["app_id"] == target_app_id].index[0]

  print(
      f'\n=== Recommendations for "{matched_title}" (CF weight:'
      f" {cf_weight}) ==="
  )

  # Get scores from both systems
  cf_scores = game_cf_similarity.getrow(cf_idx).toarray().flatten()

  # Map content similarity back to the sparse matrix app order
  hybrid_scores = {}
  for code, app_id in app_id_mapping.items():
    if code == cf_idx:
      continue
    # Find matching index in games_df
    match_rows = games_df[games_df["app_id"] == app_id]
    if match_rows.empty:
      continue
    c_idx = match_rows.index[0]

    s_cf = cf_scores[code]
    s_emb = game_content_similarity[content_idx][c_idx]

    # Weighted Hybrid Blend Formula
    s_final = (cf_weight * s_cf) + (emb_weight * s_emb)
    hybrid_scores[app_id] = (s_final, s_cf, s_emb)

  # Sort by final hybrid score descending
  sorted_recommendations = sorted(
      hybrid_scores.items(), key=lambda x: x[1][0], reverse=True
  )[:top_n]

  for i, (rec_app_id, (s_final, s_cf, s_emb)) in enumerate(
      sorted_recommendations, 1
  ):
    rec_title = games_df[games_df["app_id"] == rec_app_id].iloc[0]["title"]
    print(
        f"{i}. {rec_title} | Hybrid Score: {s_final:.4f} (CF: {s_cf:.3f},"
        f" Embed: {s_emb:.3f})"
    )


# 4. Interactive CLI Loop
if __name__ == "__main__":
  print("\n" + "=" * 55)
  print(" STEAM RECOMMENDATION ENGINE ")
  print("=" * 55)

  while True:
    user_query = input(
        "\nEnter a game title to search (or type 'exit' to quit): "
    ).strip()
    if user_query.lower() == "exit":
      print("Exiting hybrid engine.")
      break

    recommend_hybrid(user_query, top_n=5, cf_weight=0.7)
    print("-" * 55)