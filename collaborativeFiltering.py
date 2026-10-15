import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

print("Loading full dataset in chunks to manage memory...")

# Read recommendations file in chunks
chunk_size = 2000000
chunks = []

for chunk in pd.read_csv("recommendations.csv", chunksize=chunk_size):
  # Filter for positive recommendations to save memory
  filtered_chunk = chunk[chunk["is_recommended"] == True][
      ["app_id", "user_id", "is_recommended"]
  ]
  chunks.append(filtered_chunk)

# Combine filtered chunks
rec_df = pd.concat(chunks, ignore_index=True)
games_df = pd.read_csv("games.csv")

print(f"Total positive interactions loaded: {len(rec_df):,}")

# Convert user_id and app_id into categorical integer codes for matrix indexing
rec_df["user_code"] = rec_df["user_id"].astype("category").cat.codes
rec_df["game_code"] = rec_df["app_id"].astype("category").cat.codes

# Map codes back to actual app_ids
app_id_mapping = (
    rec_df[["game_code", "app_id"]]
    .drop_duplicates()
    .set_index("game_code")["app_id"]
    .to_dict()
)

# Sparse CSR Matrix
print("Building sparse interaction matrix...")
interaction_matrix = csr_matrix(
    (
        rec_df["is_recommended"].astype(int),
        (rec_df["game_code"], rec_df["user_code"]),
    )
)

print(f"Sparse Matrix shape (Games x Users): {interaction_matrix.shape}")

print("Calculating item-item cosine similarity...")
# Cosine similarity 
game_similarity = cosine_similarity(interaction_matrix, dense_output=False)

def recommend_full_scale(game_title, top_n=5):
  # Split query into words
  query_terms = game_title.strip().split()

  # mask where all terms must appear in the title
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

  # Sort by user reviews 
  if "user_reviews" in game_matches.columns:
    game_matches = game_matches.sort_values(by="user_reviews", ascending=False)

  print(f'\nFound main game matches for "{game_title}":')
  for idx, row in game_matches.head(5).iterrows():
    print(f" - ID: {row['app_id']} | Title: {row['title']}")

  # Pick first match that exists in sparse interaction matrix
  target_app_id = None
  matched_title = None
  code = None

  for _, row in game_matches.iterrows():
    candidate_id = row["app_id"]
    matching_codes = [
        c for c, aid in app_id_mapping.items() if aid == candidate_id
    ]
    if matching_codes:
      target_app_id = candidate_id
      matched_title = row["title"]
      code = matching_codes[0]
      break

  if code is None:
    print(f'\nNone of the matches for "{game_title}" have reviews in the matrix.')
    return

  print(f'\n=== Full-Dataset Recommendations for "{matched_title}" ===')
  scores = game_similarity.getrow(code).toarray().flatten()
  top_indices = scores.argsort()[::-1][1 : top_n + 1]

  for i, rec_idx in enumerate(top_indices, 1):
    rec_app_id = app_id_mapping.get(rec_idx)
    rec_title_match = games_df[games_df["app_id"] == rec_app_id]
    if not rec_title_match.empty:
      rec_title = rec_title_match.iloc[0]["title"]
      score = scores[rec_idx]
      print(f"{i}. {rec_title} (Similarity Score: {score:.4f})")


# CLI Loop 
if __name__ == "__main__":
  print("\n" + "=" * 50)
  print("STEAM RECOMMENDATION ENGINE")
  print("=" * 50)

  while True:
    user_query = input(
        "\nEnter a game title to search (or type 'exit' to quit): "
    ).strip()
    if user_query.lower() == "exit":
      print("Exiting recommendation engine.")
      break

    recommend_full_scale(user_query, top_n=5)
    print("-" * 50)