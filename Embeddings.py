import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

print('Loading datasets...')
# Load a subset of recommendations
rec_df = pd.read_csv('recommendations.csv', nrows=2000000)
games_df = pd.read_csv('games.csv')

# Filter for positive recommendations only 
rec_df = rec_df[rec_df['is_recommended'] == True]

# Keep games that have at least 50 recommendations in the sample
game_counts = rec_df["app_id"].value_counts()
popular_games = game_counts[game_counts >= 50].index
rec_df = rec_df[rec_df["app_id"].isin(popular_games)]

#pivot table 
interaction_matrix = rec_df.pivot_table(
    index='app_id', columns='user_id', values='is_recommended', aggfunc='max'
).fillna(0)

print(f'Matrix shape (Games x Users): {interaction_matrix.shape}')

print('Calculating item-item cosine similarity...')
game_similarity = cosine_similarity(interaction_matrix)

# Create a mapping from app_id to matrix index and vice versa
app_ids = interaction_matrix.index.tolist()
app_id_to_idx = {app_id: i for i, app_id in enumerate(app_ids)}

def collaborative_filtering(game_title, top_n=5):
  # Find app_id from games.csv using the title
  game_match = games_df[
      games_df['title'].str.contains(game_title, case=False, na=False)
  ]

  if len(game_match) == 0:
    print(f"Game '{game_title}' not found in games metadata.")
    return

  target_app_id = game_match.iloc[0]['app_id']
  matched_title = game_match.iloc[0]['title']

  if target_app_id not in app_id_to_idx:
    print(
        f"'{matched_title}' does not have enough user reviews in our sample"
        ' matrix yet.'
    )
    return

  idx = app_id_to_idx[target_app_id]
  print(f'\n--- Players who liked "{matched_title}" also recommended: ---')

  # Get similarity scores for this game against all other games in the matrix
  scores = game_similarity[idx]
  top_indices = scores.argsort()[::-1][1 : top_n + 1]

  for i, rec_idx in enumerate(top_indices, 1):
    rec_app_id = app_ids[rec_idx]
    rec_title_match = games_df[games_df['app_id'] == rec_app_id]
    if not rec_title_match.empty:
      rec_title = rec_title_match.iloc[0]['title']
      score = scores[rec_idx]
      print(f'{i}. {rec_title} (Behavior Match Score: {score:.2f})')

collaborative_filtering('Phasmophobia', top_n=5)