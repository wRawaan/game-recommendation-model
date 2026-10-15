# 🎮 Steam Big-Data Hybrid Recommendation Engine

An offline game recommendation system built in Python that processes **35M+ user interactions** from **12.6M users** on consumer hardware. It combines item-item collaborative filtering (behavioral co-play patterns) with dense semantic embeddings (`all-MiniLM-L6-v2`) in a hybrid scoring architecture, and avoids out-of-memory failures through chunked streaming and sparse data structures.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-Sparse%20Matrices-8CAAE6?logo=scipy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Cosine%20Similarity-F7931E?logo=scikitlearn&logoColor=white)
![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-MiniLM-yellow)

---

## Table of Contents

- [Key Features](#key-features)
- [Architecture and Scoring](#architecture-and-scoring)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Tuning the Hybrid Weight](#tuning-the-hybrid-weight)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Key Features

- **Big-data scalability**: Streams the dataset in chunks (`pd.read_csv` with chunk iteration) so 35M+ rows can be processed without running out of memory.
- **Memory-efficient storage**: Uses SciPy sparse matrices (`csr_matrix`) in place of dense Pandas pivot tables, storing only active interactions and cutting RAM overhead by over 95%.
- **Hybrid scoring engine**: Blends collaborative similarity with transformer-based content embeddings, helping surface relevant games that have little interaction data and improving cross-genre discovery.
- **Smart title filtering**: Regex filtering removes DLCs, soundtracks, and costume packs so mainline games rank first, sorted by popularity and review count.
- **Interactive CLI**: Search any game by title and get ranked recommendations with per-component score breakdowns.

---

## Architecture and Scoring

For each candidate game, the engine computes a weighted hybrid score:

```text
Final Score = (α × Collaborative Score) + ((1 − α) × Embedding Score)
```

where `α` is the collaborative-filtering weight (default `0.7`).

1. **Collaborative filtering**: Measures behavioral overlap between games using cosine similarity over sparse user-recommendation vectors from millions of Steam users.
2. **Content embeddings**: Encodes game titles with `sentence-transformers` (`all-MiniLM-L6-v2`) to capture semantic similarity independent of user behavior.
3. **Hybrid ranking**: Combines both signals into a single score and returns the top-N results.

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.12 |
| Data processing | Pandas, NumPy, SciPy (`csr_matrix`) |
| ML and NLP | scikit-learn (`cosine_similarity`), Sentence-Transformers |
| Version control | Git and GitHub |

---

## Project Structure

```text
game-recommender/
│
├── hybridRecommender.py       # Main hybrid engine with interactive CLI loop
├── collaborativeFiltering.py  # Standalone sparse-matrix collaborative filter
├── Embeddings.py              # Semantic transformer embedding module
├── .gitignore                 # Excludes local datasets and cache folders
└── README.md                  # Project documentation
```

---

## Getting Started

### Prerequisites

- Python 3.12
- `pip`
- Enough free disk space for the dataset (several GB when extracted)

### 1. Clone the repository

```bash
git clone https://github.com/your-wRawaan/game-recommendation-model.git
cd game-recommendation-model
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn scipy sentence-transformers
```

### 3. Download the dataset

This project uses the Kaggle dataset [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) by Anton Kozyriev.

1. Download and extract the dataset files `games.csv` and `recommendations.csv`.
2. Place both files in the project root directory.

> **Note:** CSV files are excluded from Git tracking because of their size.

### 4. Run the engine

```bash
python hybridRecommender.py
```

---

## Usage

Once the sparse matrix is built and similarities are computed, you can query any game interactively.

> The output below is illustrative. Scores will vary with your data and settings.

```text
=======================================================
 STEAM HYBRID ENGINE
=======================================================

Enter a game title to search (or type 'exit' to quit): Phasmophobia

Found main game matches for "Phasmophobia":
 - ID: 739630 | Title: Phasmophobia

=== Recommendations for "Phasmophobia" (CF weight: 0.7) ===
1. The Forest        | Hybrid Score: 0.1245 (CF: 0.039, Embed: 0.340)
2. Dead by Daylight  | Hybrid Score: 0.1189 (CF: 0.032, Embed: 0.310)
3. Among Us          | Hybrid Score: 0.1102 (CF: 0.043, Embed: 0.150)
4. DEVOUR            | Hybrid Score: 0.0987 (CF: 0.034, Embed: 0.230)
5. Raft              | Hybrid Score: 0.0891 (CF: 0.025, Embed: 0.280)
```

Type `exit` at the prompt to quit.

---

## Tuning the Hybrid Weight

The collaborative-filtering weight `α` controls how the two signals are balanced:

| α value | Behavior |
|---------|----------|
| Closer to `1.0` | Favors games commonly played by the same users (behavioral similarity) |
| Closer to `0.0` | Favors games with similar titles and themes (semantic similarity) |
| `0.7` (default) | Balanced blend that leans on player behavior |

---

## Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push the branch (`git push origin feature/your-feature`)
5. Open a pull request

---

## Acknowledgements

- Dataset: [Game Recommendations on Steam](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) by Anton Kozyriev
- Embedding model: [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via Sentence-Transformers

---

## License

This project is licensed under the [MIT License](LICENSE).