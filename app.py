# app.py
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, render_template, request, jsonify
from collections import defaultdict
from scipy import sparse

app = Flask(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_PATH  = "data/ml-latest-small"
MODEL_PATH = "best_lightgcn.pth"
EMBED_DIM  = 64
N_LAYERS   = 3
SEED       = 42

# ── Modèle LightGCN ───────────────────────────────────────────────────────────
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, n_layers, A_norm, dropout=0.1):
        super().__init__()
        self.n_users    = n_users
        self.n_items    = n_items
        self.n_layers   = n_layers
        self.dropout    = nn.Dropout(dropout)
        self.A_norm     = A_norm
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def _propagate(self):
        E0  = torch.cat([self.user_embed.weight, self.item_embed.weight], dim=0)
        E   = E0
        agg = [E0]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.A_norm, E)
            E = self.dropout(E)
            agg.append(E)
        E_final = torch.stack(agg, dim=0).mean(dim=0)
        return E_final[:self.n_users], E_final[self.n_users:]

    def forward(self, users, pos_items, neg_items):
        eu, ei     = self._propagate()
        u_emb      = eu[users]
        p_emb      = ei[pos_items]
        n_emb      = ei[neg_items]
        pos_scores = (u_emb * p_emb).sum(dim=1)
        neg_scores = (u_emb * n_emb).sum(dim=1)
        u0 = self.user_embed(users)
        p0 = self.item_embed(pos_items)
        n0 = self.item_embed(neg_items)
        return pos_scores, neg_scores, u0, p0, n0

    @torch.no_grad()
    def recommend(self, users):
        eu, ei = self._propagate()
        u_emb  = eu[users]
        return torch.matmul(u_emb, ei.T)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Chargement des données ────────────────────────────────────────────────────
def load_data():
    ratings_df = pd.read_csv(os.path.join(DATA_PATH, "ratings.csv"))
    movies_df  = pd.read_csv(os.path.join(DATA_PATH, "movies.csv"))
    movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))
    movies_df.set_index('movieId', inplace=True)

    # Binarisation
    df = ratings_df[ratings_df['rating'] >= 3.5][['userId', 'movieId']].copy()

    # Réindexation
    users_unique = df['userId'].unique()
    items_unique = df['movieId'].unique()
    user2idx = {int(u): i for i, u in enumerate(sorted(users_unique))}
    item2idx = {int(m): i for i, m in enumerate(sorted(items_unique))}
    idx2user = {i: int(u) for u, i in user2idx.items()}
    idx2item = {i: int(m) for m, i in item2idx.items()}

    df['user'] = df['userId'].map(user2idx)
    df['item'] = df['movieId'].map(item2idx)
    df = df[['user', 'item']].drop_duplicates().reset_index(drop=True)

    N_USERS = len(user2idx)
    N_ITEMS = len(item2idx)

    # Items positifs par utilisateur
    user_pos_items = defaultdict(set)
    for _, row in df.iterrows():
        user_pos_items[int(row['user'])].add(int(row['item']))

    # Matrice d'adjacence normalisée
    rows_r = df['user'].values
    cols_r = df['item'].values
    data_r = np.ones(len(rows_r), dtype=np.float32)
    R      = sparse.csr_matrix((data_r, (rows_r, cols_r)), shape=(N_USERS, N_ITEMS))

    N        = N_USERS + N_ITEMS
    zeros_uu = sparse.csr_matrix((N_USERS, N_USERS), dtype=np.float32)
    zeros_ii = sparse.csr_matrix((N_ITEMS, N_ITEMS), dtype=np.float32)
    A        = sparse.bmat([[zeros_uu, R], [R.T, zeros_ii]], format='csr')

    deg    = np.array(A.sum(axis=1)).flatten()
    d_inv  = np.where(deg > 0, deg ** -0.5, 0.0)
    D_inv  = sparse.diags(d_inv, format='csr')
    A_norm = D_inv @ A @ D_inv

    A_coo   = A_norm.tocoo()
    indices = torch.LongTensor(np.vstack([A_coo.row, A_coo.col]))
    values  = torch.FloatTensor(A_coo.data)
    A_torch = torch.sparse_coo_tensor(indices, values, (N, N)).to(DEVICE)

    return (N_USERS, N_ITEMS, A_torch, user2idx, item2idx,
            idx2user, idx2item, user_pos_items, movies_df)


# ── Initialisation globale ────────────────────────────────────────────────────
print("Chargement des données...")
(N_USERS, N_ITEMS, A_TORCH, user2idx, item2idx,
 idx2user, idx2item, user_pos_items, movies_df) = load_data()

print("Chargement du modèle...")
model = LightGCN(
    n_users   = N_USERS,
    n_items   = N_ITEMS,
    embed_dim = EMBED_DIM,
    n_layers  = N_LAYERS,
    A_norm    = A_TORCH,
    dropout   = 0.1
).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print(f"Modèle chargé ({model.count_parameters():,} paramètres)")


# ── Fonctions utilitaires ─────────────────────────────────────────────────────
def get_recommendations(user_idx, k=10):
    with torch.no_grad():
        u_tensor = torch.LongTensor([user_idx]).to(DEVICE)
        scores   = model.recommend(u_tensor).squeeze(0).cpu().numpy()

    for pos in user_pos_items.get(user_idx, []):
        if pos < N_ITEMS:
            scores[pos] = -1e9

    top_k_indices      = np.argsort(-scores)[:k]
    movie_id_to_title  = dict(zip(movies_df.index, movies_df['title']))
    movie_id_to_genres = dict(zip(movies_df.index, movies_df['genres']))

    recs = []
    for rank, item_idx in enumerate(top_k_indices, 1):
        movie_id = idx2item.get(int(item_idx), -1)
        title    = str(movie_id_to_title.get(movie_id, f"Film #{movie_id}"))
        genres   = [str(g) for g in movie_id_to_genres.get(movie_id, [])]
        score    = float(scores[item_idx])
        recs.append({
            'rang':   int(rank),
            'titre':  title,
            'genres': genres,
            'score':  round(score, 4)
        })
    return recs


def get_history(user_idx, n=5):
    movie_id_to_title = dict(zip(movies_df.index, movies_df['title']))
    seen_items        = list(user_pos_items.get(user_idx, set()))[:n]
    return [str(movie_id_to_title.get(idx2item.get(i, -1), "?")) for i in seen_items]


# ── Routes Flask ──────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html',
                           total_users=N_USERS,
                           total_items=N_ITEMS)


@app.route('/recommend', methods=['POST'])
def recommend():
    data    = request.get_json()
    user_id = int(data.get('user_id', 0))
    k       = int(data.get('k', 10))

    if user_id in user2idx:
        user_idx = user2idx[user_id]
    elif user_id < N_USERS:
        user_idx = user_id
    else:
        return jsonify({'error': f'Utilisateur {user_id} introuvable'}), 404

    recs    = get_recommendations(user_idx, k=k)
    history = get_history(user_idx, n=5)

    return jsonify({
        'user_idx':        int(user_idx),
        'user_real_id':    int(idx2user.get(user_idx, user_idx)),
        'history':         history,
        'recommendations': recs
    })


@app.route('/users')
def list_users():
    users = sorted(user2idx.keys())[:50]
    return jsonify({'users': [int(u) for u in users], 'total': int(N_USERS)})


if __name__ == '__main__':
    app.run(debug=True, port=5000)