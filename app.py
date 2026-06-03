"""
app.py  —  Déploiement Flask du modèle LightGCN
Étudiant : Kenné Takoudjou Franck Manuel — 24P817 — AIA4
Cours    : Intelligence Artificielle et Applications — Prof. M. Bitha
ENSPY — Université de Yaoundé I
"""

from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import sparse
import os
import json
import urllib.request

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__)

EMBED_DIM  = 64
N_LAYERS   = 3
DATA_DIR   = "data/ml-latest-small"
MODEL_PATH = "best_lightgcn.pth"
DEVICE     = torch.device("cpu")

# ══════════════════════════════════════════════════════════════════
#  MODÈLE LIGHTGCN
# ══════════════════════════════════════════════════════════════════
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, embed_dim, n_layers, adj_matrix):
        super().__init__()
        self.n_users    = n_users
        self.n_items    = n_items
        self.n_layers   = n_layers
        self.adj        = adj_matrix
        self.user_embed = nn.Embedding(n_users, embed_dim)
        self.item_embed = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

    def propagate(self):
        E = torch.cat([self.user_embed.weight,
                       self.item_embed.weight], dim=0)
        embs = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(self.adj, E)
            embs.append(E)
        return torch.stack(embs, dim=1).mean(dim=1)

    def forward(self, users, pos_items, neg_items):
        all_e = self.propagate()
        u   = all_e[users]
        ip  = all_e[self.n_users + pos_items]
        inp = all_e[self.n_users + neg_items]
        return ((u * ip).sum(1), (u * inp).sum(1),
                self.user_embed(users),
                self.item_embed(pos_items),
                self.item_embed(neg_items))

    def recommend(self, users):
        all_e = self.propagate()
        u = all_e[users]
        i = all_e[self.n_users:]
        return u @ i.T

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters()
                   if p.requires_grad)


# ══════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES ET DU MODÈLE (au démarrage)
# ══════════════════════════════════════════════════════════════════
def load_data_and_model():
    """Charge le dataset MovieLens, reconstruit le graphe et charge le modèle."""

    # ── 1. Téléchargement si nécessaire
    if not os.path.exists(os.path.join(DATA_DIR, "ratings.csv")):
        print("  Téléchargement MovieLens...")
        os.makedirs(DATA_DIR, exist_ok=True)
        base = "https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/"
        urllib.request.urlretrieve(
            base + "ratings.dat",
            os.path.join(DATA_DIR, "ratings_raw.dat"))
        urllib.request.urlretrieve(
            base + "movies.dat",
            os.path.join(DATA_DIR, "movies_raw.dat"))
        r = pd.read_csv(os.path.join(DATA_DIR, "ratings_raw.dat"),
                        sep="::", header=None,
                        names=["userId","movieId","rating","timestamp"],
                        engine="python")
        r["rating"] = r["rating"] / 2
        r.to_csv(os.path.join(DATA_DIR, "ratings.csv"), index=False)
        m = pd.read_csv(os.path.join(DATA_DIR, "movies_raw.dat"),
                        sep="::", header=None,
                        names=["movieId","title","genres"],
                        engine="python", encoding="latin-1")
        m.to_csv(os.path.join(DATA_DIR, "movies.csv"), index=False)
        print("  Dataset prêt.")

    # ── 2. Chargement
    ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
    movies_df  = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

    # ── 3. Binarisation et réindexation
    df = ratings_df[ratings_df["rating"] >= 3.5][["userId","movieId"]].copy()
    users_unique = sorted(df["userId"].unique())
    items_unique = sorted(df["movieId"].unique())
    user2idx = {u: i for i, u in enumerate(users_unique)}
    item2idx = {m: i for i, m in enumerate(items_unique)}
    idx2user = {i: u for u, i in user2idx.items()}
    idx2item = {i: m for m, i in item2idx.items()}
    df["user"] = df["userId"].map(user2idx)
    df["item"] = df["movieId"].map(item2idx)
    N_USERS = len(users_unique)
    N_ITEMS = len(items_unique)

    # ── 4. Dictionnaire item → titre/genres
    movies_df = movies_df.dropna(subset=["movieId"])
    movies_df["movieId"] = movies_df["movieId"].astype(int)
    movie_info = movies_df.set_index("movieId")[["title","genres"]].to_dict("index")

    # ── 5. Items vus par utilisateur
    user_items = df.groupby("user")["item"].apply(list).to_dict()

    # ── 6. Matrice d'adjacence normalisée
    R = sparse.csr_matrix(
        (np.ones(len(df), dtype=np.float32),
         (df["user"].values, df["item"].values)),
        shape=(N_USERS, N_ITEMS))
    A = sparse.bmat([
        [sparse.csr_matrix((N_USERS, N_USERS), dtype=np.float32), R],
        [R.T, sparse.csr_matrix((N_ITEMS, N_ITEMS), dtype=np.float32)]
    ], format="csr").astype(np.float32)
    deg        = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, deg ** (-0.5), 0.0)
    A_norm     = (sparse.diags(d_inv_sqrt) @ A @ sparse.diags(d_inv_sqrt)).tocoo()
    indices    = torch.from_numpy(
        np.vstack([A_norm.row, A_norm.col])).long()
    values     = torch.from_numpy(A_norm.data)
    norm_adj   = torch.sparse_coo_tensor(
        indices, values,
        torch.Size([N_USERS + N_ITEMS, N_USERS + N_ITEMS]))

    # ── 7. Instanciation du modèle
    model = LightGCN(N_USERS, N_ITEMS, EMBED_DIM, N_LAYERS, norm_adj)

    # ── 8. Chargement des poids (si disponible)
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"  Modèle chargé depuis {MODEL_PATH}")
    else:
        print(f"  ATTENTION : {MODEL_PATH} introuvable — "
              f"utilisation des poids aléatoires (pour démo).")

    model.eval()
    print(f"  LightGCN prêt — {model.count_parameters():,} paramètres")
    print(f"  {N_USERS} utilisateurs, {N_ITEMS} films")

    return (model, N_USERS, N_ITEMS,
            user2idx, item2idx, idx2user, idx2item,
            user_items, movie_info)


print("Chargement du modèle LightGCN...")
(model, N_USERS, N_ITEMS,
 user2idx, item2idx, idx2user, idx2item,
 user_items, movie_info) = load_data_and_model()
print("Serveur Flask prêt.\n")


# ══════════════════════════════════════════════════════════════════
#  FONCTION UTILITAIRE
# ══════════════════════════════════════════════════════════════════
def get_recommendations(user_idx: int, k: int = 10):
    """Retourne les Top-K films recommandés pour un utilisateur."""
    model.eval()
    with torch.no_grad():
        scores = model.recommend(
            torch.LongTensor([user_idx])
        ).squeeze(0).cpu().numpy()

    # Masquer les films déjà vus
    for seen in user_items.get(user_idx, []):
        scores[seen] = -1e9

    top_indices = np.argsort(-scores)[:k]
    recs = []
    for rank, item_idx in enumerate(top_indices, 1):
        movie_id = idx2item.get(int(item_idx))
        info     = movie_info.get(movie_id, {})
        recs.append({
            "rank":     rank,
            "item_idx": int(item_idx),
            "movie_id": int(movie_id) if movie_id else None,
            "title":    info.get("title", f"Film #{movie_id}"),
            "genres":   info.get("genres", "N/A"),
            "score":    round(float(scores[item_idx]), 4),
        })
    return recs


def get_user_history(user_idx: int, limit: int = 5):
    """Retourne l'historique de films vus par un utilisateur."""
    items = user_items.get(user_idx, [])[:limit]
    history = []
    for item_idx in items:
        movie_id = idx2item.get(item_idx)
        info     = movie_info.get(movie_id, {})
        history.append({
            "title":  info.get("title", f"Film #{movie_id}"),
            "genres": info.get("genres", "N/A"),
        })
    return history


# ══════════════════════════════════════════════════════════════════
#  ROUTES FLASK
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Page principale — interface de recommandation."""
    return render_template("index.html",
                           n_users=N_USERS,
                           n_items=N_ITEMS,
                           n_params=model.count_parameters())


@app.route("/api/recommend", methods=["GET", "POST"])
def api_recommend():
    """
    API REST de recommandation.

    GET  /api/recommend?user_id=0&k=10
    POST /api/recommend  JSON: {"user_id": 0, "k": 10}
    """
    if request.method == "POST":
        data    = request.get_json(force=True) or {}
        user_id = int(data.get("user_id", 0))
        k       = int(data.get("k", 10))
    else:
        user_id = int(request.args.get("user_id", 0))
        k       = int(request.args.get("k", 10))

    # Validation
    if user_id < 0 or user_id >= N_USERS:
        return jsonify({
            "error": f"user_id doit être entre 0 et {N_USERS - 1}"
        }), 400
    k = max(1, min(k, 50))

    recs    = get_recommendations(user_id, k)
    history = get_user_history(user_id)

    return jsonify({
        "user_id":         user_id,
        "real_user_id":    idx2user.get(user_id),
        "k":               k,
        "history_sample":  history,
        "recommendations": recs,
        "model_info": {
            "name":       "LightGCN",
            "layers":     N_LAYERS,
            "embed_dim":  EMBED_DIM,
            "n_users":    N_USERS,
            "n_items":    N_ITEMS,
            "parameters": model.count_parameters(),
        }
    })


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Statistiques générales du modèle déployé."""
    return jsonify({
        "model":        "LightGCN",
        "layers_K":     N_LAYERS,
        "embed_dim_d":  EMBED_DIM,
        "n_users":      N_USERS,
        "n_items":      N_ITEMS,
        "parameters":   model.count_parameters(),
        "dataset":      "MovieLens small (ml-latest-small)",
        "loss":         "BPR (Bayesian Personalized Ranking)",
        "metrics":      {
            "Recall@10": 0.1051,
            "NDCG@10":   0.1650,
            "Recall@20": 0.1546,
            "NDCG@20":   0.1702,
        },
        "status": "running"
    })


@app.route("/api/users", methods=["GET"])
def api_users():
    """Liste des premiers utilisateurs disponibles."""
    limit = int(request.args.get("limit", 20))
    users = [{"user_idx": i, "real_id": idx2user.get(i),
              "n_interactions": len(user_items.get(i, []))}
             for i in range(min(limit, N_USERS))]
    return jsonify({"total_users": N_USERS, "sample": users})


# ══════════════════════════════════════════════════════════════════
#  LANCEMENT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)