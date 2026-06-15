"""
app.py  —  Déploiement Flask du modèle LightGCN
Étudiant : Kenné Takoudjou Franck Manuel — 24P817 — AIA4
Cours    : Intelligence Artificielle et Applications — Prof. M. Bitha
ENSPY — Université de Yaoundé I
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from scipy import sparse
from flask import Flask, request, jsonify, render_template

# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBED_DIM = 64
N_LAYERS = 3
DATA_DIR = "data/ml-latest-small"
MODEL_PATH = "best_lightgcn.pth"
DEVICE = torch.device("cpu")


# ══════════════════════════════════════════════════════════════════
#  HELPERS DE CONVERSION
# ══════════════════════════════════════════════════════════════════
def convert_to_python_types(obj: Any) -> Any:
    """Convertit les types NumPy/PyTorch en types Python sérialisables."""
    if isinstance(obj, dict):
        return {k: convert_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().detach().item() if obj.numel() == 1 else obj.cpu().detach().tolist()
    return obj


class NumpyEncoder(json.JSONEncoder):
    """Encodeur JSON personnalisé pour NumPy/PyTorch types."""
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


app.json_encoder = NumpyEncoder


# ══════════════════════════════════════════════════════════════════
#  MODÈLE LIGHTGCN
# ══════════════════════════════════════════════════════════════════
class LightGCN(nn.Module):
    def __init__(self, n_users: int, n_items: int, embed_dim: int,
                 n_layers: int, adj_matrix):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.n_layers = n_layers
        self.adj = adj_matrix
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
        u = all_e[users]
        ip = all_e[self.n_users + pos_items]
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

    def count_parameters(self) -> int:
        return int(sum(p.numel() for p in self.parameters()
                       if p.requires_grad))


# ══════════════════════════════════════════════════════════════════
#  CHARGEMENT DES DONNÉES ET DU MODÈLE
# ══════════════════════════════════════════════════════════════════
def load_data_and_model() -> Tuple:
    """Charge le dataset MovieLens et initialise le modèle."""
    logger.info("Démarrage du chargement des données...")

    # 1. Vérifier et charger les fichiers CSV
    if not os.path.exists(os.path.join(DATA_DIR, "ratings.csv")):
        logger.error(f"Fichier ratings.csv introuvable dans {DATA_DIR}")
        raise FileNotFoundError(f"ratings.csv not found in {DATA_DIR}")

    if not os.path.exists(os.path.join(DATA_DIR, "movies.csv")):
        logger.error(f"Fichier movies.csv introuvable dans {DATA_DIR}")
        raise FileNotFoundError(f"movies.csv not found in {DATA_DIR}")

    try:
        ratings_df = pd.read_csv(os.path.join(DATA_DIR, "ratings.csv"))
        movies_df = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))
        logger.info(f"Chargé: {len(ratings_df)} ratings, {len(movies_df)} films")
    except Exception as e:
        logger.error(f"Erreur lors du chargement CSV: {e}")
        raise

    # 2. Binarisation et réindexation
    df = ratings_df[ratings_df["rating"] >= 3.5][["userId", "movieId"]].copy()

    if len(df) == 0:
        logger.error("Aucune interaction trouvée après filtrage")
        raise ValueError("No interactions found after filtering")

    users_unique = sorted(df["userId"].unique())
    items_unique = sorted(df["movieId"].unique())

    user2idx = {int(u): int(i) for i, u in enumerate(users_unique)}
    item2idx = {int(m): int(i) for i, m in enumerate(items_unique)}
    idx2user = {int(i): int(u) for u, i in user2idx.items()}
    idx2item = {int(i): int(m) for m, i in item2idx.items()}

    df["user"] = df["userId"].map(user2idx).astype(int)
    df["item"] = df["movieId"].map(item2idx).astype(int)

    N_USERS = len(users_unique)
    N_ITEMS = len(items_unique)

    logger.info(f"Après filtrage: {N_USERS} utilisateurs, {N_ITEMS} films")

    # 3. Dictionnaire item → titre/genres (nettoyé)
    movies_df = movies_df.dropna(subset=["movieId"])
    movies_df["movieId"] = movies_df["movieId"].astype(int)

    movie_info = {}
    for _, row in movies_df.iterrows():
        movie_id = int(row["movieId"])
        movie_info[movie_id] = {
            "title": str(row.get("title", "Unknown")).strip(),
            "genres": str(row.get("genres", "N/A")).strip()
        }

    logger.info(f"Métadonnées de {len(movie_info)} films chargées")

    # 4. Items vus par utilisateur
    user_items = {}
    for user_idx, group in df.groupby("user")["item"]:
        user_items[int(user_idx)] = [int(x) for x in group.tolist()]

    # 5. Matrice d'adjacence normalisée
    R = sparse.csr_matrix(
        (np.ones(len(df), dtype=np.float32),
         (df["user"].values, df["item"].values)),
        shape=(N_USERS, N_ITEMS))

    A = sparse.bmat([
        [sparse.csr_matrix((N_USERS, N_USERS), dtype=np.float32), R],
        [R.T, sparse.csr_matrix((N_ITEMS, N_ITEMS), dtype=np.float32)]
    ], format="csr").astype(np.float32)

    deg = np.array(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.where(deg > 0, deg ** (-0.5), 0.0)
    A_norm = (sparse.diags(d_inv_sqrt) @ A @ sparse.diags(d_inv_sqrt)).tocoo()

    indices = torch.from_numpy(
        np.vstack([A_norm.row, A_norm.col])).long()
    values = torch.from_numpy(A_norm.data)
    norm_adj = torch.sparse_coo_tensor(
        indices, values,
        torch.Size([N_USERS + N_ITEMS, N_USERS + N_ITEMS]))

    logger.info("Matrice d'adjacence construite")

    # 6. Instanciation du modèle
    model = LightGCN(N_USERS, N_ITEMS, EMBED_DIM, N_LAYERS, norm_adj)
    model.to(DEVICE)

    # 7. Chargement des poids
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            logger.info(f"Modèle chargé depuis {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle: {e}")
            logger.warning("Utilisation des poids aléatoires")
    else:
        logger.warning(f"Fichier {MODEL_PATH} introuvable — poids aléatoires")

    model.eval()
    n_params = model.count_parameters()
    logger.info(f"LightGCN prêt — {n_params:,} paramètres")

    return (model, N_USERS, N_ITEMS,
            user2idx, item2idx, idx2user, idx2item,
            user_items, movie_info)


# Chargement au démarrage
try:
    (model, N_USERS, N_ITEMS,
     user2idx, item2idx, idx2user, idx2item,
     user_items, movie_info) = load_data_and_model()
    logger.info("Serveur Flask prêt.")
except Exception as e:
    logger.critical(f"Impossible de démarrer l'application: {e}")
    raise


# ══════════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════
def get_recommendations(user_idx: int, k: int = 10) -> List[Dict]:
    """Retourne les Top-K films recommandés."""
    try:
        model.eval()
        with torch.no_grad():
            scores = model.recommend(
                torch.LongTensor([user_idx]).to(DEVICE)
            ).squeeze(0).cpu().numpy()

        # Masquer les films déjà vus
        for seen in user_items.get(user_idx, []):
            scores[seen] = -1e9

        top_indices = np.argsort(-scores)[:k]
        recs = []

        for rank, item_idx in enumerate(top_indices, 1):
            item_idx = int(item_idx)
            movie_id = idx2item.get(item_idx)
            info = movie_info.get(movie_id, {"title": f"Film #{movie_id}", "genres": "N/A"})

            recs.append({
                "rank": int(rank),
                "item_idx": item_idx,
                "movie_id": int(movie_id) if movie_id else None,
                "title": str(info.get("title", "Unknown")),
                "genres": str(info.get("genres", "N/A")),
                "score": float(scores[item_idx]),
            })

        return recs
    except Exception as e:
        logger.error(f"Erreur dans get_recommendations: {e}")
        return []


def get_user_history(user_idx: int, limit: int = 5) -> List[Dict]:
    """Retourne l'historique de films vus."""
    try:
        items = user_items.get(user_idx, [])[:limit]
        history = []

        for item_idx in items:
            movie_id = idx2item.get(int(item_idx))
            info = movie_info.get(movie_id, {"title": f"Film #{movie_id}", "genres": "N/A"})

            history.append({
                "title": str(info.get("title", "Unknown")),
                "genres": str(info.get("genres", "N/A")),
            })

        return history
    except Exception as e:
        logger.error(f"Erreur dans get_user_history: {e}")
        return []


# ══════════════════════════════════════════════════════════════════
#  ROUTES FLASK
# ══════════════════════════════════════════════════════════════════
@app.route("/", methods=["GET"])
def index():
    """Page principale."""
    try:
        return render_template("index.html",
                               n_users=int(N_USERS),
                               n_items=int(N_ITEMS),
                               n_params=model.count_parameters())
    except Exception as e:
        logger.error(f"Erreur index: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/recommend", methods=["GET", "POST"])
def api_recommend():
    """API REST de recommandation."""
    try:
        if request.method == "POST":
            data = request.get_json(force=True) or {}
            user_id = int(data.get("user_id", 0))
            k = int(data.get("k", 10))
        else:
            user_id = int(request.args.get("user_id", 0))
            k = int(request.args.get("k", 10))

        # Validation
        if user_id < 0 or user_id >= N_USERS:
            return jsonify({
                "error": f"user_id must be between 0 and {N_USERS - 1}",
                "received": int(user_id)
            }), 400

        k = max(1, min(k, 50))

        recs = get_recommendations(user_id, k)
        history = get_user_history(user_id)

        response = {
            "success": True,
            "user_id": int(user_id),
            "real_user_id": int(idx2user.get(user_id, -1)),
            "k": int(k),
            "history_sample": history,
            "recommendations": recs,
            "model_info": {
                "name": "LightGCN",
                "layers": int(N_LAYERS),
                "embed_dim": int(EMBED_DIM),
                "n_users": int(N_USERS),
                "n_items": int(N_ITEMS),
                "parameters": model.count_parameters(),
            }
        }

        return jsonify(convert_to_python_types(response))
    except ValueError as e:
        logger.error(f"Erreur de validation: {e}")
        return jsonify({"error": f"Invalid input: {str(e)}"}), 400
    except Exception as e:
        logger.error(f"Erreur api_recommend: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Statistiques du modèle."""
    try:
        response = {
            "success": True,
            "model": "LightGCN",
            "layers_K": int(N_LAYERS),
            "embed_dim_d": int(EMBED_DIM),
            "n_users": int(N_USERS),
            "n_items": int(N_ITEMS),
            "parameters": model.count_parameters(),
            "dataset": "MovieLens small (ml-latest-small)",
            "loss": "BPR (Bayesian Personalized Ranking)",
            "metrics": {
                "Recall@10": 0.1051,
                "NDCG@10": 0.1650,
                "Recall@20": 0.1546,
                "NDCG@20": 0.1702,
            },
            "status": "running"
        }

        return jsonify(convert_to_python_types(response))
    except Exception as e:
        logger.error(f"Erreur api_stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/users", methods=["GET"])
def api_users():
    """Liste des utilisateurs disponibles."""
    try:
        limit = int(request.args.get("limit", 20))
        limit = max(1, min(limit, N_USERS))

        users = []
        for i in range(limit):
            users.append({
                "user_idx": int(i),
                "real_id": int(idx2user.get(i, -1)),
                "n_interactions": int(len(user_items.get(i, [])))
            })

        response = {
            "success": True,
            "total_users": int(N_USERS),
            "sample": users
        }

        return jsonify(convert_to_python_types(response))
    except Exception as e:
        logger.error(f"Erreur api_users: {e}")
        return jsonify({"error": str(e)}), 500


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "success": False}), 404


@app.errorhandler(500)
def server_error(e):
    logger.error(f"Erreur serveur: {e}")
    return jsonify({"error": "Internal server error", "success": False}), 500


# ══════════════════════════════════════════════════════════════════
#  LANCEMENT
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
