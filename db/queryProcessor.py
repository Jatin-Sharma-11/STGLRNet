import torch.nn as nn
import numpy as np
import faiss
import os
import torch

class Retriever:
    def __init__(self, db_path, window_size=12):
        """
        Loads index and time series data
        """
        self.db_path = db_path
        self.window_size = window_size
        self.index_path = os.path.join(db_path, "faiss.index")
        self.data_path = os.path.join(db_path, "time_series.npy")

        self._load()

    def _load(self):
        if not os.path.exists(self.index_path) or not os.path.exists(self.data_path):
            raise FileNotFoundError("Missing FAISS index or time series file.")

        self.index = faiss.read_index(self.index_path)
        self.time_series = np.load(self.data_path)

        self.feature_dim = self.time_series.shape[1]
        self.channel_dim = self.time_series.shape[2]
        self.num_windows = self.time_series.shape[0] - self.window_size + 1

    def retrieve(self, query_seq, X):
        assert query_seq.shape == (self.window_size, self.feature_dim, self.channel_dim)
        flat_query = query_seq.reshape(1, -1).astype(np.float32)
        _, indices = self.index.search(flat_query, 1)
        start_index = indices[0][0]

        available = max(0, start_index)
        padding = X - available

        past_seq = self.time_series[max(0, start_index - X): start_index]
        if padding > 0:
            zero_pad = np.zeros((padding, self.feature_dim, self.channel_dim), dtype=self.time_series.dtype)
            past_seq = np.concatenate([zero_pad, past_seq], axis=0)

        match_seq = self.time_series[start_index : start_index + self.window_size]
        matched_seq = np.concatenate([past_seq, match_seq], axis=0)
        return matched_seq, start_index

    def retrieve_batch(self, query_batch, X):
        B, W, F, C = query_batch.shape
        assert W == self.window_size and F == self.feature_dim and C == self.channel_dim

        flat_queries = query_batch.reshape(B, -1).astype(np.float32)
        _, indices = self.index.search(flat_queries, 1)
        start_indices = indices.flatten()

        matched_seqs = []
        for i in range(B):
            idx = start_indices[i]
            available = max(0, idx)
            padding = X - available

            past_seq = self.time_series[max(0, idx - X): idx]
            if padding > 0:
                zero_pad = np.zeros((padding, self.feature_dim, self.channel_dim), dtype=self.time_series.dtype)
                past_seq = np.concatenate([zero_pad, past_seq], axis=0)

            match_seq = self.time_series[idx : idx + self.window_size]
            full_seq = np.concatenate([past_seq, match_seq], axis=0)
            matched_seqs.append(full_seq)

        return np.stack(matched_seqs, axis=0), start_indices
