import numpy as np

def tuned_lens(hidden_states, A, b, W_U):
    """
    Apply a per-layer learned affine probe followed by the shared
    unembedding to produce vocabulary logits at each layer.

    Args:
        hidden_states: np.ndarray of shape (L, d)
        A:             np.ndarray of shape (L, d, d)
        b:             np.ndarray of shape (L, d)
        W_U:           np.ndarray of shape (d, V)

    Returns:
        np.ndarray of shape (L, V) with per-layer logits.
    """
    logits_tuned = (np.einsum('ldk, lk -> ld', A, hidden_states) + b ) @ W_U
    return logits_tuned
    
