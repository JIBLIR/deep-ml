import numpy as np

def logit_lens(hidden_states, ln_gamma, ln_beta, W_U, top_k=1):
    """
    Apply the logit lens to per-layer hidden states.

    Returns:
        A list of length L; each entry is a list of top_k token indices.
    """
    mean = np.mean(hidden_states,axis=-1).reshape(-1,1)
    var = np.var(hidden_states)
    normed = (hidden_states - mean) / np.sqrt(var + 1e-5)
    scaled = normed * ln_gamma + ln_beta
    logits = scaled @ W_U
    return np.argsort(-logits, axis=-1, kind='stable')[:, :top_k]
