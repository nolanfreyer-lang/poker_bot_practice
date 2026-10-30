"""Fast ICM (Malmuth-Harville), vectorised with numpy: ~0.03 ms for 8 players."""
try:
    import numpy as np
except ImportError:          # no numpy -> the bot falls back to chip EV
    np = None

_ICM_CACHE = {}


def _icm_layers(n):
    if n not in _ICM_CACHE:
        masks = np.arange(1 << n)
        M = ((masks[:, None] >> np.arange(n)) & 1).astype(float)
        pop = M.sum(1).astype(int)
        layers = []
        for k in range(n):
            frm, ply = np.nonzero((pop[:, None] == k) & (M == 0))
            layers.append((frm, ply, frm | (1 << ply)))
        _ICM_CACHE[n] = (M, layers)
    return _ICM_CACHE[n]


if np is not None:   # build every table size now, so no decision pays the setup cost
    for _n in range(2, 10):
        _icm_layers(_n)


def icm_points(stacks, payouts):
    """stacks: {seat: chips} of players still in. -> {seat: expected points}.
    Players with 0 chips (busting in this scenario) take the lowest places."""
    alive = [s for s, c in stacks.items() if c > 0]
    out = {s: 0.0 for s in stacks}
    busted = [s for s in stacks if stacks[s] <= 0]
    n = len(alive)
    pays = list(payouts[:len(stacks)]) + [0] * max(0, len(stacks) - len(payouts))
    if busted:   # they share the places below the survivors
        share = sum(pays[n:len(stacks)]) / len(busted)
        for s in busted:
            out[s] = share
    if n == 1:
        out[alive[0]] = pays[0]
        return out
    M, layers = _icm_layers(n)
    st = np.asarray([stacks[s] for s in alive], float)
    tot = st.sum()
    S = M @ st
    prob = np.zeros(1 << n)
    prob[0] = 1.0
    eq = np.zeros(n)
    for k, (frm, ply, to) in enumerate(layers):
        c = prob[frm] * st[ply] / (tot - S[frm])
        if pays[k]:
            eq += np.bincount(ply, c * pays[k], n)
        prob += np.bincount(to, c, 1 << n)
    for i, s in enumerate(alive):
        out[s] = float(eq[i])
    return out
