"""Three constructed text tables; no external corpus or refund-agent records."""

import numpy as np

PATTERNS = {
    "refund": {
        "damaged": (
            "The item arrived broken.",
            "It is broken.",
            "The box arrived and the screen is broken.",
            "Arrived broken, never worked.",
            "It arrived with a broken handle.",
        ),
        "changed_mind": (
            "I changed my mind.",
            "It arrived, but I changed my mind.",
            "Never mind, I changed my mind.",
            "It arrived fine, I do not want it.",
            "I changed my mind before it arrived.",
        ),
        "not_delivered": (
            "My parcel never arrived.",
            "It never came.",
            "The order has not arrived.",
            "It never arrived and I am still waiting.",
            "Tracking says delivered, but it never arrived.",
        ),
    },
    "ticket": {
        "network": (
            "The network connection is broken.",
            "The network dropped every packet.",
            "The network is down, never connected.",
            "The connection to the network is broken.",
            "The network never responds.",
        ),
        "database": (
            "The database query is slow.",
            "The database has a broken index.",
            "The database never finishes the query.",
            "The query locks the database.",
            "The database query timed out.",
        ),
        "application": (
            "The application never starts.",
            "The application screen is broken.",
            "The application closed without saving.",
            "The application never opens the file.",
            "The application is broken after the update.",
        ),
    },
    "clinic": {
        "appointments": (
            "Please change my appointment.",
            "My appointment confirmation never arrived.",
            "I need a new appointment.",
            "The appointment booking page is broken.",
            "Please cancel my appointment.",
        ),
        "prescriptions": (
            "My prescription never arrived.",
            "Please renew my prescription.",
            "The prescription has the wrong dose.",
            "My prescription refill is missing.",
            "The prescription request page is broken.",
        ),
        "billing": (
            "My billing statement never arrived.",
            "The billing page is broken.",
            "Please explain this billing charge.",
            "My billing payment is missing.",
            "I need a corrected billing statement.",
        ),
    },
}
WORDS = {
    "refund": ("arrived", "broken", "mind", "never"),
    "ticket": ("application", "broken", "database", "network"),
    "clinic": ("appointment", "billing", "never", "prescription"),
}


def generate(domain="refund", n=300, seed=7):
    """Draw a class, then a pattern, independently; shares are in table order."""
    table = PATTERNS[domain]
    rng = np.random.default_rng(seed)
    labels = rng.choice(3, size=n, p=(0.55, 0.25, 0.20))
    choices = rng.integers(5, size=n)
    classes = tuple(table)
    texts = [table[classes[y]][choice] for y, choice in zip(labels, choices)]
    return texts, labels
