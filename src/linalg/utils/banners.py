"""Small reusable banner printer for exercises and scripts."""

def banner(title: str) -> None:
    """Print a one-line title with an underline.

    Example
    -------
    >>> banner("Exercise 1: Variables")
    ~ Exercise 1: Variables
    ───────────────────────
    """
    print(f"\n~ {title}")
    print("─" * (len(title) + 2))
