def fn_name_or_repr(fn):
    """Restituisce __name__ se esiste, altrimenti repr()."""
    return getattr(fn, "__name__", repr(fn))
