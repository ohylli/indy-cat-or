"""indycat — core recognition pipeline for "is this the cat Indy?".

This package holds the importable, UI-agnostic core: the
detect -> crop -> embed -> decide pipeline. It knows nothing about how it is
presented. Any UI (CLI, Streamlit, web) is a thin layer in ``scripts/`` that
calls into here.
"""
