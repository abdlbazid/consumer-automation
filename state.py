# state.py
state = {
    "app": None,          # current app, e.g., gmail, amazon
    "page": None,         # current page, e.g., inbox, search_results
    "focused": None,      # last focused element / "this" pointer
    "items": []           # visible items (emails, products, etc.)
}

def dump_context():
    """Clear the short-term context memory"""
    state["page"] = None
    state["focused"] = None
    state["items"] = []

def update_app(new_app):
    if state["app"] != new_app:
        dump_context()  # reset context when switching apps
    state["app"] = new_app

def update_page(new_page):
    if state["page"] != new_page:
        dump_context()  # reset context when page changes
    state["page"] = new_page

def new_conversation():
    dump_context()      # reset context for a new conversation
