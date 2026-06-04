# src/rbac.py

# Role hierarchy — each role can access its own documents
# plus all documents from roles below it in the hierarchy
ROLE_ACCESS = {
    "intern":    ["intern"],
    "engineer":  ["intern", "engineer"],
    "manager":   ["intern", "engineer", "manager"],
    "executive": ["intern", "engineer", "manager", "executive"]
}

# Hardcoded users for Version 1
# In production this would be a database with hashed passwords
USERS = {
    "alice": {
        "password": "intern123",
        "role": "intern",
        "display_name": "Alice Chen"
    },
    "bob": {
        "password": "engineer123",
        "role": "engineer",
        "display_name": "Bob Martinez"
    },
    "carol": {
        "password": "manager123",
        "role": "manager",
        "display_name": "Carol Singh"
    },
    "dave": {
        "password": "executive123",
        "role": "executive",
        "display_name": "Dave Thompson"
    }
}


def get_allowed_namespaces(role: str) -> list:
    """
    Returns the list of Pinecone namespaces this role can search.

    Args:
        role (str): The user's role

    Returns:
        list: Namespaces this role is allowed to query

    Example:
        get_allowed_namespaces("engineer") -> ["intern", "engineer"]
    """
    if role not in ROLE_ACCESS:
        raise ValueError(f"Unknown role: {role}. "
                         f"Must be one of {list(ROLE_ACCESS.keys())}")
    return ROLE_ACCESS[role]


def authenticate_user(username: str, password: str) -> dict | None:
    """
    Validates username and password against the user store.

    Args:
        username (str): Submitted username
        password (str): Submitted password

    Returns:
        dict: User info if valid, None if invalid
    """
    user = USERS.get(username)
    if user and user["password"] == password:
        return {
            "username": username,
            "role": user["role"],
            "display_name": user["display_name"]
        }
    return None


def get_role_badge_color(role: str) -> str:
    """
    Returns a CSS color class for the role badge in the UI.
    """
    colors = {
        "intern": "badge-intern",
        "engineer": "badge-engineer",
        "manager": "badge-manager",
        "executive": "badge-executive"
    }
    return colors.get(role, "badge-intern")