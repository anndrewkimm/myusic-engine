"""Checks that raw account fields do not cross the normalization boundary."""

from collections.abc import Mapping, Sequence

SENSITIVE_RAW_FIELDS = frozenset(
    {
        "address",
        "birthdate",
        "email",
        "ip_addr",
        "postal_code",
        "user_agent",
        "username",
    }
)


class PrivacyBoundaryError(ValueError):
    """Raised when a sensitive raw field appears in a cleaned record."""


def find_sensitive_fields(value: object) -> set[str]:
    """Return sensitive key names found anywhere in a JSON-like value."""

    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            normalized_key = str(key).casefold()
            if normalized_key in SENSITIVE_RAW_FIELDS:
                found.add(normalized_key)
            found.update(find_sensitive_fields(nested_value))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            found.update(find_sensitive_fields(item))
    return found


def assert_privacy_safe(value: object) -> None:
    """Raise without echoing values when a sensitive field name is present."""

    sensitive_fields = find_sensitive_fields(value)
    if sensitive_fields:
        field_list = ", ".join(sorted(sensitive_fields))
        raise PrivacyBoundaryError(
            f"Sensitive field names crossed the privacy boundary: {field_list}"
        )
