"""
CORS configuration — regression test.

The default CORS_ALLOWED_ORIGINS exact-string list must include the
actual deployed frontend origin (https://ai-decision-support-system.vercel.app),
not just localhost — otherwise the browser blocks GET /system/info and
/system/models with "Access-Control-Allow-Origin missing" even though
the backend returns 200. This project does not need Vercel preview-
deployment support, so `AppSettings.cors_allowed_origin_regex` is empty
(disabled) by default; if preview support is ever needed, that field
should be set to a regex matching this project/team's hashed preview
URLs (see the field's docstring in app/core/config.py).
"""

import re

from app.core.config import app_settings


def test_localhost_is_allowed_by_default():
    assert "http://localhost:3000" in app_settings.cors_allowed_origins.split(",")


def test_production_vercel_origin_is_allowed_by_default():
    assert (
        "https://ai-decision-support-system.vercel.app"
        in app_settings.cors_allowed_origins.split(",")
    )


def test_preview_regex_is_disabled_by_default():
    """No preview-deployment support needed for this project by default."""
    assert app_settings.cors_allowed_origin_regex == ""


def test_disabled_regex_does_not_accidentally_match_anything():
    # main.py passes `allow_origin_regex or None`, so an empty string
    # must be falsy/harmless here — this just documents that contract.
    assert not app_settings.cors_allowed_origin_regex
