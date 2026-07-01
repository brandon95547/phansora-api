"""Platform-level configuration for the Phanoris API.

This holds only settings shared across products (app metadata, HTTP/CORS, and
which products to mount). Each product keeps its own product-specific config
module under ``phanoris.products.<name>`` — this file never imports those, to
keep the dependency direction platform -> product (never the reverse).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from dotenv import load_dotenv

load_dotenv()

# All products the platform knows about. A product is mounted only if it imports
# cleanly (its optional heavy dependencies are installed) — see phanoris.main.
ALL_PRODUCTS = ("spokenverse", "chrono_origin", "dossier_nova")


def _split_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class Settings:
    app_name: str = "Phanoris API"
    version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_allow_origins: List[str] = field(default_factory=lambda: ["*"])
    # Subset of ALL_PRODUCTS to expose. Empty => expose every product that imports.
    enabled_products: List[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            app_name=os.getenv("PHANORIS_APP_NAME", "Phanoris API"),
            version=os.getenv("PHANORIS_VERSION", "0.1.0"),
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            cors_allow_origins=_split_csv(os.getenv("CORS_ALLOW_ORIGINS", "*")),
            enabled_products=_split_csv(os.getenv("PHANORIS_ENABLED_PRODUCTS", "")),
        )


settings = Settings.from_env()
