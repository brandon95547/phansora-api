"""Request and response shapes. The wire format is camelCase, like the rest of the dashboard's JSON."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class Model(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class IllustrationRequest(Model):
    prompt: str = Field(min_length=1, max_length=4000)
    style_id: str
    # Width over height. The picture keeps the base size's pixel budget at this shape.
    aspect: float = Field(gt=0.2, lt=5)
    purpose: Literal["interior", "cover"] = "interior"
    # The book's pages: pictures for dark pages are drawn on a dark ground.
    theme: Literal["light", "dark"] = "light"
    seed: int | None = None


class IllustrationResponse(Model):
    # A path under the API's root (e.g. /talari/users/7/images/<hex>.png). A browser that
    # cannot send a header with an <img> adds ?pt=<token>, which the auth gate accepts.
    src: str
    seed: int


class ServiceState(Model):
    available: bool
    detail: str = ""


class Status(Model):
    illustration: ServiceState
