# phansora-api: how the back end is built

FastAPI, on Python 3.10 in prod and 3.11 in dev. One app serves every product under its own path prefix. `README.md` has the products, their prefixes and the AI models each one uses. Keep new work inside the structure below.

## Where code goes

Imports only point down. Two products meet only in `shared/`.

| Layer | Path | May import |
| --- | --- | --- |
| shared | `src/phansora/shared/` (ai, auth, net, audio, ocr, conversions, utils, errors, paths) | shared |
| product | `src/phansora/products/<name>/` | shared, its own product |
| app | `src/phansora/main.py`, `cli.py`, `sweep.py` | everything |

- **A product's layout.** Routes go in `server.py`, request and response models in `models.py` or `schemas.py`, and settings in `config.py`. The work goes in `services/`, or in modules beside `server.py`.
- **Thin handlers.** A route checks the input, calls a service and shapes the reply. Work that outlasts a request runs in a worker, as Book Alchemy's does.
- **Code a second product needs** moves to `shared/`. Never import from another product, and never copy it.
  - Talari and Narrava Studio still import `book_alchemy.parsers`. That parser belongs in `shared/`, and nothing new imports across products.
- **A module that grows past a few hundred lines,** or starts doing two jobs, is split along its parts before more is added.

## AI and outside calls

- **Models.** Every model comes from `.env` through `shared/ai/models.py`, which resolves `<PRODUCT>_MODEL`, then `<PROVIDER>_MODEL`. Never write a model name in code.
- **Provider calls.** They go through the shared clients in `shared/ai/` (`deepseek.py`, with the shared `retries.py`), each with a timeout. A product wraps them in its own `llm.py`, as Narrava Studio and Talari do. No ad-hoc HTTP to a provider.
- **Errors.** A failed call becomes a response through `shared/errors.fail()`: the caller gets a fixed sentence and a reference, and the log gets the traceback. An exception's text never goes into a response.

## Files

- **Where they go.** Everything a person can see or download is written under the data folder: `PHANSORA_DATA_DIR`, through `shared/paths.py`. That is `<repo>/data` on prod, and it is backed up hourly. Working files nobody sees stay out of it.
- **Deleting.** A feature that writes files deletes them in the same change: when the item goes, when the project goes, and when the account goes. One folder per project keeps that to a single delete.

## Dependencies

- Pinned in `requirements.txt` (prod) and `requirements-ci.txt` (tests), never in `pyproject.toml`.
- A package pinned in both lists has the same version in each (`tests/test_requirements.py`).

## Tests and checks

- **Test files.** One file per product and topic: `tests/test_<product>_<topic>.py`.
- **Stubbing.** Stub the models and the network at the boundary (monkeypatch the product's `llm` calls), so the suite runs without keys.
- **Swallowed errors.** Ruff forbids a silent `except Exception: pass` (S110). Log what was swallowed, or say in the `noqa` why it is safe to drop.
- **Before committing.** Run what CI runs:

```bash
.venv/bin/ruff check && .venv/bin/python -m pytest -q
```
