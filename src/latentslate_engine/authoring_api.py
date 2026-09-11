"""Authoring endpoints, deliberately separate from execution tools and jobs."""

from copy import deepcopy
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .artifact_library import ArtifactLibrary
from .authoring import operation_descriptors, validate_document
from .authoring_store import RecipeStore, StoreError


def authoring_router(
    store: RecipeStore, builtins: dict, library: ArtifactLibrary
) -> APIRouter:
    router = APIRouter(prefix="/v1/authoring")

    async def body(request: Request) -> dict:
        try:
            value = await request.json()
        except ValueError:
            raise StoreError(400, "Request body must be valid JSON") from None
        if not isinstance(value, dict):
            raise StoreError(422, "Request body must be an object")
        return value

    def builtin(key):
        if key not in builtins:
            raise StoreError(404, "Built-in recipe not found")
        return deepcopy(builtins[key])

    @router.get("/roots")
    def roots():
        return {"roots": library.roots()}

    @router.post("/roots", status_code=201)
    async def add_root(request: Request):
        value = await body(request)
        if set(value) - {"path", "name"}:
            raise StoreError(
                422, "Root registration accepts only path and optional name"
            )
        return library.add_root(value.get("path"), value.get("name"))

    @router.delete("/roots/{root_id}")
    def remove_root(root_id: str):
        library.remove_root(root_id)
        return {"removed": True}

    @router.post("/artifacts/refresh")
    def refresh_artifacts():
        return library.refresh()

    @router.get("/artifacts/search")
    def search_artifacts(operation: str, field: str, q: str = "", limit: int = 50):
        return library.search(operation, field, q, limit)

    @router.get("/operations")
    def operations():
        return {"operations": operation_descriptors()}

    @router.get("/builtins")
    def list_builtins():
        return {
            "recipes": [
                {"key": key, "immutable": True, "document": doc}
                for key, doc in builtins.items()
            ]
        }

    @router.get("/builtins/{key}")
    def get_builtin(key: str):
        return {"key": key, "immutable": True, "document": builtin(key)}

    @router.post("/builtins/{key}/duplicate", status_code=201)
    async def duplicate(key: str, request: Request):
        value = await body(request)
        if set(value) - {"name"}:
            raise StoreError(422, "Duplicate accepts only an optional name")
        document = builtin(key)
        document["id"] = str(uuid4())
        document["name"] = value.get("name", document["name"] + " Copy")
        return store.save(document, base_revision=None)

    @router.post("/validate")
    async def validate(request: Request):
        return validate_document(await body(request))

    @router.get("/recipes")
    def list_recipes():
        return {"recipes": store.list()}

    @router.post("/recipes", status_code=201)
    async def create(request: Request):
        return store.save(await body(request), base_revision=None)

    @router.get("/recipes/{recipe_id}")
    def read(recipe_id: str):
        return store.read(recipe_id)

    @router.put("/recipes/{recipe_id}")
    async def update(recipe_id: str, request: Request):
        value = await body(request)
        if (
            set(value) != {"base_revision", "document"}
            or value["base_revision"] is None
        ):
            raise StoreError(422, "Update requires base_revision and document")
        if (
            not isinstance(value["document"], dict)
            or value["document"].get("id") != recipe_id
        ):
            raise StoreError(422, "Document UUID must match the URL")
        return store.save(value["document"], base_revision=value["base_revision"])

    @router.get("/recipes/{recipe_id}/revisions")
    def revisions(recipe_id: str):
        return {"revisions": store.revisions(recipe_id)}

    @router.get("/recipes/{recipe_id}/revisions/{revision}")
    def read_revision(recipe_id: str, revision: int):
        return store.read(recipe_id, revision)

    return router


async def authoring_error(_: Request, error: StoreError):
    content = {"error": {"message": str(error)}}
    if error.validation is not None:
        content["validation"] = error.validation
    return JSONResponse(status_code=error.status, content=content)
