import os
import sys

ENV_PORT = "PORT"
ENV_RELOAD = "RELOAD"
ENV_APP_MODE = "APP_MODE"
MODE_SERVE = "serve"
DEFAULT_PORT = 8080
DEFAULT_RELOAD = "true"

def _maybe_prepare_dataset() -> None:
    from src.ingest.prepare_llm_dataset import main as prep_main
    try:
        prep_main()
    except Exception as e:
        print(f"[WARN] Dataset prep skipped: {e}")

def _build_index() -> None:
    from src.ingest.build_index import build_index
    info = build_index()
    print(f"[INDEX] {info}")

def _serve() -> None:
    import uvicorn
    port = int(os.getenv(ENV_PORT, str(DEFAULT_PORT)))
    reload_enabled = os.getenv(ENV_RELOAD, DEFAULT_RELOAD).lower() == "true"
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=port, reload=reload_enabled)

def main(argv: list[str] | None = None) -> int:
    argv = list(argv or sys.argv[1:])
    mode = (argv[0].lower() if argv else os.getenv(ENV_APP_MODE, "").lower())
    if mode == MODE_SERVE:
        _serve()
        return 0
    _maybe_prepare_dataset()
    _build_index()
    _serve()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())