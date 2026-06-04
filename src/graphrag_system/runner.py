"""
GraphRAG runner for the installed Microsoft GraphRAG CLI.
"""
import subprocess
from pathlib import Path

from src.config import settings as cfg
from src.utils.runtime import resolve_graphrag_cli


DEFAULT_API_BASE = ""

LOOKUP_SEARCH_PROMPT = """---Role---

You answer questions using only the data tables provided.

---Rules---

- Use only facts that appear in the data tables.
- For direct lookup questions, copy the exact value from the most relevant record.
- If a transaction record already contains the requested field, prefer that explicit field over inferring from policy thresholds.
- If the question asks for an amount, return only the amount itself. If the value already uses a currency symbol such as $, omit a trailing currency code like USD unless the user asks for it.
- If the question asks for an approval level, return the exact approval text from the relevant transaction when available.
- Keep the answer concise and do not add explanations unless they are required to answer the question.
- Do not use markdown headings, bullets, citations, or commentary.
- If the tables do not contain the answer, say: I do not know.

---Target response length and format---

{response_type}

---Data tables---

{context_data}
"""


def _graphrag_cli() -> str:
    return resolve_graphrag_cli()


def _run_cli(
    args: list[str],
    cwd: Path,
    timeout: int = 120,
    verbose: bool = True,
) -> subprocess.CompletedProcess:
    """Run a GraphRAG CLI command and optionally print tail output."""
    result = subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd),
    )
    if verbose:
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        if stdout:
            print(f"  STDOUT: {stdout[-1200:]}")
        if stderr:
            print(f"  STDERR: {stderr[-1200:]}")
    return result


def _active_graphrag_provider() -> str:
    return "openai" if cfg.MODEL_BACKEND == "local_openai" else "gemini"


def _active_generation_model(model: str | None = None) -> str:
    if model:
        return model
    if cfg.MODEL_BACKEND == "local_openai":
        return cfg.LOCAL_GENERATION_MODEL
    return cfg.GENERATION_MODEL


def _active_graphrag_index_model(model: str | None = None) -> str:
    if model:
        return model
    if cfg.MODEL_BACKEND == "local_openai":
        return cfg.LOCAL_GRAPHRAG_INDEX_MODEL
    return cfg.GENERATION_MODEL


def _active_embedding_model(model: str | None = None) -> str:
    if model:
        return model
    if cfg.MODEL_BACKEND == "local_openai":
        return cfg.LOCAL_EMBEDDING_MODEL
    return cfg.EMBEDDING_MODEL


def _active_api_base(api_base: str | None = None) -> str:
    if cfg.MODEL_BACKEND == "local_openai":
        return cfg.LOCAL_LLM_BASE_URL
    return api_base or DEFAULT_API_BASE


def _settings_yaml(
    model_provider: str,
    model: str,
    indexing_model: str,
    embedding_model: str,
    api_base: str,
    api_key_env_var: str,
    input_dir: str = "input",
) -> str:
    """Build a GraphRAG settings.yaml compatible with the installed 3.x CLI."""
    completion_api_base = f"\n    api_base: {api_base}" if api_base else ""
    embedding_api_base = f"\n    api_base: {api_base}" if api_base else ""
    return f"""### Auto-generated GraphRAG settings
# Force parallel LLM calls - local server can handle concurrent requests
concurrent_requests: 8

completion_models:
  default_completion_model:
    model_provider: {model_provider}
    model: {model}
    auth_method: api_key
    api_key: ${{{api_key_env_var}}}{completion_api_base}
    retry:
      type: exponential_backoff

  graph_index_completion_model:
    model_provider: {model_provider}
    model: {indexing_model}
    auth_method: api_key
    api_key: ${{{api_key_env_var}}}{completion_api_base}
    retry:
      type: exponential_backoff

embedding_models:
  default_embedding_model:
    model_provider: {model_provider}
    model: {embedding_model}
    auth_method: api_key
    api_key: ${{{api_key_env_var}}}{embedding_api_base}
    retry:
      type: exponential_backoff

input:
  type: text

chunking:
  type: tokens
  size: {cfg.GRAPHRAG_CHUNK_SIZE}
  overlap: {cfg.GRAPHRAG_CHUNK_OVERLAP}
  encoding_model: cl100k_base

input_storage:
  type: file
  base_dir: "{input_dir}"

output_storage:
  type: file
  base_dir: "output"

reporting:
  type: file
  base_dir: "logs"

cache:
  type: json
  storage:
    type: file
    base_dir: "cache"

vector_store:
  type: lancedb
  db_uri: output\\lancedb

embed_text:
  embedding_model_id: default_embedding_model

extract_graph:
  completion_model_id: graph_index_completion_model
  prompt: "prompts/extract_graph.txt"
  entity_types: [organization, person, geo, event]
  max_gleanings: 0

summarize_descriptions:
  completion_model_id: graph_index_completion_model
  prompt: "prompts/summarize_descriptions.txt"
  max_length: 300

extract_graph_nlp:
  text_analyzer:
    extractor_type: regex_english

cluster_graph:
  max_cluster_size: 8

extract_claims:
  enabled: false
  completion_model_id: graph_index_completion_model
  prompt: "prompts/extract_claims.txt"
  description: "Any claims or facts that could be relevant to information discovery."
  max_gleanings: 0

community_reports:
  completion_model_id: graph_index_completion_model
  graph_prompt: "prompts/community_report_graph.txt"
  text_prompt: "prompts/community_report_text.txt"
  max_length: 1000
  max_input_length: 4000

snapshots:
  graphml: false
  embeddings: false

local_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/local_search_system_prompt.txt"
  max_context_tokens: {cfg.GRAPHRAG_LOCAL_MAX_CONTEXT_TOKENS}

global_search:
  completion_model_id: default_completion_model
  map_prompt: "prompts/global_search_map_system_prompt.txt"
  reduce_prompt: "prompts/global_search_reduce_system_prompt.txt"
  knowledge_prompt: "prompts/global_search_knowledge_system_prompt.txt"

drift_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/drift_search_system_prompt.txt"
  reduce_prompt: "prompts/drift_reduce_prompt.txt"

basic_search:
  completion_model_id: default_completion_model
  embedding_model_id: default_embedding_model
  prompt: "prompts/basic_search_system_prompt.txt"
"""


def _write_search_prompts(project_dir: Path) -> None:
    prompts_dir = project_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    for prompt_name in (
        "local_search_system_prompt.txt",
        "basic_search_system_prompt.txt",
    ):
        (prompts_dir / prompt_name).write_text(
            LOOKUP_SEARCH_PROMPT,
            encoding="utf-8",
        )


def create_graphrag_config(
    project_dir: Path,
    api_key: str,
    model: str | None = None,
    embedding_model: str | None = None,
    api_base: str | None = None,
    force: bool = False,
) -> Path:
    """Create a GraphRAG project configured for the active backend."""
    project_dir.mkdir(parents=True, exist_ok=True)

    active_model = _active_generation_model(model)
    active_indexing_model = _active_graphrag_index_model()
    active_embedding_model = _active_embedding_model(embedding_model)
    active_api_base = _active_api_base(api_base)
    active_provider = _active_graphrag_provider()
    api_key_env_var = "GRAPHRAG_API_KEY"

    input_dir = project_dir / "input"
    prompts_dir = project_dir / "prompts"
    if force or not prompts_dir.exists():
        print(f"  Initializing GraphRAG project in {project_dir} ...")
        _run_cli(
            [
                _graphrag_cli(),
                "init",
                "-r",
                str(project_dir),
                "-f",
                "-m",
                active_model,
                "-e",
                active_embedding_model,
            ],
            cwd=project_dir,
            timeout=180,
            verbose=False,
        )

    settings_path = project_dir / "settings.yaml"
    settings_path.write_text(
        _settings_yaml(
            model_provider=active_provider,
            model=active_model,
            indexing_model=active_indexing_model,
            embedding_model=active_embedding_model,
            api_base=active_api_base,
            api_key_env_var=api_key_env_var,
            input_dir=input_dir.name,
        ),
        encoding="utf-8",
    )
    (project_dir / ".env").write_text(
        f"{api_key_env_var}={api_key}\n",
        encoding="utf-8",
    )
    input_dir.mkdir(exist_ok=True)
    _write_search_prompts(project_dir)
    print(f"  Wrote GraphRAG config to {settings_path}")
    return project_dir


def has_graphrag_index(project_dir: Path) -> bool:
    """Return True when an index output directory already exists."""
    output_dir = project_dir / "output"
    required = [
        output_dir / "entities.parquet",
        output_dir / "relationships.parquet",
        output_dir / "communities.parquet",
        output_dir / "text_units.parquet",
    ]
    return all(path.exists() for path in required)


def run_graphrag_index(
    project_dir: Path,
    verbose: bool = True,
    dry_run: bool = False,
    method: str = "standard",
    use_cache: bool = True,
    timeout_seconds: int = 1800,
) -> bool:
    """Run GraphRAG indexing for the configured project.

    Output is streamed directly to the terminal (no pipe capture) to avoid
    the Windows pipe-buffer deadlock that occurs when GraphRAG emits large
    volumes of progress text.
    """
    print(f"\n  Running GraphRAG indexing in {project_dir} ...")
    args = [_graphrag_cli(), "index", "-r", str(project_dir), "-m", method]
    if dry_run:
        args.append("--dry-run")
    if not use_cache:
        args.append("--no-cache")

    try:
        # stdout/stderr=None: inherited from parent so output streams live to
        # the terminal.  This prevents the pipe-buffer deadlock on Windows and
        # lets you see real-time graphrag progress.
        result = subprocess.run(
            args,
            stdout=None,
            stderr=None,
            timeout=timeout_seconds,
            cwd=str(project_dir),
        )
        return result.returncode == 0
    except FileNotFoundError:
        print("  ERROR: graphrag CLI not found. Install with: pip install graphrag")
        return False
    except subprocess.TimeoutExpired:
        print(f"  ERROR: GraphRAG indexing timed out ({timeout_seconds}s limit)")
        return False


def run_graphrag_query(
    project_dir: Path,
    question: str,
    method: str = "local",
    response_type: str = "Single sentence",
) -> str:
    """Run a GraphRAG query using the installed CLI syntax."""
    output_dir = project_dir / "output"
    args = [
        _graphrag_cli(),
        "query",
        "-r",
        str(project_dir),
        "-m",
        method,
        "--response-type",
        response_type,
    ]
    if output_dir.exists():
        args.extend(["-d", str(output_dir)])
    args.append(question)

    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=str(project_dir),
        )
        if result.returncode == 0:
            return (result.stdout or "").strip()
        err = (result.stderr or result.stdout or "").strip()
        return f"ERROR: {err[-500:]}"
    except FileNotFoundError:
        return "ERROR: graphrag CLI not found"
    except subprocess.TimeoutExpired:
        return "ERROR: query timed out"


def run_graphrag_on_questions(
    project_dir: Path,
    questions: list[dict],
    method: str = "local",
) -> list[dict]:
    """Run GraphRAG on a list of questions and return formatted results."""
    results = []
    for i, question in enumerate(questions):
        print(f"  GraphRAG Q{i + 1}: {question['question'][:80]}...")
        answer = run_graphrag_query(project_dir, question["question"], method)
        results.append(
            {
                "question_id": question.get("question_id", f"q_{i}"),
                "question": question["question"],
                "gold_answer": question.get("answer", question.get("gold_answer", "")),
                "predicted_answer": answer,
                "method": method,
            }
        )
    return results
