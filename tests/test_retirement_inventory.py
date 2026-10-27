"""Narrow AST retirement gate; the JSON contract is the reviewed classification."""

from __future__ import annotations

import ast
import json
import re
import shutil
from pathlib import Path
from typing import Final, TypedDict

import pytest

ROOT: Final = Path(__file__).resolve().parents[1]
CONTRACT: Final = Path("docs/development/api-retirement-20261027.md")
HISTORICAL_READERS: Final = (
    "src/fedference/evidence_contracts/run_schema.py::RunReceipt.from_dict",
    "src/fedference/federation/transport.py::deserialize_envelope",
    "src/fedference/federation/transport.py::deserialize_result",
    "src/fedference/federation/socket_transport.py::aggregation_config_from_replay",
)
SCIENTIFIC_CONTROLS: Final = (
    "src/fedference/aggregation.py::log_linear_pool",
    "src/fedference/aggregation.py::robust_aggregate",
    "src/fedference/aggregation.py::variational_aggregate",
    "src/fedference/aggregation_comparators.py::linear_opinion_pool",
    "src/fedference/aggregation_comparators.py::clr_geometric_median_pool",
)


class Surface(TypedDict):
    symbol: str
    replacement: str
    kind: str


class Entry(Surface):
    callsites: list[str]
    disposition: str
    status: str
    historical_reader: str
    owner: str
    scope: str
    rationale: str


class Inventory(TypedDict):
    schema_version: int
    entries: list[Entry]


class References(TypedDict):
    class_members: list[str]
    names: dict[str, list[str]]
    keywords: dict[str, list[str]]


def read_inventory(root: Path) -> Inventory:
    """Read the single JSON block, independently of surrounding prose."""
    blocks = re.findall(r"```json\n(.*?)\n```", (root / CONTRACT).read_text(), re.S)
    assert len(blocks) == 1, "expected one machine-readable inventory"
    inventory: Inventory = json.loads(blocks[0])
    assert inventory["schema_version"] == 1
    return inventory


def definitions(tree: ast.AST, prefix: str = "") -> list[tuple[str, ast.AST]]:
    """List definitions including class members and module import facades."""
    result: list[tuple[str, ast.AST]] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            name = prefix + node.name
            result.append((name, node))
            result.extend(definitions(node, name + "."))
        elif isinstance(node, ast.ImportFrom) and not prefix:
            result.extend((alias.asname or alias.name, node) for alias in node.names)
        elif isinstance(node, ast.Assign) and not prefix:
            result.extend((target.id, node) for target in node.targets if isinstance(target, ast.Name))
    return result


def discover(root: Path) -> dict[str, Surface]:
    """Find warning sites, literal legacy pops, and shared-resolver aliases."""
    result: dict[str, Surface] = {}
    for path in sorted((root / "src").rglob("*.py")):
        for name, function in definitions(ast.parse(path.read_text())):
            if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            nodes = list(ast.walk(function))
            warned = any(
                isinstance(n, ast.Name) and n.id in {"DeprecationWarning", "PendingDeprecationWarning"}
                for n in nodes
            )
            aliases: dict[str, str] = {}
            for node in nodes:
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    for old, new in re.findall(
                        r"([\w.]+)(?:\(q=\.\.\.\))? is deprecated; use ([\w.]+)", node.value
                    ):
                        aliases[old.split(".")[-1]] = new
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_resolve_compat_alias"
                ):
                    keywords = {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
                    aliases.update((old, keywords["canonical"]) for old in keywords["aliases"])
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "pop"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "legacy"
                    and node.args
                    and isinstance(node.args[0], ast.Constant)
                ):
                    aliases.setdefault(str(node.args[0].value), "see canonical signature")
                if isinstance(node, ast.For) and isinstance(node.iter, ast.Tuple):
                    for item in node.iter.elts:
                        if (
                            isinstance(item, ast.Tuple)
                            and len(item.elts) == 3
                            and all(
                                isinstance(e, ast.Constant) and isinstance(e.value, str) for e in item.elts
                            )
                        ):
                            old, new, _ = ast.literal_eval(item)
                            if warned:
                                aliases[old] = new
            if name == "rcce" and warned:
                aliases = {"q": "q_loss"}
            if warned and not aliases:
                aliases["<warning-branch>"] = "remove compatibility warning helper after caller migration"
            for alias, replacement in aliases.items():
                kind = "keyword"
                if alias == name.split(".")[-1]:
                    kind = "symbol"
                if alias == "<warning-branch>":
                    kind = "warning-helper"
                symbol = f"{path.relative_to(root).as_posix()}::{name}"
                if kind == "keyword":
                    symbol += f"({alias}=)"
                result[symbol] = {"symbol": symbol, "replacement": replacement, "kind": kind}
    direct = list(result.values())
    for path in sorted((root / "src").rglob("*.py")):
        for name, definition in definitions(ast.parse(path.read_text())):
            forwarded_calls = {
                node.func.id
                for node in ast.walk(definition)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and any(
                    keyword.arg is None
                    and isinstance(keyword.value, ast.Name)
                    and keyword.value.id == "legacy"
                    for keyword in node.keywords
                )
            }
            for surface in direct:
                target = surface["symbol"].split("::")[1]
                leaf = target.split("(")[0].split(".")[-1]
                if (
                    isinstance(definition, ast.ImportFrom)
                    and surface["kind"] == "symbol"
                    and any(a.name == leaf for a in definition.names)
                    and name == leaf
                ):
                    symbol = f"{path.relative_to(root).as_posix()}::{name}"
                    result[symbol] = {
                        "symbol": symbol,
                        "replacement": surface["replacement"],
                        "kind": "reexport",
                    }
                if (
                    isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and surface["kind"] == "keyword"
                ):
                    if leaf in forwarded_calls:
                        symbol = f"{path.relative_to(root).as_posix()}::{name}({target.split('(')[1]}"
                        result[symbol] = {
                            "symbol": symbol,
                            "replacement": surface["replacement"],
                            "kind": "keyword",
                        }
    return result


def reference_index(root: Path) -> References:
    """Index syntactic references once per tree; never cache across mutations."""
    names: dict[str, set[str]] = {}
    keywords: dict[str, set[str]] = {}
    class_members: list[str] = []
    for directory in ("src", "tests", "examples", "scripts"):
        for path in sorted((root / directory).rglob("*.py")):
            if path.name == "test_retirement_inventory.py":
                continue
            relative = path.relative_to(root).as_posix()
            tree = ast.parse(path.read_text())
            for name, definition in definitions(tree):
                if isinstance(definition, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
                    isinstance(d, ast.Name) and d.id in {"classmethod", "staticmethod"}
                    for d in definition.decorator_list
                ):
                    class_members.append(relative + "::" + name)

            def visit(node: ast.AST, scope: tuple[str, ...] = ()) -> None:
                location = relative + "::" + (".".join(scope) or "<module>")
                if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
                    name = node.id if isinstance(node, ast.Name) else node.attr
                    key = f"{relative}::{name}" if name == "__all__" else name
                    names.setdefault(key, set()).add(location)
                    if isinstance(node, ast.Attribute):
                        names.setdefault("@" + node.attr, set()).add(location)
                        if isinstance(node.value, ast.Name):
                            names.setdefault("@" + node.value.id + "." + node.attr, set()).add(location)
                if isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        names.setdefault(alias.name, set()).add(location)
                if isinstance(node, ast.Call):
                    name = getattr(node.func, "id", getattr(node.func, "attr", ""))
                    for keyword in node.keywords:
                        if keyword.arg:
                            keywords.setdefault(f"{name}:{keyword.arg}", set()).add(location)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    scope = (*scope, node.name)
                for child in ast.iter_child_nodes(node):
                    visit(child, scope)

            visit(tree)
    return {
        "class_members": sorted(class_members),
        "names": {k: sorted(v) for k, v in names.items()},
        "keywords": {k: sorted(v) for k, v in keywords.items()},
    }


def callsites(index: References, surface: Surface) -> list[str]:
    """Conservative syntactic references: names are not type-resolved."""
    target = surface["symbol"].split("::")[1]
    keyword = target.split("(")[1][:-2] if "(" in target else None
    qualified = target.split("(")[0]
    leaf = qualified.split(".")[-1]
    if leaf == "__init__":
        leaf = qualified.split(".")[-2]
    if keyword is not None:
        return index["keywords"].get(f"{leaf}:{keyword}", [])
    if surface["symbol"] in index["class_members"]:
        leaf = "@" + qualified
    elif "." in qualified and not qualified.endswith(".__init__"):
        leaf = "@" + leaf
    if leaf == "__all__":
        leaf = surface["symbol"].split("::")[0] + "::__all__"
    return index["names"].get(leaf, [])


def assert_contract(root: Path, inventory: Inventory) -> None:
    """Fail on unclassified surfaces, stale callers, or removed retained entries."""
    entries = inventory["entries"]
    by_symbol = {entry["symbol"]: entry for entry in entries}
    assert len(by_symbol) == len(entries), "duplicate inventory symbol"
    for required, disposition in (
        (HISTORICAL_READERS, "retain-history"),
        (SCIENTIFIC_CONTROLS, "retain-science"),
    ):
        for symbol in required:
            assert symbol in by_symbol and by_symbol[symbol]["disposition"] == disposition, (
                f"missing protected classification: {symbol}"
            )
    current = discover(root)
    index = reference_index(root)
    assert not current.keys() - by_symbol.keys(), (
        f"unclassified deprecated surfaces: {sorted(current.keys() - by_symbol.keys())}"
    )
    for entry in entries:
        symbol = entry["symbol"]
        assert entry["disposition"] in {
            "remove-deprecated",
            "retain-science",
            "owner-decision",
            "retain-history",
        }, symbol
        assert entry["status"] in {"present", "retired"}, symbol
        assert all(
            entry[key] for key in ("replacement", "owner", "scope", "rationale", "historical_reader")
        ), symbol
        if entry["disposition"] == "remove-deprecated":
            assert (symbol in current) == (entry["status"] == "present"), f"stale retirement status: {symbol}"
            if symbol in current:
                assert entry["replacement"] == current[symbol]["replacement"], f"replacement drift: {symbol}"
        else:
            path, name = symbol.split("::")
            names = {name for name, _ in definitions(ast.parse((root / path).read_text()))}
            assert (name in names) == (entry["status"] == "present"), (
                f"required retained surface missing: {symbol}"
            )
        if entry["status"] == "retired":
            path, target = symbol.split("::")
            for name, definition in definitions(ast.parse((root / path).read_text())):
                if "(" not in target:
                    assert name != target, f"retired symbol still defined: {symbol}"
                elif name == target.split("(")[0] and isinstance(
                    definition, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    keyword = target.split("(")[1][:-2]
                    arguments = (
                        definition.args.posonlyargs + definition.args.args + definition.args.kwonlyargs
                    )
                    assert keyword not in {arg.arg for arg in arguments}, (
                        f"retired keyword still accepted: {symbol}"
                    )
        if entry["disposition"] in {"retain-science", "retain-history", "owner-decision"}:
            assert entry["status"] == "present", f"unauthorized retained-surface retirement: {symbol}"
        assert entry["callsites"] == callsites(index, entry), f"caller inventory drift: {symbol}"
    evidence = ast.parse((root / "src/fedference/evidence_contracts/schema.py").read_text())
    transport = ast.parse((root / "src/fedference/federation/transport.py").read_text())
    for tree, name, expected in (
        (evidence, "LEGACY_EVIDENCE_SCHEMA_VERSION", "1.1"),
        (transport, "PROTOCOL_VERSION", 1),
    ):
        values = [
            ast.literal_eval(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)
        ]
        assert values == [expected], f"historical format changed without migration: {name}"


def test_inventory_covers_deprecated_runtime_symbols() -> None:
    """Current source and recorded callers agree with the reviewed JSON."""
    assert_contract(ROOT, read_inventory(ROOT))


def test_unclassified_symbol_or_historical_reader_removal(tmp_path: Path) -> None:
    """Real disposable source copies reject additions and protected removals."""
    for directory in ("src", "tests", "examples", "scripts"):
        shutil.copytree(ROOT / directory, tmp_path / directory, ignore=shutil.ignore_patterns("__pycache__"))
    inventory = read_inventory(ROOT)
    injected = tmp_path / "src/fedference/unclassified_alias.py"
    injected.write_text(
        "import warnings\ndef unclassified_alias():\n    warnings.warn('retire me', DeprecationWarning)\n"
    )
    with pytest.raises(AssertionError, match="unclassified deprecated surfaces") as failure:
        assert_contract(tmp_path, inventory)
    print(f"UNCLASSIFIED_REJECTED: {failure.value}")
    injected.unlink()
    reader = tmp_path / "src/fedference/federation/transport.py"
    reader.write_text(reader.read_text().replace("def deserialize_envelope(", "def removed_reader("))
    with pytest.raises(AssertionError, match="required retained surface missing") as failure:
        assert_contract(tmp_path, inventory)
    print(f"HISTORICAL_READER_REMOVAL_REJECTED: {failure.value}")


def test_caller_identity_ignores_layout_and_unrelated_stores(tmp_path: Path) -> None:
    for directory in ("src", "tests", "examples", "scripts"):
        (tmp_path / directory).mkdir()
    source = tmp_path / "src/consumer.py"
    source.write_text("def consumer():\n    return old()\n")
    unrelated = tmp_path / "src/unrelated.py"
    unrelated.write_text("__all__ = []\nold = 42\n")
    surface = Surface(symbol="src/api.py::old", replacement="new", kind="symbol")
    before = callsites(reference_index(tmp_path), surface)
    assert before == ["src/consumer.py::consumer"]
    exports = Surface(symbol="src/api.py::__all__", replacement="exports", kind="symbol")
    assert callsites(reference_index(tmp_path), exports) == []
    source.write_text("\n\n" + source.read_text())
    assert callsites(reference_index(tmp_path), surface) == before
    source.write_text(source.read_text().replace("old()", "new()"))
    assert callsites(reference_index(tmp_path), surface) == []


def test_member_callers_exclude_local_names_and_other_classes(tmp_path: Path) -> None:
    for directory in ("src", "tests", "examples", "scripts"):
        (tmp_path / directory).mkdir()
    (tmp_path / "src/api.py").write_text(
        "class First:\n    @classmethod\n    def from_dict(cls, raw): return cls()\n"
        "class Second:\n    @classmethod\n    def from_dict(cls, raw): return cls()\n"
    )
    source = tmp_path / "src/consumer.py"
    source.write_text(
        "from api import First, Second\ndef consumer():\n"
        "    beliefs = [1]\n    Second.from_dict({})\n    return beliefs\n"
    )
    property_surface = Surface(symbol="src/api.py::Episode.beliefs", replacement="posterior", kind="symbol")
    class_surface = Surface(symbol="src/api.py::First.from_dict", replacement="from_dict", kind="symbol")
    assert callsites(reference_index(tmp_path), property_surface) == []
    assert callsites(reference_index(tmp_path), class_surface) == []
    source.write_text(source.read_text().replace("Second.from_dict", "First.from_dict"))
    assert callsites(reference_index(tmp_path), class_surface) == ["src/consumer.py::consumer"]
    source.write_text(source.read_text() + "def read_episode(episode):\n    return episode.beliefs\n")
    assert callsites(reference_index(tmp_path), property_surface) == ["src/consumer.py::read_episode"]
