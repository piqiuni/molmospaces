"""Generate isolated ROS launch files; never rewrite tracked baseline files."""

import hashlib
from pathlib import Path
import shlex
import xml.etree.ElementTree as ET

from . import VARIANTS
from .perception import REFRESH_PROFILES


def native_runner(repo):
    return repo / "scripts/InteractiveNav/run_interactive_nav_v3_ros_eval_test.zsh"


def runner_path(repo, directory, variant, refresh_profile="baseline"):
    if variant not in VARIANTS:
        raise ValueError(variant)
    return native_runner(repo) if variant == "full" and refresh_profile == "baseline" else directory / "runner.sh"


def _replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Baseline launch interface changed: expected exactly one {old!r}")
    return text.replace(old, new, 1)


def render_artifacts(repo: Path, directory: Path, variant: str, python_bin: str, refresh_profile="baseline"):
    """Return path/content pairs without creating files (also used by dry-run)."""
    runner_path(repo, directory, variant, refresh_profile)
    if refresh_profile not in REFRESH_PROFILES:
        raise ValueError(refresh_profile)
    if variant == "full" and refresh_profile == "baseline":
        return {}
    source = repo / "Interactive-Nav-SG-nav/src"
    package = Path(__file__).resolve().parent
    artifacts = {directory / "code/ablations" / path.name: path.read_text()
                 for path in sorted(package.glob("*.py"))}
    code_digest = hashlib.sha256("".join(artifacts.values()).encode()).hexdigest()
    nav_source = source / "nav_pkg/launch/molmospaces_nav_system.launch"
    nav = ET.fromstring(nav_source.read_text())
    roles = [] if variant == "full" else [("decision", "semantic_decision_py_pkg", "semantic_decision.launch", "semantic_rule_decision_node.py")]
    if variant in {"no_interaction_graph", "no_outcome_update"}:
        roles.append(("candidate", "semantic_decision_py_pkg", "semantic_decision.launch", "semantic_candidate_node.py"))
    if variant == "no_outcome_update":
        roles.append(("mapping", "semantic_mapping_py_pkg", "semantic_mapping_py.launch", "semantic_mapping_node.py"))
        roles.append(("inference", "semantic_mapping_py_pkg", "semantic_mapping_py.launch", "interaction_attribute_inference_node.py"))
    trees = {}
    for role, pkg, filename, executable in roles:
        key = (pkg, filename)
        if key not in trees:
            trees[key] = ET.fromstring((source / pkg / "launch" / filename).read_text())
        tree = trees[key]
        nodes = [node for node in tree.iter("node") if node.get("type") == executable]
        if len(nodes) != 1 or nodes[0].get("launch-prefix"):
            raise ValueError(f"Unexpected baseline node layout: {filename}/{executable}")
        nodes[0].set("launch-prefix", shlex.join([
            python_bin, str(directory / "code/ablations/node_entry.py"),
            "--repo", str(repo), "--variant", variant, "--role", role, "--",
        ]))
        ET.SubElement(nodes[0], "param", name="module_ablation", value=variant)
    if refresh_profile != "baseline":
        key = ("semantic_mapping_py_pkg", "semantic_mapping_py.launch")
        if key not in trees:
            trees[key] = ET.fromstring((source / key[0] / "launch" / key[1]).read_text())
        inference = next(n for n in trees[key].iter("node") if n.get("type") == "interaction_attribute_inference_node.py")
        for name, value in REFRESH_PROFILES[refresh_profile].items():
            param = next((p for p in inference.findall("param") if p.get("name") == name), None)
            if param is None:
                param = ET.SubElement(inference, "param", name=name)
            param.set("value", str(value))
    for (pkg, filename), tree in trees.items():
        destination = directory / filename
        includes = [inc for inc in nav.iter("include")
                    if inc.get("file") == f"$(find {pkg})/launch/{filename}"]
        if not includes:
            raise ValueError(f"Missing baseline include: {filename}")
        for inc in includes:
            inc.set("file", str(destination))
        artifacts[destination] = ET.tostring(tree, encoding="unicode") + "\n"
    nav_path = directory / "nav.launch"
    artifacts[nav_path] = ET.tostring(nav, encoding="unicode") + "\n"
    runner = native_runner(repo).read_text()
    runner = _replace_once(
        runner, 'SCRIPT_DIR=${INTERACTIVE_NAV_SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}',
        "SCRIPT_DIR=" + shlex.quote(str(native_runner(repo).parent)),
    )
    runner = _replace_once(
        runner, '"${ROS_SOURCE_DIR}/nav_pkg/launch/molmospaces_nav_system.launch"',
        shlex.quote(str(nav_path)),
    )
    # The native batch resume signature hashes this runner, so include adapter identity.
    runner += f"\n# module_ablation={variant} refresh_profile={refresh_profile} adapter_sha256={code_digest}\n"
    artifacts[directory / "runner.sh"] = runner
    return artifacts


def write_artifacts(artifacts):
    for path, content in artifacts.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as handle:
            handle.write(content)


def artifact_digests(artifacts):
    return {str(path): hashlib.sha256(content.encode()).hexdigest()
            for path, content in artifacts.items()}
