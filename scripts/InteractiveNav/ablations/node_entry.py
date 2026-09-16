#!/usr/bin/env python3
"""ROS launch-prefix entry; preserve the original node's remapping arguments."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ablations import VARIANTS, add_source_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS[1:], required=True)
    parser.add_argument("--role", choices=("decision", "mapping"), required=True)
    parser.add_argument("node_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    node_argv = args.node_argv[1:] if args.node_argv[:1] == ["--"] else args.node_argv
    expected = "semantic_rule_decision_node.py" if args.role == "decision" else "semantic_mapping_node.py"
    if not node_argv or Path(node_argv[0]).name != expected:
        parser.error(f"expected ROS executable {expected}")
    add_source_paths(args.repo.resolve())
    sys.argv = node_argv
    import rospy
    from ablations.nodes import decision_node_class, mapping_node_class
    if args.role == "decision":
        from semantic_rule_decision_node import SemanticRuleDecisionNode
        node_type = decision_node_class(SemanticRuleDecisionNode, args.variant)
    else:
        from semantic_mapping_node import SemanticMappingNode
        node_type = mapping_node_class(SemanticMappingNode, args.variant)
    node = node_type()
    rospy.set_param("~module_ablation", args.variant)
    if args.role == "decision":
        effective = (
            f"policy_backend={node.policy_backend} "
            f"policy={type(node.policy).__name__} "
            f"model_policy={type(node.model_policy).__name__} "
            f"curator={type(node.candidate_curator).__name__}"
        )
    else:
        effective = (
            f"command_callback={type(node).interaction_command_callback.__qualname__} "
            f"result_callback={type(node).interaction_result_callback.__qualname__}"
        )
    rospy.logwarn(
        "[module-ablation] variant=%s role=%s %s",
        args.variant,
        args.role,
        effective,
    )
    rospy.spin()
    return node


if __name__ == "__main__":
    main()
