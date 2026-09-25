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
    parser.add_argument("--role", choices=("decision", "mapping", "candidate", "inference"), required=True)
    parser.add_argument("node_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    node_argv = args.node_argv[1:] if args.node_argv[:1] == ["--"] else args.node_argv
    expected = {"decision": "semantic_rule_decision_node.py", "mapping": "semantic_mapping_node.py",
                "candidate": "semantic_candidate_node.py", "inference": "interaction_attribute_inference_node.py"}[args.role]
    if not node_argv or Path(node_argv[0]).name != expected:
        parser.error(f"expected ROS executable {expected}")
    add_source_paths(args.repo.resolve())
    sys.argv = node_argv
    import rospy
    from ablations.nodes import decision_node_class, mapping_node_class
    if args.role == "decision":
        from semantic_rule_decision_node import SemanticRuleDecisionNode
        node_type = decision_node_class(SemanticRuleDecisionNode, args.variant)
    elif args.role == "mapping":
        from semantic_mapping_node import SemanticMappingNode
        node_type = mapping_node_class(SemanticMappingNode, args.variant)
    elif args.role == "candidate":
        from semantic_candidate_node import SemanticCandidateNode
        if args.variant == "no_interaction_graph":
            from ablations.flat_memory import candidate_node_class
        elif args.variant == "no_outcome_update":
            from ablations.sensory_candidates import candidate_node_class
        else:
            parser.error("candidate wrapper requires flat memory or perception-only updates")
        node_type = candidate_node_class(SemanticCandidateNode)
    else:
        if args.variant != "no_outcome_update":
            parser.error("inference wrapper is only for perception-only updates")
        from interaction_attribute_inference_node import InteractionAttributeInferenceNode
        from ablations.perception import inference_node_class
        node_type = inference_node_class(InteractionAttributeInferenceNode)
    node = node_type()
    rospy.set_param("~module_ablation", args.variant)
    if args.role == "decision":
        effective = (
            f"policy_backend={node.policy_backend} "
            f"policy={type(node.policy).__name__} "
            f"model_policy={type(node.model_policy).__name__} "
            f"curator={type(node.candidate_curator).__name__}"
        )
    elif args.role == "mapping":
        effective = (
            f"command_callback={type(node).interaction_command_callback.__qualname__} "
            f"result_callback={type(node).interaction_result_callback.__qualname__}"
        )
    else:
        effective = f"adapter={type(node).__name__}"
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
