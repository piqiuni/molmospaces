"""Static regression checks for the odometry-locked GMapping fast path.

The physical RGB-D profile enables ``use_odom_pose_for_mapping``.  In that
mode ``processScan`` registers the current scan directly into its sole
particle map; running ``resample`` afterwards registers the same scan a second
time and emits synchronous diagnostics on every frame.  Keep this contract
visible in a lightweight test that does not require a ROS master or hardware.
"""

from __future__ import annotations

from pathlib import Path


SOURCE = (
    Path(__file__).resolve().parents[4]
    / "Interactive-Nav-SG-nav"
    / "src"
    / "openslam_gmapping"
    / "gridfastslam"
    / "gridslamprocessor.cpp"
)


def test_odom_locked_scan_does_not_resample_or_re_register() -> None:
    source = SOURCE.read_text(encoding="utf-8")

    # The only processScan call must be behind the scan-matching branch.  This
    # prevents an accidental second registerScan for physical odometry mode.
    assert source.count("resample(plainReading, adaptParticles, reading_copy);") == 1
    guarded_call = (
        "if (!m_useOdometryPose)\n"
        "\t   resample(plainReading, adaptParticles, reading_copy);"
    )
    assert guarded_call in source

    # Subsequent odometry-locked frames must not allocate a tree reading; the
    # first frame still creates the initial TNode and owns its copy.
    assert "if (!m_useOdometryPose || !m_count)" in source


def test_occ_publish_does_not_copy_the_complete_best_particle() -> None:
    source = (
        SOURCE.parents[2]
        / "struct_mapping_pkg"
        / "src"
        / "slam_gmapping.cpp"
    ).read_text(encoding="utf-8")

    assert "const GMapping::GridSlamProcessor::Particle& best =" in source
    assert "GMapping::GridSlamProcessor::Particle best =" not in source
