"""Compile/run bounded timing metadata independently of ROS and sensors."""

from pathlib import Path
import shutil
import subprocess

import pytest


PACKAGE = Path(__file__).resolve().parents[4] / "Interactive-Nav-SG-nav/src/struct_mapping_pkg"


def test_ingress_timing_buffer_runtime(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ required for C++ timing buffer test")
    executable = tmp_path / "test_ingress_timing_buffer"
    subprocess.run([
        compiler, "-std=c++11", "-O2", "-pthread", "-Wall", "-Wextra", "-Werror",
        "-I", str(PACKAGE / "include"),
        str(PACKAGE / "test/test_ingress_timing_buffer.cpp"), "-o", str(executable),
    ], check=True, capture_output=True, text=True, timeout=30)
    subprocess.run([str(executable)], check=True, timeout=5)


def test_each_filter_records_before_add_and_consumes_on_success_and_failure():
    source = (PACKAGE / "src/slam_gmapping.cpp").read_text()
    for buffer, filter_name in [
        ("pointcloud_ingress_timing_", "scan_filter_"),
        ("organized_depth_ingress_timing_", "organized_depth_scan_filter_"),
    ]:
        receipt = source.index(buffer + ".record(msg.get(), ros::SteadyTime::now().toSec())")
        delivery = source.index(filter_name + "->add(msg)", receipt)
        assert delivery - receipt < 160
        assert source.count(buffer + ".takeMs(") == 2
    assert "pointcloud_ingress_connection_.disconnect();" in source
    assert "organized_depth_ingress_connection_.disconnect();" in source
    assert "PIPELINE_STAGE_MAP_SOURCE_AGE" in source


def test_pointcloud_freshness_is_rechecked_after_projection_before_mapping():
    source = (PACKAGE / "src/slam_gmapping.cpp").read_text()
    callback = source.index("SlamGMapping::pointCloudCallback")
    projection = source.index("convertPointCloudToLaserScan", callback)
    second_age_guard = source.index(
        "Drop pointcloud after projection", projection
    )
    mapper_init = source.index("if(!got_first_scan_)", projection)
    add_scan = source.index("addScan(scan, odom_pose)", projection)

    assert projection < second_age_guard < mapper_init < add_scan
    assert "enable_time_sync_guard_ && enforce_cloud_age_drop_" in source[
        projection:second_age_guard
    ]
