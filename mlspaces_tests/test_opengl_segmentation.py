import mujoco
import numpy as np

from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer


def test_segmentation_is_single_sample_and_preserves_rgb_and_depth():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <visual><quality offsamples="4"/></visual>
          <worldbody>
            <camera name="test_camera" pos="0 0 3"/>
            <body name="occluder" pos="0 0 0.5">
              <geom type="box" size="0.6 0.6 0.1" rgba="0.8 0.4 0.2 1"/>
            </body>
            <body name="target">
              <geom name="target_geom" type="sphere" size="0.2"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    target_id = model.geom("target_geom").id
    with MjOpenGLRenderer(model=model, width=160, height=120) as renderer:
        renderer.update(data, "test_camera")
        rgb_before = renderer.render().copy()
        renderer.enable_depth_rendering()
        depth_before = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        hidden = renderer.render().copy()
        context = renderer._segmentation_context
        assert context.offSamples == 0
        assert renderer._mjr_context.offSamples == 4
        assert model.vis.quality.offsamples == 4
        assert not np.any(hidden[..., 0] == target_id)
        np.testing.assert_array_equal(renderer.render(), hidden)
        assert renderer._segmentation_context is context
        renderer.disable_segmentation_rendering()
        np.testing.assert_array_equal(renderer.render(), rgb_before)
        renderer.enable_depth_rendering()
        np.testing.assert_array_equal(renderer.render(), depth_before)
        model.body_pos[model.body("occluder").id, 0] = 2.0
        mujoco.mj_forward(model, data)
        renderer.update(data, "test_camera")
        renderer.enable_segmentation_rendering()
        visible = renderer.render()
        assert np.count_nonzero(visible[..., 0] == target_id) > 0
        assert renderer._segmentation_context is context
    assert renderer._segmentation_context is None
    renderer.close()
