import os.path
import numpy as np
import pytest

from .. import LSP
from .. import JHMDB
from .. import HARPET
from .. import MPII

DS_PATH = "../../datasets"


class TestDataLoaders():
    def test_LSP(self):
        lsp = LSP(os.path.join(DS_PATH, "lsp"))
        # check dataset sizes and get_data accessors on different elements
        assert lsp.get_data("filenames")[0].shape == (2000, )
        assert lsp.get_data("keypoints", "train")[0].shape == (1000, 14, 3)
        d = lsp.get_data(("filenames", "keypoints"), "test")
        assert d[0].shape == (1000, )
        assert d[1].shape == (1000, 14, 3)
        # test iterator access
        it = lsp.get_iterator(("filenames", "keypoints"), "train")
        filename, keypoints = next(it)
        # check we got the correct first element
        assert filename.endswith("images/im0001.jpg")
        np.testing.assert_allclose(keypoints[0],
                                   np.array([473.95283, 552.631521, 0.]))

    def test_JHMDB(self):
        jhmdb = JHMDB(os.path.join(DS_PATH, "jhmdb"), 1)
        # check dataset sizes and get_data accessors on different single
        # elements
        assert jhmdb.get_data("filenames")[0].shape == (928, )
        # keypoints don't have a full shape here yet because the number of
        # frames per video varies
        assert jhmdb.get_data("keypoints", "train")[0].shape == (660, )
        assert jhmdb.get_data("scales", "test")[0].shape == (268, )
        assert jhmdb.get_data("actions", "test")[0].shape == (268, )
        # test iterator access
        it = jhmdb.get_iterator(
            ("filenames", "keypoints", "scales", "actions"))
        filename, keypoints, scale, action = next(it)
        # check we got the correct first element
        assert filename.endswith(
            "videos/brush_hair/Aussie_Brunette_Brushing_Long_Hair_brush_hair_u_nm_np1_ba_med_3.avi"
        )
        assert keypoints.shape == (40, 15, 2)
        assert scale.shape == (40, )
        np.testing.assert_allclose(keypoints[0, 0],
                                   np.array([125.14277428, 111.23125258]))
        assert action == 0

    def test_HARPET(self):
        harpet = HARPET(os.path.join(DS_PATH, "harpet"))
        # check dataset sizes
        assert harpet.get_data("filenames")[0].shape == (424, 3)
        assert harpet.get_data("keypoints",
                               "train")[0].shape == (297, 3, 18, 2)
        assert harpet.get_data("actions", "valid")[0].shape == (64, )
        # checked all single accessors above, now check accessing two
        # elements at once
        d = harpet.get_data(("keypoints", "actions"), "test")
        assert d[0].shape == (63, 3, 18, 2)
        assert d[1].shape == (63, )
        # check iterator access on non-existing element raises an exception
        with pytest.raises(Exception):
            it = harpet.get_iterator(
                ("filenames", "keypoints", "scales", "actions"))
            filename, keypoints, scales, action = next(it)
        # check iterator access on subset
        it = harpet.get_iterator(("filenames", "keypoints", "actions"),
                                 "train")
        filename, keypoints, action = next(it)
        # check we got the correct first element
        assert filename[0].endswith(
            "images_train/ImageSequencesBackward_1_70.jpg")
        assert keypoints.shape == (3, 18, 2)
        np.testing.assert_allclose(keypoints[0, 0],
                                   np.array([245.0161, 305.9839]))
        assert action == 0

    def test_MPII(self):
        mpii = MPII(os.path.join(DS_PATH, "mpii_human_pose"))
        # check dataset sizes and get_data accessors on different elements
        assert mpii.get_data("filenames")[0].shape == (22155, )
        # keypoint shape is not known here as some images contain several
        # people
        assert mpii.get_data("keypoints", "train")[0].shape == (15247, )
        assert mpii.get_data("scales", "test")[0].shape == (6908, )

        d = mpii.get_data(("centres", "head_bboxes"), "test")
        assert d[0].shape == (6908, )
        assert d[1].shape == (6908, )
        # test iterator access
        it = mpii.get_iterator(
            ("filenames", "keypoints", "scales", "centres", "head_bboxes"),
            "train")
        filename, keypoints, scale, centre, head_bbox = next(it)
        # check we got the correct first element
        assert filename.endswith("images/015601864.jpg")
        print(keypoints[0, 0:3])
        print(scale)
        print(centre)
        print(head_bbox)
        assert keypoints.shape == (2, 16, 3)
        np.testing.assert_allclose(
            keypoints[0, 0:3],
            np.array([[620, 394, 1], [616, 269, 1], [573, 185, 1]]))
        np.testing.assert_allclose(scale, np.array([0.33101116, 0.40451168]))
        np.testing.assert_allclose(centre, np.array([[594, 257], [952, 222]]))
        np.testing.assert_allclose(
            head_bbox, np.array([[627, 100, 706, 198], [841, 145, 902, 228]]))