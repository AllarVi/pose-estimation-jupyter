from unittest import TestCase

from body_part_not_available import BodyPartPointNotAvailable
from smoother import Smoother


class TestSmoother(TestCase):

    def test_get_next_available_body_part_point_start_idx(self):
        start = 1
        body_part_data = [100, 101, 0, 0, 102, 103]

        next_available = Smoother.get_next_available_body_part_point(start, body_part_data)

        self.assertEqual(101, next_available)

    def test_get_next_available_body_part_point(self):
        start = 1
        body_part_data = [100, 0, 0, 0, 102, 103]

        next_available = Smoother.get_next_available_body_part_point(start, body_part_data)

        self.assertEqual(102, next_available)

    def test_get_next_available_body_part_point_nothing_left(self):
        start = 1
        body_part_data = [100, 0, 0, 0, 0, 0]

        self.assertRaises(BodyPartPointNotAvailable, Smoother.get_next_available_body_part_point, start, body_part_data)
