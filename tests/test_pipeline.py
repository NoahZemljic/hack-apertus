import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import run_pipeline
from starter.scene_io import Scene
from src.compute_geometry import compute_objects, compute_relations, measure_width, object_contour
from src.create_scene_graph import create_scene_graph
from src.perception import QwenPerception, PerceptionGenerationError
from src.pipeline import RoboticsPipeline
from src.apertus import ApertusPlanner
from src.validation import validate_perception, validate_scene_graph
from src.visualization import write_scene_overlay


def scene():
    transform = np.diag([1., -1., -1., 1.])
    transform[2, 3] = 1004
    return Scene('synthetic', np.full((240, 320, 3), 30, np.uint8), None, None,
                 np.array([[1000., 0, 160], [0, 1000., 120], [0, 0, 1.]]),
                 transform, 85, [('synthetic_i1', 'simple', 'Pick the red part')])


def detection(label='red_part', bbox=None, target=None):
    return {'label': label, 'bbox_2d': bbox or [75, 70, 245, 170],
            'target_pixel': target or [160, 120]}


def rectangle(width):
    return np.array([[[100, 60]], [[220, 60]], [[220, 60 + width]], [[100, 60 + width]]], np.float32)


class VisualizationTests(unittest.TestCase):
    def test_overlay_draws_box_and_target_without_changing_source(self):
        rgb = np.full((120, 160, 3), [10, 20, 30], dtype=np.uint8)
        original = rgb.copy()
        obj = dict(detection(bbox=[20, 30, 80, 90], target=[50, 60]), id='red_part_1')

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'overlays' / 'scene.detections.png'
            write_scene_overlay(rgb, [obj], path)
            overlay = cv2.imread(str(path))

        np.testing.assert_array_equal(rgb, original)
        self.assertEqual(overlay.shape, rgb.shape)
        background = [30, 20, 10]
        np.testing.assert_array_equal(overlay[110, 150], background)
        np.testing.assert_array_equal(overlay[60, 80], background)
        self.assertFalse(np.array_equal(overlay[60, 20], background))
        self.assertFalse(np.array_equal(overlay[60, 79], background))
        self.assertFalse(np.array_equal(overlay[60, 50], background))


class GeometryTests(unittest.TestCase):
    def test_rotated_segment_uses_short_side(self):
        sample = scene()
        outline = cv2.boxPoints(((160, 120), (120, 40), 30)).astype(np.int32)
        cv2.fillConvexPoly(sample.rgb, outline, (230, 35, 30))
        contour = object_contour(sample.rgb, [95, 65, 225, 175], [160, 120])
        width, measurement_rectangle, width_line = measure_width(sample, contour, [160, 120])
        self.assertAlmostEqual(width, 40, delta=2)
        self.assertEqual(measurement_rectangle.shape, (4, 2))
        self.assertEqual(width_line.shape, (2, 2))
        self.assertAlmostEqual(float(np.linalg.norm(width_line[1] - width_line[0])), width, places=3)
        self.assertGreater(cv2.boundingRect(contour)[3], 80)

    def test_target_selects_one_of_disconnected_foreground_regions(self):
        sample = scene()
        cv2.rectangle(sample.rgb, (90, 90), (120, 120), (230, 35, 30), -1)
        cv2.rectangle(sample.rgb, (160, 90), (220, 130), (230, 35, 30), -1)
        contour = object_contour(sample.rgb, [80, 80, 230, 140], [190, 110])
        width, measurement_rectangle, width_line = measure_width(sample, contour, [190, 110])
        self.assertAlmostEqual(width, 40, delta=1)
        self.assertEqual(measurement_rectangle.shape, (4, 2))
        self.assertEqual(width_line.shape, (2, 2))
        self.assertGreaterEqual(float(contour[:, 0, 0].min()), 160)

    def test_graspability_threshold_includes_exactly_85(self):
        for width, expected in [(85, True), (85.01, False)]:
            with self.subTest(width=width), patch('src.compute_geometry.object_contour', return_value=rectangle(width)):
                measured = compute_objects(scene(), {'objects': [detection()]})[0]
                self.assertEqual(measured['graspable'], expected)
                self.assertAlmostEqual(measured['width_mm'], width, places=3)

    def test_target_depth_sets_measurement_scale(self):
        sample = scene()
        sample.depth_mm = np.full((240, 320), 1000, np.uint16)
        sample.depth_mm[115:126, 155:166] = 500
        width, measurement_rectangle, width_line = measure_width(sample, rectangle(80), [160, 120])
        self.assertAlmostEqual(width, 40, places=3)
        self.assertEqual(measurement_rectangle.shape, (4, 2))
        self.assertAlmostEqual(float(np.linalg.norm(width_line[1] - width_line[0])), 80, places=3)

    def test_missing_calibration_does_not_invent_graspability(self):
        sample = scene()
        sample.base_T_camera = None
        with patch('src.compute_geometry.object_contour', return_value=rectangle(80)):
            measured = compute_objects(sample, {'objects': [detection()]})[0]
        self.assertNotIn('width_mm', measured)
        self.assertNotIn('graspable', measured)
        self.assertIn('calibration', measured['measurement_error'])

    def test_out_of_box_target_is_preserved_with_unknown_width(self):
        source = detection(target=[10, 10])
        measured = compute_objects(scene(), {'objects': [source]})[0]
        self.assertEqual(measured['target_pixel'], [10, 10])
        self.assertIn('measurement_error', measured)
        self.assertNotIn('width_mm', measured)

    def test_repeated_label_ids_are_assigned_by_code(self):
        sources = {'objects': [detection(target=[1, 1]), detection(target=[2, 2])]}
        objects = compute_objects(scene(), sources)
        self.assertEqual([o['id'] for o in objects], ['red_part_1', 'red_part_2'])
        self.assertNotIn('id', sources['objects'][0])


class RelationTests(unittest.TestCase):
    def objects(self):
        return [dict(detection('gear', [0, 0, 80, 80], [20, 20]), id='gear_1'),
                dict(detection('grey_tray', [10, 10, 100, 100], [70, 70]), id='grey_tray_1')]

    def test_inside_replaces_directions_to_the_tray(self):
        relations = compute_relations(self.objects())
        edges = {(r['subject'], r['relation'], r['object']) for r in relations}
        self.assertEqual(edges, {('gear_1', 'inside', 'grey_tray_1')})
        self.assertEqual(compute_relations(list(reversed(self.objects()))), relations)
        self.assertTrue(all(0 <= r['confidence'] <= 1 for r in relations))

    def test_separated_boxes_have_no_mirrored_duplicates(self):
        objects = [
            dict(detection('red_cube', [0, 0, 20, 20], [10, 10]), id='red_cube_1'),
            dict(detection('green_mug', [40, 60, 60, 80], [50, 70]), id='green_mug_1'),
        ]
        relations = compute_relations(objects)
        self.assertEqual(len(relations), 2)
        self.assertEqual({r['relation'] for r in relations}, {'image_left_of', 'image_up_from'})
        self.assertTrue(all(r['subject'] == 'red_cube_1' for r in relations))
        self.assertTrue(all(r['object'] == 'green_mug_1' for r in relations))

        reversed_relations = compute_relations(list(reversed(objects)))
        self.assertEqual(len(reversed_relations), 2)
        self.assertEqual({r['relation'] for r in reversed_relations}, {'image_right_of', 'image_down_from'})
        self.assertEqual(
            [r['confidence'] for r in reversed_relations],
            [r['confidence'] for r in relations],
        )

        objects[0]['target_pixel'] = [19, 19]
        objects[1]['target_pixel'] = [40, 60]
        self.assertEqual(compute_relations(objects), relations)

    def test_tall_tray_beside_cube_has_no_vertical_relation(self):
        objects = [
            dict(detection('grey_tray', [0, 0, 100, 1000], [50, 800]), id='grey_tray_1'),
            dict(detection('red_cube', [130, 10, 170, 50], [150, 30]), id='red_cube_1'),
        ]
        relations = compute_relations(objects)
        edges = {(r['subject'], r['relation'], r['object']) for r in relations}
        self.assertEqual(edges, {('grey_tray_1', 'image_left_of', 'red_cube_1')})

    def test_objects_in_same_tray_keep_their_relative_direction(self):
        objects = [
            dict(detection('grey_tray', [0, 0, 200, 200], [20, 160]), id='grey_tray_1'),
            dict(detection('green_mug', [100, 20, 150, 70], [120, 40]), id='green_mug_1'),
            dict(detection('yellow_gear', [100, 90, 130, 120], [115, 100]), id='yellow_gear_1'),
        ]
        relations = compute_relations(objects)
        self.assertEqual(len(relations), 3)
        self.assertEqual({(r['subject'], r['relation'], r['object']) for r in relations}, {
            ('green_mug_1', 'inside', 'grey_tray_1'),
            ('yellow_gear_1', 'inside', 'grey_tray_1'),
            ('green_mug_1', 'image_up_from', 'yellow_gear_1'),
        })

    def test_inside_uses_target_not_whole_box(self):
        relations = compute_relations(self.objects())
        self.assertEqual(sum(r['relation'] == 'inside' for r in relations), 1)

    def test_tray_edges_use_half_open_bounds(self):
        for point, inside in [([10, 10], True), ([99, 99], True), ([100, 50], False), ([50, 100], False)]:
            with self.subTest(point=point):
                objects = self.objects()
                objects[0]['target_pixel'] = point
                actual = any(r['relation'] == 'inside' for r in compute_relations(objects))
                self.assertEqual(actual, inside)

    def test_overlapping_boxes_produce_no_direction_and_mat_is_not_tray(self):
        objects = self.objects()
        objects[1]['label'] = 'black_work_mat'
        self.assertEqual(compute_relations(objects), [])


class ValidationTests(unittest.TestCase):
    def test_semantic_contradictions_are_not_validated(self):
        obj = {'id': 'gear_1', 'label': 'gear', 'target_pixel': [-1, 20000],
               'width_mm': 100, 'graspable': True}
        graph = {'objects': [obj, obj], 'relations': [
            {'subject': 'missing', 'relation': 'inside', 'object': 'missing', 'confidence': 0.5}]}
        self.assertIs(validate_scene_graph(graph), graph)

    def test_schema_rejects_wrong_types_extra_fields_and_old_relation(self):
        base = {'objects': [{'id': 'gear_1', 'label': 'gear', 'target_pixel': [10, 20]}], 'relations': []}
        variants = []
        for field, value in [('target_pixel', [True, 20]), ('target_pixel', [10]), ('confidence', 0.9)]:
            graph = copy.deepcopy(base)
            graph['objects'][0][field] = value
            variants.append(graph)
        graph = copy.deepcopy(base)
        graph['relations'] = [{'subject': 'gear_1', 'object': 'gear_1', 'relation': 'on', 'confidence': .9}]
        variants.append(graph)
        for graph in variants:
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                validate_scene_graph(graph)

    def test_non_json_numbers_are_rejected(self):
        graph = {'objects': [], 'relations': [{'subject': 'a', 'object': 'b', 'relation': 'image_left_of', 'confidence': float('nan')}]}
        with self.assertRaises(ValueError):
            validate_scene_graph(graph)

    def test_model_contract_contains_no_derived_fields(self):
        source = {'objects': [detection()]}
        validate_perception(source)
        source['objects'][0]['graspable'] = True
        with self.assertRaises(ValueError):
            validate_perception(source)


class FakeVLM:
    def __init__(self, replies, normalized=True):
        self.replies = iter(replies)
        self.normalized = normalized
        self.requests = []

    def load_model(self):
        pass

    def uses_normalized_coordinates(self):
        return self.normalized

    def run_inference(self, messages):
        self.requests.append(messages)
        return next(self.replies)


class PipelineTests(unittest.TestCase):
    def test_cli_runs_only_the_selected_workflow(self):
        cases = [(False, '{"objects": []}', 0), (True, '{"objects": []}', 0)]

        for perception_only, reply, expected_status in cases:
            with self.subTest(perception_only=perception_only, reply=reply):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    scene_dir = path / 'synthetic'
                    scene_dir.mkdir()
                    (scene_dir / 'rgb.png').touch()

                    sample = scene()
                    sample.instructions.append(('synthetic_i2', 'simple', 'Pick another part'))
                    vlm = FakeVLM([reply], normalized=False)
                    planner = Mock()
                    planner.create_plan.return_value = ({
                        'decision': 'ask_user', 'steps': [],
                        'question': 'Which object?', 'reason': None,
                    }, [])

                    output = path / 'outputs' / 'results.json'
                    arguments = ['run_pipeline.py', '--data', str(path),
                                 '--out', str(output), '--qwen-validation-retries', '0']
                    if perception_only:
                        arguments.append('--perception-only')

                    with (
                        patch('sys.argv', arguments),
                        patch.object(run_pipeline, 'load_scene', return_value=sample),
                        patch.object(run_pipeline, 'QwenVLM', return_value=vlm),
                        patch.object(run_pipeline, 'ApertusPlanner', return_value=planner),
                        redirect_stdout(StringIO()) as console,
                    ):
                        status = run_pipeline.main()

                    self.assertEqual(status, expected_status)
                    self.assertEqual(console.getvalue(), '')
                    self.assertEqual(len(vlm.requests), 1)

                    graph_directory = output.parent / 'scene_graphs'
                    self.assertEqual(
                        sorted(file.name for file in graph_directory.iterdir()),
                        ['synthetic.detections.png', 'synthetic.json'],
                    )
                    overlay = cv2.imread(str(graph_directory / 'synthetic.detections.png'))
                    np.testing.assert_array_equal(overlay, cv2.cvtColor(sample.rgb, cv2.COLOR_RGB2BGR))
                    self.assertFalse(output.with_suffix('.raw.jsonl').exists())

                    if perception_only:
                        planner.create_plan.assert_not_called()
                        self.assertFalse(output.exists())
                    else:
                        self.assertEqual(planner.create_plan.call_count, 2)
                        submission = json.loads(output.read_text())
                        self.assertEqual(len(submission['results']), 2)

    def test_perception_failure_raises_without_logs_in_both_modes(self):
        for perception_only in (False, True):
            with self.subTest(perception_only=perception_only):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory)
                    first = scene()
                    second = scene()
                    second.name = 'second'
                    vlm = FakeVLM(['{}', '{"objects": []}'], normalized=False)
                    planner = Mock()
                    pipeline = RoboticsPipeline(QwenPerception(vlm, 0), planner, path)

                    with redirect_stdout(StringIO()) as console:
                        with self.assertRaises(PerceptionGenerationError):
                            pipeline.run(
                                [(path, first), (path, second)],
                                perception_only=perception_only,
                            )

                    self.assertEqual(console.getvalue(), '')
                    self.assertEqual(len(vlm.requests), 1)
                    planner.create_plan.assert_not_called()
                    self.assertEqual(list(path.iterdir()), [])

    def test_normalized_qwen_coordinates_are_converted_once(self):
        source = {'objects': [detection(bbox=[100, 200, 500, 600], target=[250, 350])]}
        vlm = FakeVLM([json.dumps(source)])
        result, _ = QwenPerception(vlm).detect(Path('image.png'), 2000, 1000)
        self.assertEqual(result['objects'][0]['bbox_2d'], [200, 200, 1000, 600])
        self.assertEqual(result['objects'][0]['target_pixel'], [500, 350])
        self.assertEqual(len(vlm.requests), 1)
        self.assertEqual(vlm.requests[0][0]['role'], 'system')

    def test_format_retry_and_original_pixel_coordinates(self):
        source = {'objects': [detection()]}
        vlm = FakeVLM(['not json', json.dumps(source)], normalized=False)
        result, replies = QwenPerception(vlm, 1).detect(Path('image.png'), 320, 240)
        self.assertEqual(result, source)
        self.assertEqual(len(replies), 2)
        with self.assertRaises(PerceptionGenerationError):
            QwenPerception(FakeVLM(['{}']), 0).detect(Path('image.png'), 320, 240)

    def test_detection_geometry_graph_plan_and_cache(self):
        sample = scene()
        sample.instructions.append(('synthetic_i2', 'simple', 'Pick the part again'))
        cv2.rectangle(sample.rgb, (100, 90), (220, 150), (230, 35, 30), -1)
        perception = QwenPerception(FakeVLM([json.dumps({'objects': [detection()]})], normalized=False))

        class Planner:
            def __init__(self):
                self.graphs = []

            def create_plan(self, graph, instruction):
                self.graphs.append(graph)
                return {'decision': 'plan', 'steps': [{'action': 'pick', 'object_id': 'red_part_1'}],
                        'question': None, 'reason': None}, []

        planner = Planner()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pipeline = RoboticsPipeline(perception, planner, path)
            results = pipeline.run([(path, sample)], None)
            self.assertEqual(len(results), 2)
            self.assertEqual(len(perception.vlm.requests), 1)
            self.assertIs(planner.graphs[0], planner.graphs[1])
            self.assertEqual(results[0]['target_pixel'], [160, 120])
            graph = planner.graphs[0]
            self.assertEqual(set(graph), {'objects', 'relations'})
            self.assertEqual(set(graph['objects'][0]), {'id', 'label', 'target_pixel', 'width_mm', 'graspable'})
            self.assertTrue(graph['objects'][0]['graspable'])
            self.assertEqual(json.loads((path / 'synthetic.json').read_text()), graph)
            overlay_path = path / 'synthetic.detections.png'
            overlay = cv2.imread(str(overlay_path))
            self.assertEqual(overlay.shape, sample.rgb.shape)
            self.assertFalse(np.array_equal(overlay[120, 75], sample.rgb[120, 75]))
            overlay_modified = overlay_path.stat().st_mtime_ns
            self.assertEqual(pipeline.run([(path, sample)]), results)
            self.assertEqual(len(perception.vlm.requests), 1)
            self.assertEqual(overlay_path.stat().st_mtime_ns, overlay_modified)

    def test_apertus_receives_system_prompt_and_compact_graph(self):
        planner = ApertusPlanner()
        graph = {'objects': [], 'relations': []}
        response = {'decision': 'ask_user', 'steps': [], 'question': 'Which object?', 'reason': None}
        with patch.object(planner, 'generate_response', return_value=json.dumps(response)) as server:
            actual, _ = planner.create_plan(graph, 'Pick the object closest to the mug')
        self.assertEqual(actual, response)
        server.assert_called_once()
        prompt = server.call_args.args[0][0]['content']
        self.assertIn(json.dumps(graph, separators=(',', ':')), prompt)
        self.assertIn('Instruction: "Pick the object closest to the mug"', prompt)

    def test_planning_failure_raises_without_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'synthetic.json').write_text('{"objects": [], "relations": []}')
            planner = Mock()
            planner.create_plan.side_effect = ValueError('Invalid plan')
            pipeline = RoboticsPipeline(Mock(), planner, path)

            with self.assertRaisesRegex(ValueError, 'Invalid plan'):
                pipeline.run([(path, scene())])

            self.assertEqual([file.name for file in path.iterdir()], ['synthetic.json'])

    def test_empty_scene(self):
        self.assertEqual(create_scene_graph([], compute_relations([])), {'objects': [], 'relations': []})


if __name__ == '__main__':
    unittest.main()
