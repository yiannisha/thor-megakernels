import unittest
from types import SimpleNamespace
from megakernels.scheduler import DAG_Node, Schedule, assign_dag_to_sms, round_robin_assign_to_sms


class Op:
    def __init__(self, cost):
        self.duration = cost

    def cost(self, globs):
        return self.duration

    def opcode(self):
        return 1

    def serialize(self):
        return [1]


class SchedulerTests(unittest.TestCase):
    def schedule(self, nodes, workers=2):
        return Schedule(SimpleNamespace(sm_count=lambda: workers), nodes, nodes[-1])

    def test_dependency_ready_time_and_repeat(self):
        a = DAG_Node(Op(100), [])
        b = DAG_Node(Op(1), [a])
        s = self.schedule([a, b])
        for _ in range(2):
            queues = assign_dag_to_sms(s)
            self.assertEqual(sum(map(len, queues)), 2)
            self.assertGreaterEqual(b.start_time, a.end_time)

    def test_identical_serializations_are_distinct_nodes(self):
        nodes = [DAG_Node(Op(1), []) for _ in range(5)]
        self.assertEqual(sum(map(len, assign_dag_to_sms(self.schedule(nodes)))), 5)

    def test_cycle_and_missing_producer(self):
        a = DAG_Node(Op(1), [])
        b = DAG_Node(Op(1), [a])
        with self.assertRaises(ValueError):
            assign_dag_to_sms(self.schedule([b]))
        a.dependencies = [b]
        with self.assertRaises(ValueError):
            assign_dag_to_sms(self.schedule([a, b]))

    def test_wave_cost_order_respects_dependencies_within_opcode(self):
        from megakernels.scheduler import wave_assign_to_sms, validate_queues
        a = DAG_Node(Op(1), [])
        b = DAG_Node(Op(100), [a])
        schedule = self.schedule([a, b], workers=1)
        queues = wave_assign_to_sms(schedule)
        self.assertEqual(queues, [[a.instruction, b.instruction]])
        validate_queues(schedule, queues)

    def test_random_waves_remain_dependency_safe(self):
        import random
        from megakernels.scheduler import wave_assign_to_sms, validate_queues
        rng = random.Random(19)
        for _ in range(100):
            nodes = []
            for i in range(30):
                nodes.append(DAG_Node(Op(rng.randint(1, 1000)),
                                      [n for n in nodes if rng.random() < .1]))
            for workers in (1, 3, 7):
                schedule = self.schedule(nodes, workers)
                validate_queues(schedule, wave_assign_to_sms(schedule))

    def test_zero_workers(self):
        with self.assertRaises(ValueError):
            round_robin_assign_to_sms([], 0)

class QueueValidationTests(unittest.TestCase):
    def test_reverse_queue_dependency_detected(self):
        from megakernels.scheduler import validate_queues
        a, b = DAG_Node(Op(1), []), DAG_Node(Op(1), [])
        b.dependencies = [a]
        schedule = Schedule(SimpleNamespace(sm_count=lambda: 1), [a, b], b)
        with self.assertRaises(ValueError):
            validate_queues(schedule, [[b.instruction, a.instruction]])
        validate_queues(schedule, [[a.instruction, b.instruction]])

    def test_duplicate_missing_and_random_dags(self):
        import random
        from megakernels.scheduler import validate_queues
        rng = random.Random(7)
        for _ in range(100):
            nodes = []
            for i in range(25):
                parents = [node for node in nodes if rng.random() < .12]
                nodes.append(DAG_Node(Op(rng.randint(1, 100)), parents))
            schedule = Schedule(SimpleNamespace(sm_count=lambda: 3), nodes, nodes[-1])
            queues = assign_dag_to_sms(schedule)
            validate_queues(schedule, queues)
            for node in nodes:
                for parent in node.dependencies:
                    self.assertGreaterEqual(node.start_time, parent.end_time)
            broken = [list(queue) for queue in queues]
            broken[0].append(nodes[0].instruction)
            with self.assertRaises(ValueError):
                validate_queues(schedule, broken)
            queues = [[ins for ins in q if ins is not nodes[0].instruction] for q in queues]
            with self.assertRaises(ValueError):
                validate_queues(schedule, queues)


if __name__ == '__main__':
    unittest.main()
