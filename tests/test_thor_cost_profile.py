import unittest
from tools.experiments.calibrated_schedule import fit_profile, signature


class CostProfileTests(unittest.TestCase):
    def test_reject_failed_or_invalid_measurements(self):
        profile={'schema_version':'thor.instruction-profile.v1','status':'PASS',
                 'checks':[{'passed':True}], 'rows':[{'opcode':1,'default_cost':8,'consumer_max_cycles':20},
                                                     {'opcode':1,'default_cost':8,'consumer_max_cycles':30}]}
        self.assertEqual(fit_profile(profile)['1:8']['median_cycles'],25)
        for changes in ({'status':'FAIL'},{'checks':[]},{'rows':[]},
                        {'rows':[{'opcode':1,'default_cost':8,'consumer_max_cycles':-1}]}):
            with self.assertRaises(ValueError):fit_profile(profile|changes)

    def test_signature_allows_only_measurement_and_scheduler_changes(self):
        meta={'model':'llama','model_provenance':{'sha':'123'},'precision':'bf16','context':1024,'batch':1,
              'native_build':{'source_hash':'abc'},'config':{'timing':True,'scheduler':'rr','name':'p','workers':20}}
        other=meta|{'config':meta['config']|{'timing':False,'scheduler':'dag','name':'c'}}
        self.assertEqual(signature(meta),signature(other))
        self.assertNotEqual(signature(meta),signature(other|{'context':128}))
        self.assertNotEqual(signature(meta),signature(other|{'config':other['config']|{'workers':40}}))


class CriticalPathTests(unittest.TestCase):
    def test_random_dags_and_locality_preserve_progress(self):
        import random
        from types import SimpleNamespace
        from megakernels.scheduler import DAG_Node, Schedule, validate_queues
        from tools.experiments.calibrated_schedule import assign_critical_path
        rng=random.Random(78)
        class Op:
            def cost(self, globs): return self.duration
            def opcode(self): return 1
        differing=0
        for _ in range(100):
            nodes=[]
            for i in range(30):
                op=Op();op.duration=rng.randint(1,100)
                nodes.append(DAG_Node(op,[n for n in nodes if rng.random()<.1]))
            schedule=Schedule(SimpleNamespace(sm_count=lambda:3),nodes,nodes[-1])
            plain=assign_critical_path(schedule)
            local=assign_critical_path(schedule,locality=True)
            differing+=plain!=local
            validate_queues(schedule,local)
            for n in nodes:
                self.assertTrue(all(n.start_time>=p.end_time for p in n.dependencies))
        self.assertGreater(differing,0)
        nodes[0].dependencies=[nodes[-1]];nodes[-1].dependencies=[nodes[0]]
        with self.assertRaises(ValueError):assign_critical_path(schedule)
