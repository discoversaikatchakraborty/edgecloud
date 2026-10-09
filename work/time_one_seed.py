import sys, time
sys.path.insert(0, "/Users/arlina/Downloads/all_projects/25_EDGECLOUD/work/pkg_v2.7")
import core
t0 = time.time()
out = core.experiment(101, log=lambda *a: None)
print("one seed of experiment() done; returned", type(out), "elapsed_s", round(time.time() - t0, 1))
