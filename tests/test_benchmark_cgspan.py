import unittest
from cgspan_mining.cgspan import cgSpan

class CompoundBenchmarkTests(unittest.TestCase):
    """
    Before running a benchmark test
    make sure that @profile is not commented out in gSpan if memory usage info is desired
    @profile lets the memory profiler work
    """
    # The test below took longer than 12 hours to complete. Results unknown
    # def test_compound_min_graph_size_2_support_3_percent(self):
    #     graph_dataset_size = 422
    #
    #     file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
    #     supp = graph_dataset_size * 0.03
    #     min_size_graph = 2
    #     gs = gSpan(
    #         database_file_name=file_name,
    #         min_support=supp,
    #         min_num_vertices=min_size_graph
    #     )
    #     gs.run()

    def test_compound_min_graph_size_2_support_4_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.04
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_compound_min_graph_size_2_support_5_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_compound_min_graph_size_2_support_6_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.06
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_compound_min_graph_size_2_support_7_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.07
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_compound_min_graph_size_2_support_8_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.08
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_compound_min_graph_size_2_support_9_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.09
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_compound_min_graph_size_2_support_10_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.1
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_compound_min_graph_size_7_support_5_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 7
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_compound_min_graph_size_12_support_5_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 12
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_compound_min_graph_size_17_support_5_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 17
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_compound_min_graph_size_22_support_5_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 22
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_compound_min_graph_size_27_support_5_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 27
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_compound_min_graph_size_32_support_5_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 32
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_compound_min_graph_size_37_support_5_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 37
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_compound_min_graph_size_42_support_5_percent(self):
        graph_dataset_size = 422

        file_name = "../graphdata/benchmark_tests/Coumpound_422.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 42
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

class ChemicalBenchmarkTests(unittest.TestCase):

    @classmethod
    def setUpClass(self) -> None:
        self.f = open("benchmarkCloseGraphOutput.txt", "w")

    @classmethod
    def tearDownClass(self) -> None:
        self.f.close()

    def updateOutput(self, results):
        self.f.write(str(sorted(results)) + "\n")

    def convert_results_format(self, results):
        results_as_tuples = []
        for result in results:
            support, description, num_vertices = result[0], result[1], result[2]
            results_as_tuples.append((support, description, num_vertices))

        return results_as_tuples

    def test_chemical_min_graph_size_2_support_3_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.03
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_chemical_min_graph_size_2_support_4_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.04
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_chemical_min_graph_size_2_support_5_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_chemical_min_graph_size_2_support_6_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.06
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_chemical_min_graph_size_2_support_7_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.07
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_chemical_min_graph_size_2_support_8_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.08
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_chemical_min_graph_size_2_support_9_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.09
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

    def test_chemical_min_graph_size_2_support_10_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.1
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()
    #End of min support tests

    def test_chemical_min_graph_size_2(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.45
        min_size_graph = 2
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()
        cg.graph_dataset_stats()

        results = cg._report_df.to_numpy().astype(str)
        results = self.convert_results_format(results)
        self.updateOutput(results)

    #Start of min graph size tests
    def test_chemical_min_graph_size_7_support_5_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 7
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_chemical_min_graph_size_12_support_5_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 12
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_chemical_min_graph_size_17_support_5_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 17
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_chemical_min_graph_size_22_support_5_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 22
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_chemical_min_graph_size_27_support_5_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 27
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

    def test_chemical_min_graph_size_32_support_5_percent(self):
        graph_dataset_size = 340

        file_name = "../graphdata/benchmark_tests/Chemical_340.txt"
        supp = graph_dataset_size * 0.05
        min_size_graph = 32
        cg = cgSpan(
            database_file_name=file_name,
            min_support=supp,
            min_num_vertices=min_size_graph
        )
        cg.run()
        cg.time_stats()

if __name__ == '__main__':
    unittest.main()
