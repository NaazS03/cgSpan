"""Implementation of closeGraph."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import collections
import copy
import itertools
import time
import enum

from graph import AUTO_EDGE_ID
from graph import Graph
from graph import VACANT_GRAPH_ID
from graph import VACANT_VERTEX_LABEL
from graph import DatabaseGraph
from graph import FrequentGraph
from graph import DFSlabel
from graph import EdgeDirection
from dfscode import DFSedge
from dfscode import DFScode
from trie import Trie

import pandas as pd


def record_timestamp(func):
    """Record timestamp before and after call of `func`."""

    def deco(self):
        self.timestamps[func.__name__ + '_in'] = time.time()
        func(self)
        self.timestamps[func.__name__ + '_out'] = time.time()

    return deco

class PDFS(object):
    """PDFS class."""

    def __init__(self, gid=VACANT_GRAPH_ID, edge=None, prev=None):
        """Initialize PDFS instance."""
        self.gid = gid
        self.edge = edge
        self.prev = prev

    def _root_eid(self):
        pdfs = self
        while pdfs.prev is not None:
            pdfs = pdfs.prev

        return pdfs.edge.eid


class Projected(list):
    """Projected is a list of PDFS.

    Each element of Projected is a projection one frequent graph in one
    original graph.
    """

    def __init__(self):
        """Initialize Projected instance."""
        super(Projected, self).__init__()

    def push_back(self, gid, edge, prev):
        """Update this Projected instance."""
        self.append(PDFS(gid, edge, prev))
        return self


class History(object):
    """History class."""

    def __init__(self, g, pdfs):
        """Initialize History instance."""
        super(History, self).__init__()
        self.edges = list()
        self.vertices_used = collections.defaultdict(int)
        self.edges_used = collections.defaultdict(int)
        if pdfs is None:
            return
        while pdfs:
            e = pdfs.edge
            self.edges.append(e)
            (self.vertices_used[e.frm],
             self.vertices_used[e.to],
             self.edges_used[e.eid]) = 1, 1, 1

            pdfs = pdfs.prev
        self.edges = self.edges[::-1]

    def has_vertex(self, vid):
        """Check if the vertex with vid exists in the history."""
        return self.vertices_used[vid] == 1

    def has_edge(self, eid):
        """Check if the edge with eid exists in the history."""
        return self.edges_used[eid] == 1


class closeGraph(object):
    """`closeGraph` algorithm."""

    def __init__(self,
                 database_file_name,
                 min_support=10,
                 min_num_vertices=1,
                 max_num_vertices=float('inf'),
                 max_ngraphs=float('inf'),
                 is_undirected=True,
                 verbose=False,
                 visualize=False,
                 where=False):
        """Initialize closeGraph instance."""
        self._database_file_name = database_file_name
        self.graphs = dict()
        self._max_ngraphs = max_ngraphs
        self._is_undirected = is_undirected
        self._min_support = min_support
        self._min_num_vertices = min_num_vertices
        self._max_num_vertices = max_num_vertices
        self._DFScode = DFScode()
        self._support = 0
        # Include subgraphs with
        # any num(but >= 2, <= max_num_vertices) of vertices.
        self._frequent_subgraphs = list()
        #self._early_termination_failure_causing_graphs = dict()
        self._counter = itertools.count()
        self._verbose = verbose
        self._visualize = visualize
        self._where = where
        self.timestamps = dict()
        if self._max_num_vertices < self._min_num_vertices:
            print('Max number of vertices can not be smaller than '
                  'min number of that.\n'
                  'Set max_num_vertices = min_num_vertices.')
            self._max_num_vertices = self._min_num_vertices
        self._report_df = pd.DataFrame()
        self.closed_graphs_hash_table = dict()
        self._report_df_cumulative = pd.DataFrame()
        self.i = 0
        self._current_frequent_gid = -1
        #self.etfDFScodes = list()
        self.trie = Trie()

    def time_stats(self):
        """Print stats of time."""
        func_names = ['_read_graphs', 'run']
        time_deltas = collections.defaultdict(float)
        for fn in func_names:
            time_deltas[fn] = round(
                self.timestamps[fn + '_out'] - self.timestamps[fn + '_in'],
                2
            )

        # print('Read:\t{} s'.format(time_deltas['_read_graphs']))
        # print('Mine:\t{} s'.format(
        #     time_deltas['run'] - time_deltas['_read_graphs']))
        # print('Total:\t{} s'.format(time_deltas['run']))

        print('****************************')
        print('Time Statistics')
        print('Read:\t{} s'.format(time_deltas['_read_graphs']))
        print('CloseGraph:\t{} s'.format(round(time_deltas['run'] - time_deltas['_read_graphs'], 2)))
        print('Total:\t{} s'.format(time_deltas['run']))
        print('****************************')

        return self

    def graph_dataset_stats(self):
        """Print the number of graphs and closed graphs in the provided dataset"""
        num_closed_graphs = len(self._frequent_subgraphs)
        print("The total number of closed graphs found were: {}".format(num_closed_graphs))

    @record_timestamp
    def _read_graphs(self):
        self.graphs = dict()
        with codecs.open(self._database_file_name, 'r', 'utf-8') as f:
            lines = [line.strip() for line in f.readlines()]
            tgraph, graph_cnt = None, 0
            for i, line in enumerate(lines):
                cols = line.split(' ')
                if cols[0] == 't':
                    if tgraph is not None:
                        self.graphs[graph_cnt] = tgraph
                        graph_cnt += 1
                        tgraph = None
                    if cols[-1] == '-1' or graph_cnt >= self._max_ngraphs:
                        break
                    tgraph = DatabaseGraph(graph_cnt,
                                           is_undirected=self._is_undirected,
                                           eid_auto_increment=True)
                elif cols[0] == 'v':
                    tgraph.add_vertex(cols[1], cols[2])
                elif cols[0] == 'e':
                    tgraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], cols[3])
            # adapt to input files that do not end with 't # -1'
            if tgraph is not None:
                self.graphs[graph_cnt] = tgraph
        return self

    def _write_graphs(self):
        t = ''
        for index in range(len(self.graphs)):
            tgraph = self.graphs[index]
            display_str = 't # {}\n'.format(tgraph.gid)

            min_vid = -1
            for vid in tgraph.vertices.keys():
                if min_vid == -1:
                    min_vid = int(vid)
                else:
                    if int(vid) < min_vid:
                        min_vid = int(vid)

            for vid in tgraph.vertices:
                display_str += 'v {} {}\n'.format(int(vid) - min_vid, tgraph.vertices[vid].vlb)
            for frm in tgraph.vertices:
                edges = tgraph.vertices[frm].edges
                for to in edges:
                    if tgraph.is_undirected:
                        if frm < to:
                            display_str += 'e {} {} {}\n'.format(
                                int(frm) - min_vid, int(to) - min_vid, edges[to].elb)
                    else:
                        display_str += 'e {} {} {}\n'.format(int(frm) - min_vid, int(to) - min_vid, edges[to].elb)
            t = t + display_str

        return t

    @record_timestamp
    def _generate_1edge_frequent_subgraphs(self):
        vlb_counter = collections.Counter()
        vevlb_counter = collections.Counter()
        vlb_counted = set()
        vevlb_counted = set()
        for g in self.graphs.values():
            for v in g.vertices.values():
                if (g.gid, v.vlb) not in vlb_counted:
                    vlb_counter[v.vlb] += 1
                vlb_counted.add((g.gid, v.vlb))
                for to, e in v.edges.items():
                    vlb1, vlb2 = v.vlb, g.vertices[to].vlb
                    if self._is_undirected and vlb1 > vlb2:
                        vlb1, vlb2 = vlb2, vlb1
                    if (g.gid, (vlb1, e.elb, vlb2)) not in vevlb_counter:
                        vevlb_counter[(vlb1, e.elb, vlb2)] += 1
                    vevlb_counted.add((g.gid, (vlb1, e.elb, vlb2)))
        # add frequent vertices.
        for vlb, cnt in vlb_counter.items():
            if cnt >= self._min_support:
                g = Graph(gid=next(self._counter),
                          is_undirected=self._is_undirected)
                g.add_vertex(0, vlb)
            else:
                continue
        if self._min_num_vertices > 1:
            self._counter = itertools.count()

    @record_timestamp
    # @profile #Uncomment if memory profiler is desired
    def run(self):
        """Run the closeGraph algorithm."""
        self._read_graphs()
        self._generate_1edge_frequent_subgraphs()
        if self._max_num_vertices < 2:
            return
        root = collections.defaultdict(Projected)
        for gid, g in self.graphs.items():
            for vid, v in g.vertices.items():
                edges = self._get_forward_root_edges(g, vid)
                for e in edges:
                    root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                        PDFS(gid, e, None)
                    )

        vevlbs = list(root.keys())
        vevlbs.sort()
        for vevlb in vevlbs:
            self._DFScode.append(DFSedge(0, 1, vevlb))
            self._frequent_graph_gid_before_root = 0 if len(self._frequent_subgraphs) == 0 else self._frequent_subgraphs[-1].gid
            self._subgraph_mining(root[vevlb])
            self._DFScode.pop()


    def _get_support_graphs(self, projected):
        return len(set([pdfs.gid for pdfs in projected]))

    def _get_where_graphs(self, projected):
        return set([pdfs.gid for pdfs in projected])

    def _get_support_projections(self, projected):
        return len(list([pdfs.gid for pdfs in projected]))

    def _get_where_projections(self, projected):
        return list([pdfs.gid for pdfs in projected])

    def _report_size1(self, g, support):
        g.display()
        line1 = '\nSupport: {}'.format(support)
        line2 = '\n-----------------\n'

        # self._final_report += line1
        # self._final_report += line2

        print(line1)
        print(line2)

    def _report(self, g):
        if g.get_num_vertices() < self._min_num_vertices:
            return
        self._frequent_subgraphs.append(g)
        display_str = g.display()
        support = g.support_graphs
        print('\nSupport: {}'.format(support))

        # Add some report info to pandas dataframe "self._report_df".
        self._report_df = self._report_df.append(
            pd.DataFrame(
                {
                    'support': support,  # [self._support],
                    'description': [display_str],
                    'num_vert': g.get_num_vertices()
                },
                index=[int(repr(self._counter)[6:-1])]
            )
        )
        if self._visualize:
            g.plot()
        if self._where:
            where = g.where_graphs
            print('where: {}'.format(list(where)))
        print('\n-----------------\n')

    def _get_forward_root_edges(self, g, frm):
        result = []
        v_frm = g.vertices[frm]
        for to, e in v_frm.edges.items():
            if (not self._is_undirected) or v_frm.vlb <= g.vertices[to].vlb:
                result.append(e)
        return result

    def _get_backward_edge(self, g, e1, e2, history, dfs_code_root_eid=None):
        if self._is_undirected and e1 == e2:
            return None
        for to, e in g.vertices[e2.to].edges.items():
            if history.has_edge(e.eid) or e.to != e1.frm:
                continue
            # if reture here, then self._DFScodep[0] != dfs_code_min[0]
            # should be checked in _is_min(). or:
            if self._is_undirected:
                if e1.elb < e.elb or (
                        e1.elb == e.elb and
                        g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
                    return e
            else:
                if g.vertices[e1.frm].vlb < g.vertices[e2.to].vlb or (
                        g.vertices[e1.frm].vlb == g.vertices[e2.to].vlb and
                        e1.elb <= e.elb):
                    return e
        return None

    def _get_forward_pure_edges(self, g, rm_edge, min_vlb, history):
        result = []
        for to, e in g.vertices[rm_edge.to].edges.items():
            if (min_vlb <= g.vertices[e.to].vlb) and (
                    not history.has_vertex(e.to)):
                result.append(e)
        return result

    def _get_forward_rmpath_edges(self, g, rm_edge, min_vlb, history, dfs_code_root_eid=None):
        result = []
        to_vlb = g.vertices[rm_edge.to].vlb
        for to, e in g.vertices[rm_edge.frm].edges.items():
            new_to_vlb = g.vertices[to].vlb
            if (rm_edge.to == e.to or
                    (min_vlb > new_to_vlb) or
                    history.has_vertex(e.to)):
                continue
            if rm_edge.elb < e.elb or (
                    rm_edge.elb == e.elb and
                    to_vlb <= new_to_vlb):
                result.append(e)
        return result


    def _has_equivalent_extended_occurrence_projections(self, projected, rmpath, num_vertices, maxtoc):
        backward_root = collections.defaultdict(set)
        current_support = self._get_support_projections(projected)
        for i, p in enumerate(projected):
            g = self.graphs[p.gid]
            history = History(g, p)
            # backward
            for rmpath_i in rmpath[::-1]:
                e1 = history.edges[rmpath_i]
                e2 = history.edges[rmpath[0]]
                e = None
                if not (self._is_undirected and e1 == e2):
                    for to, e3 in g.vertices[e2.to].edges.items():
                        if history.has_edge(e3.eid) or e3.to != e1.frm:
                            continue
                        e = e3
                        break

                if e is not None:
                    backward_root[
                        (self._DFScode[rmpath_i].frm, e.elb)
                    ].add(i)

        for to, elb in backward_root:
            if current_support == len(backward_root[(to, elb)]):
                return True

        forward_root = collections.defaultdict(set)
        for i, p in enumerate(projected):
            g = self.graphs[p.gid]
            history = History(g, p)
            if num_vertices >= self._max_num_vertices:
                break
            edges = []
            for to, e in g.vertices[history.edges[rmpath[0]].to].edges.items():
                if not history.has_vertex(e.to):
                    edges.append(e)
            for e in edges:
                forward_root[
                    (maxtoc, e.elb, g.vertices[e.to].vlb)
                ].add(i)
            # rmpath forward
            for rmpath_i in rmpath:
                rmpath_edges = []
                for to, e in g.vertices[history.edges[rmpath_i].frm].edges.items():
                    if (history.edges[rmpath_i].to == e.to or
                            history.has_vertex(e.to)):
                        continue
                    rmpath_edges.append(e)
                for e in rmpath_edges:
                    forward_root[
                        (self._DFScode[rmpath_i].frm,
                         e.elb, g.vertices[e.to].vlb)
                    ].add(i)

        for frm, elb, vlb2 in forward_root:
            if current_support == len(forward_root[(frm, elb, vlb2)]):
                return True

        return False

    def _has_equivalent_extended_occurrence_graphs(self, projected, path, num_vertices, maxtoc):
        backward_root = collections.defaultdict(set)
        current_support = self._get_support_graphs(projected)
        for p in projected:
            g = self.graphs[p.gid]
            history = History(g, p)
            # backward
            for path_j in path:
                for path_i in path[::-1]:
                    e1 = history.edges[path_i]
                    e2 = history.edges[path_j]
                    e = None
                    if not (self._is_undirected and e1 == e2):
                        for to, e3 in g.vertices[e2.to].edges.items():
                            if history.has_edge(e3.eid) or e3.to != e1.frm:
                                continue
                            e = e3
                            break

                    if e is not None:
                        backward_root[
                            (min(self._DFScode[path_j].to, self._DFScode[path_i].frm),
                             max(self._DFScode[path_j].to, self._DFScode[path_i].frm), e.elb)
                        ].add(g.gid)

                    e = None
                    if not (self._is_undirected and e1 == e2):
                        for to, e3 in g.vertices[e2.to].edges.items():
                            if history.has_edge(e3.eid) or e3.to != e1.to:
                                continue
                            e = e3
                            break

                    if e is not None:
                        backward_root[
                            (min(self._DFScode[path_j].to, self._DFScode[path_i].to),
                             max(self._DFScode[path_j].to, self._DFScode[path_i].to), e.elb)
                        ].add(g.gid)

        for frm, to, elb in backward_root:
            if current_support == len(backward_root[(frm, to, elb)]):
                return True

        forward_root = collections.defaultdict(set)
        for p in projected:
            g = self.graphs[p.gid]
            history = History(g, p)
            if num_vertices >= self._max_num_vertices:
                break

            # path forward
            checked_vertices = set()
            for path_i in path:
                if history.edges[path_i].frm not in checked_vertices:
                    path_edges = []
                    for to, e in g.vertices[history.edges[path_i].frm].edges.items():
                        if (history.edges[path_i].to == e.to or
                                history.has_vertex(e.to)):
                            continue
                        path_edges.append(e)
                    for e in path_edges:
                        forward_root[
                            (self._DFScode[path_i].frm,
                             e.elb, g.vertices[e.to].vlb)
                        ].add(g.gid)
                    checked_vertices.add(history.edges[path_i].frm)

                if history.edges[path_i].to not in checked_vertices:
                    path_edges = []
                    for to, e in g.vertices[history.edges[path_i].to].edges.items():
                        if (history.edges[path_i].frm == e.to or
                                history.has_vertex(e.to)):
                            continue
                        path_edges.append(e)
                    for e in path_edges:
                        forward_root[
                            (self._DFScode[path_i].to,
                             e.elb, g.vertices[e.to].vlb)
                        ].add(g.gid)
                    checked_vertices.add(history.edges[path_i].to)
        for frm, elb, vlb2 in forward_root:
            if current_support == len(forward_root[(frm, elb, vlb2)]):
                return True

        return False

    def _is_min(self):
        if self._verbose:
            print('is_min: checking {}'.format(self._DFScode))
        if len(self._DFScode) == 1:
            return True
        g = self._DFScode.to_graph(gid=VACANT_GRAPH_ID,
                                   is_undirected=self._is_undirected)
        dfs_code_min = DFScode()
        root = collections.defaultdict(Projected)
        for vid, v in g.vertices.items():
            edges = self._get_forward_root_edges(g, vid)
            for e in edges:
                root[(v.vlb, e.elb, g.vertices[e.to].vlb)].append(
                    PDFS(g.gid, e, None))
        min_vevlb = min(root.keys())
        dfs_code_min.append(DFSedge(0, 1, min_vevlb))

        # No need to check if is min code because of pruning in get_*_edge*.

        def project_is_min(projected):
            dfs_code_min.build_rmpath()
            rmpath = dfs_code_min.rmpath
            min_vlb = dfs_code_min[0].vevlb[0]
            maxtoc = dfs_code_min[rmpath[0]].to

            backward_root = collections.defaultdict(Projected)
            flag, newto = False, 0,
            end = 0 if self._is_undirected else -1
            for i in range(len(rmpath) - 1, end, -1):
                if flag:
                    break
                for p in projected:
                    history = History(g, p)
                    e = self._get_backward_edge(g,
                                                history.edges[rmpath[i]],
                                                history.edges[rmpath[0]],
                                                history)
                    if e is not None:
                        backward_root[e.elb].append(PDFS(g.gid, e, p))
                        newto = dfs_code_min[rmpath[i]].frm
                        flag = True
            if flag:
                backward_min_elb = min(backward_root.keys())
                dfs_code_min.append(DFSedge(
                    maxtoc, newto,
                    (VACANT_VERTEX_LABEL,
                     backward_min_elb,
                     VACANT_VERTEX_LABEL)
                ))
                idx = len(dfs_code_min) - 1
                if self._DFScode[idx] != dfs_code_min[idx]:
                    return False
                return project_is_min(backward_root[backward_min_elb])

            forward_root = collections.defaultdict(Projected)
            flag, newfrm = False, 0
            for p in projected:
                history = History(g, p)
                edges = self._get_forward_pure_edges(g,
                                                     history.edges[rmpath[0]],
                                                     min_vlb,
                                                     history)
                if len(edges) > 0:
                    flag = True
                    newfrm = maxtoc
                    for e in edges:
                        forward_root[
                            (e.elb, g.vertices[e.to].vlb)
                        ].append(PDFS(g.gid, e, p))
            for rmpath_i in rmpath:
                if flag:
                    break
                for p in projected:
                    history = History(g, p)
                    edges = self._get_forward_rmpath_edges(g,
                                                           history.edges[
                                                               rmpath_i],
                                                           min_vlb,
                                                           history)
                    if len(edges) > 0:
                        flag = True
                        newfrm = dfs_code_min[rmpath_i].frm
                        for e in edges:
                            forward_root[
                                (e.elb, g.vertices[e.to].vlb)
                            ].append(PDFS(g.gid, e, p))

            if not flag:
                return True

            forward_min_evlb = min(forward_root.keys())
            dfs_code_min.append(DFSedge(
                newfrm, maxtoc + 1,
                (VACANT_VERTEX_LABEL, forward_min_evlb[0], forward_min_evlb[1]))
            )
            idx = len(dfs_code_min) - 1
            if self._DFScode[idx] != dfs_code_min[idx]:
                return False
            return project_is_min(forward_root[forward_min_evlb])

        res = project_is_min(root[min_vevlb])
        return res

    def _subgraph_mining(self, projected):
        self.i = self.i + 1

        self._support = self._get_support_graphs(projected)
        if self._support < self._min_support:
            return

        if not self._is_min():
            return

        self._DFScode.build_rmpath()
        early_termination, early_termination_failure = self._early_termination(projected)
        if early_termination:
            #print(self.i, " early termination#")
            return
        #else:
        #    if early_termination_failure:
        #        print(self.i, " early termination failure")

        num_vertices = self._DFScode.get_num_vertices()
        self._DFScode.build_rmpath()
        rmpath = self._DFScode.rmpath
        maxtoc = self._DFScode[rmpath[0]].to
        min_vlb = self._DFScode[0].vevlb[0]

        forward_root = collections.defaultdict(Projected)
        backward_root = collections.defaultdict(Projected)
        for p in projected:
            g = self.graphs[p.gid]
            history = History(g, p)
            dfs_code_root_eid = None
            # backward
            for rmpath_i in rmpath[::-1]:
                e = self._get_backward_edge(g,
                                            history.edges[rmpath_i],
                                            history.edges[rmpath[0]],
                                            history,
                                            dfs_code_root_eid)
                if e is not None:
                    backward_root[
                        (self._DFScode[rmpath_i].frm, e.elb)
                    ].append(PDFS(g.gid, e, p))
            # pure forward
            if num_vertices >= self._max_num_vertices:
                continue
            edges = self._get_forward_pure_edges(g,
                                                 history.edges[rmpath[0]],
                                                 min_vlb,
                                                 history)
            for e in edges:
                forward_root[
                    (maxtoc, e.elb, g.vertices[e.to].vlb)
                ].append(PDFS(g.gid, e, p))

            # rmpath forward
            for rmpath_i in rmpath:
                edges = self._get_forward_rmpath_edges(g,
                                                       history.edges[rmpath_i],
                                                       min_vlb,
                                                       history,
                                                       dfs_code_root_eid)
                for e in edges:
                    forward_root[
                        (self._DFScode[rmpath_i].frm,
                         e.elb, g.vertices[e.to].vlb)
                    ].append(PDFS(g.gid, e, p))

        last_closed_gid = self._current_frequent_gid

        self._detect_early_termination_failure_cases(maxtoc, forward_root, backward_root, projected, rmpath)

        # Begin searching through forward and backward edges

        # backward
        for to, elb in backward_root:
            self._DFScode.append(DFSedge(
                maxtoc, to,
                (VACANT_VERTEX_LABEL, elb, VACANT_VERTEX_LABEL))
            )
            following_projection = backward_root[(to, elb)]

            self._subgraph_mining(following_projection)

            self._DFScode.pop()

        # forward
        forward_root_sorted = [(frm, elb, vlb2) for frm, elb, vlb2 in forward_root]
        forward_root_sorted.sort(key=lambda x: (-x[0], x[1], x[2]))
        for frm, elb, vlb2 in forward_root_sorted:
            self._DFScode.append(DFSedge(
                frm, maxtoc + 1,
                (VACANT_VERTEX_LABEL, elb, vlb2))
            )
            following_projection = forward_root[(frm, elb, vlb2)]

            self._subgraph_mining(following_projection)

            self._DFScode.pop()

        if early_termination_failure:
            return

        has_equivalent_extended_occurrence = self._has_equivalent_extended_occurrence_projections(projected, rmpath, num_vertices,
                                                                 maxtoc);
        if has_equivalent_extended_occurrence:
            return

        edges_hash_keys = self._create_edges_hash_keys(projected)
        where_graphs = self._get_where_graphs(projected)
        where_projections = self._get_where_projections(projected)
        projected_edges_lists, max_gid = self._collect_projected_edges_all_pdfs(projected)
        self._current_frequent_gid = next(self._counter)
        g = self._DFScode.to_frequent_graph(edges_hash_keys,
                                            where_graphs,
                                            where_projections,
                                            projected_edges_lists,
                                            self._DFScode,
                                            max_gid,
                                            gid=self._current_frequent_gid,
                                            is_undirected=self._is_undirected)

        self._add_closed_graph(g)  # before reporting update the dictionary
        self._report(g)

        return

    def _projected_tail(self, projected, index):
        projected_tails = list()
        for pdfs in projected:
            for i in range(0,index):
               pdfs = pdfs.prev
            already_exist = False
            for pdfs_tail in projected_tails:
                i = pdfs
                j = pdfs_tail
                if i.gid != j.gid:
                    break
                if i.edge.eid != j.edge.eid:
                    break
                i = i.prev
                j = j.prev
                if i is None:
                    already_exist = True
                    break
            if not already_exist:
                projected_tails.append(pdfs)
        return projected_tails

    def _detect_early_termination_failure_cases(self, maxtoc, forward_root, backward_root, projected, rmpath):
        forward_root_etf = collections.defaultdict(set)
        backward_root_etf = collections.defaultdict(set)
        backward_root_etf_rmpath = collections.defaultdict(set)
        backward_root_etf_rmpath_tail = collections.defaultdict(set)
        backward_root_etf_rmpath_gaps = collections.defaultdict(set)

        forward_etf_path = None
        backward_etf_path = None

        histories = dict()

        # type 1
        if self._DFScode[-1].to > self._DFScode[-1].frm:
            for p_i, p in enumerate(projected):
                g = self.graphs[p.gid]
                if (g.gid, p_i) in histories:
                    history = histories[(g.gid, p_i)]
                else:
                    history = History(g, p)
                    histories[(g.gid, p_i)] = history

                g_rmpath_vertices = set()
                for rmpath_i in rmpath:
                    g_rmpath_vertices.add(history.edges[rmpath_i].frm)
                    g_rmpath_vertices.add(history.edges[rmpath_i].to)

                for to, e in g.vertices[history.edges[rmpath[0]].to].edges.items():

                    if history.has_edge(e.eid):
                        continue

                    if e.frm != history.edges[rmpath[0]].to:
                        continue

                    if e.to in g_rmpath_vertices:
                        continue

                    if history.has_vertex(e.to):
                        backward_root_etf[
                            (e.elb, g.vertices[e.to].vlb)
                        ].add(g.gid)

            for elb, vlb in backward_root_etf:
                if (maxtoc, elb, vlb) in forward_root:
                    for pdfs in forward_root[(maxtoc, elb, vlb)]:
                        backward_root_etf[(elb, vlb)].add(pdfs.gid)

                #if len(backward_root_etf[(elb, vlb)]) >= self._min_support:
                if len(set(backward_root_etf[(elb, vlb)])) >= self._min_support:
                    #self.etfDFScodes.append(self._DFScode.copy())
                    self.trie.insert(self._DFScode.copy())
                    return

        # type 2
        if self._DFScode[-1].to > self._DFScode[-1].frm:
            for i,rmpath_i in enumerate(rmpath):
                v_start = self._DFScode[rmpath_i].frm
                v_end = self._DFScode[rmpath_i].to
                if v_end == v_start + 1:
                    continue
                for j in range(i + 1, len(rmpath)):
                    rmpath_j = rmpath[j]
                    v_frm = self._DFScode[rmpath_j].frm

                    for p_i, p in enumerate(projected):
                        g = self.graphs[p.gid]
                        if (g.gid, p_i) in histories:
                            history = histories[(g.gid, p_i)]
                        else:
                            history = History(g, p)
                            histories[(g.gid, p_i)] = history

                        dfs_to_g_vertices = dict()
                        g_to_dfs_vertices = dict()

                        for k, e in enumerate(history.edges):
                            dfs_to_g_vertices[self._DFScode[k].frm] = history.edges[k].frm
                            dfs_to_g_vertices[self._DFScode[k].to] = history.edges[k].to
                            g_to_dfs_vertices[history.edges[k].frm] = self._DFScode[k].frm
                            g_to_dfs_vertices[history.edges[k].to] = self._DFScode[k].to

                        g_rmpath_to_vertices = set()

                        for v in range(v_start + 1, v_end):
                            g_rmpath_to_vertices.add(dfs_to_g_vertices[v])

                        for to, e in g.vertices[dfs_to_g_vertices[v_frm]].edges.items():

                            if history.has_edge(e.eid):
                                continue

                            if e.frm != dfs_to_g_vertices[v_frm]:
                                continue

                            if e.to not in g_rmpath_to_vertices:
                                continue

                            backward_root_etf_rmpath_gaps[
                                (v_frm, e.elb, g.vertices[e.to].vlb)
                            ].add(g.gid)

            for rmpath_frm, elb, vlb in backward_root_etf_rmpath_gaps:
                if (rmpath_frm, elb, vlb) in forward_root:
                    for pdfs in forward_root[(rmpath_frm, elb, vlb)]:
                        backward_root_etf_rmpath_gaps[(rmpath_frm, elb, vlb)].add(pdfs.gid)

                #if len(backward_root_etf_rmpath_gaps[(rmpath_frm, elb, vlb)]) >= self._min_support:
                if len(set(backward_root_etf_rmpath_gaps[(rmpath_frm, elb, vlb)])) >= self._min_support:
                    #self.etfDFScodes.append(self._DFScode.copy())
                    self.trie.insert(self._DFScode.copy())
                    return

        # type 3
        if self._DFScode[-1].to > self._DFScode[-1].frm and len(rmpath) > 2 \
                and self._DFScode[rmpath[-1]].vevlb[1] == self._DFScode[rmpath[-2]].vevlb[1] \
                and self._DFScode[rmpath[-1]].vevlb[2] == self._DFScode[rmpath[-2]].vevlb[2]:
            for p_i, p in enumerate(projected):
                g = self.graphs[p.gid]
                if (g.gid, p_i) in histories:
                    history = histories[(g.gid, p_i)]
                else:
                    history = History(g, p)
                    histories[(g.gid, p_i)] = history

                g_rmpath_first_vertex = history.edges[rmpath[-1]].frm

                for to, e in g.vertices[history.edges[rmpath[0]].to].edges.items():

                    if history.has_edge(e.eid):
                        continue

                    if e.frm != history.edges[rmpath[0]].to:
                        continue

                    if e.to != g_rmpath_first_vertex:
                        continue

                    backward_root_etf_rmpath_tail[
                            (e.elb, g.vertices[e.to].vlb)
                        ].add(g.gid)

            for elb, vlb in backward_root_etf_rmpath_tail:
                if (maxtoc, elb, vlb) in forward_root:
                    for pdfs in forward_root[(maxtoc, elb, vlb)]:
                        backward_root_etf_rmpath_tail[(elb, vlb)].add(pdfs.gid)

                #if len(backward_root_etf_rmpath_tail[(elb, vlb)]) >= self._min_support:
                if len(set(backward_root_etf_rmpath_tail[(elb, vlb)])) >= self._min_support:
                    #self.etfDFScodes.append(self._DFScode.copy())
                    self.trie.insert(self._DFScode.copy())
                    return

        # type 4
        # check if the last edge is backward
        if self._DFScode[-1].to < self._DFScode[-1].frm:
            rmpath_loop = None
            rmpath_loop_index = None
            for i,rmpath_i in enumerate(rmpath):

                if self._DFScode[rmpath_i].frm != self._DFScode[-1].to:
                    continue
                else:
                    rmpath_loop = rmpath_i
                    rmpath_loop_index = i
                    break

            for i,rmpath_i in enumerate(rmpath):
                if rmpath_i < rmpath_loop:
                    break

                elb = self._DFScode[rmpath_i].vevlb[1]
                vlb = self._DFScode[rmpath_i].vevlb[0]
                if vlb == VACANT_VERTEX_LABEL:
                    if i+1 == len(rmpath):
                        for e in self._DFScode:
                            if e.frm == self._DFScode[rmpath_i].frm and e.vevlb[0] != VACANT_VERTEX_LABEL:
                                vlb = e.vevlb[0]
                                break
                    else:
                        vlb = self._DFScode[rmpath[i+1]].vevlb[2]

                for p_i, p in enumerate(projected):
                    g = self.graphs[p.gid]
                    if (g.gid, p_i) in histories:
                        history = histories[(g.gid, p_i)]
                    else:
                        history = History(g, p)
                        histories[(g.gid, p_i)] = history

                    for to, e in g.vertices[history.edges[rmpath_i].to].edges.items():
                        if e.elb != elb:
                            continue

                        if history.has_edge(e.eid):
                            continue

                        if g.vertices[e.to].vlb != vlb:
                            continue

                        if history.has_vertex(e.to):
                            continue

                        #self.etfDFScodes.append(self._DFScode.copy())
                        self.trie.insert(self._DFScode.copy())
                        return

        # type 5
#        if backward_etf_path is None:
        dfs_codes = self._DFScode.build_all_dfs_codes()
        for dfs_code in dfs_codes[1:]:
            if dfs_code[-1].to < dfs_code[-1].frm:
                rmpath_loop = None
                rmpath_loop_index = None
                for i, rmpath_i in enumerate(dfs_code.rmpath):

                    if dfs_code[rmpath_i].frm != dfs_code[-1].to:
                        continue
                    else:
                        rmpath_loop = rmpath_i
                        rmpath_loop_index = i
                        break

                # check symmetry
                rmpath_before_loop_index = rmpath_loop_index + 1
                if (rmpath_before_loop_index < len(dfs_code.rmpath)):
                    rmpath_before_loop_index_vlb = dfs_code[dfs_code.rmpath[rmpath_before_loop_index]].vevlb[0]
                    if rmpath_before_loop_index_vlb == VACANT_VERTEX_LABEL:
                        vid = dfs_code[dfs_code.rmpath[rmpath_before_loop_index]].frm
                        for i in range(0, len(dfs_code)):
                            if dfs_code[i].frm == vid and dfs_code[i].vevlb[0] != VACANT_VERTEX_LABEL:
                                rmpath_before_loop_index_vlb = dfs_code[i].vevlb[0]
                                break
                            if dfs_code[i].to == vid and dfs_code[i].vevlb[2] != VACANT_VERTEX_LABEL:
                                rmpath_before_loop_index_vlb = dfs_code[i].vevlb[2]
                                break

                    rmpath_0_vlb = dfs_code[dfs_code.rmpath[-1]].vevlb[0]
                    if rmpath_before_loop_index_vlb == VACANT_VERTEX_LABEL:
                        vid = dfs_code[dfs_code.rmpath[-1]].frm
                        for i in range(0, len(dfs_code)):
                            if dfs_code[i].frm == vid and dfs_code[i].vevlb[0] != VACANT_VERTEX_LABEL:
                                rmpath_0_vlb = dfs_code[i].vevlb[0]
                                break
                            if dfs_code[i].to == vid and dfs_code[i].vevlb[2] != VACANT_VERTEX_LABEL:
                                rmpath_0_vlb = dfs_code[i].vevlb[2]
                                break

                    if rmpath_0_vlb != dfs_code[dfs_code.rmpath[rmpath_before_loop_index]].vevlb[2] or \
                        dfs_code[dfs_code.rmpath[-1]].vevlb[1] != dfs_code[dfs_code.rmpath[rmpath_before_loop_index]].vevlb[1] or \
                        dfs_code[dfs_code.rmpath[-1]].vevlb[2] != rmpath_before_loop_index_vlb:
                        continue

                dfs_code_projected = self._projected_tail(projected, len(self._DFScode) - len(dfs_code))

                for i, rmpath_i in enumerate(dfs_code.rmpath):
                    if rmpath_i < rmpath_loop:
                        break

                    elb = dfs_code[rmpath_i].vevlb[1]
                    vlb = dfs_code[rmpath_i].vevlb[0]
                    if vlb == VACANT_VERTEX_LABEL:
                        if i + 1 == len(dfs_code.rmpath):
                            for e in dfs_code:
                                if e.frm == dfs_code[rmpath_i].frm and e.vevlb[0] != VACANT_VERTEX_LABEL:
                                    vlb = e.vevlb[0]
                                    break
                        else:
                            vlb = dfs_code[dfs_code.rmpath[i + 1]].vevlb[2]

                    for p in dfs_code_projected:
                        g = self.graphs[p.gid]
                        history = History(g, p)

                        for to, e in g.vertices[history.edges[rmpath_i].to].edges.items():
                            if e.elb != elb:
                                continue

                            if history.has_edge(e.eid):
                                continue

                            if g.vertices[e.to].vlb != vlb:
                                continue

                            if history.has_vertex(e.to):
                                continue

                            #self.etfDFScodes.append(self._DFScode.copy())
                            self.trie.insert(self._DFScode.copy())
                            return

    def _early_termination(self, projected):
        projected_edges, g, edge = self._collect_projected_edges(projected)

        edge_hash_key = frozenset(projected_edges)

        if not edge_hash_key in self.closed_graphs_hash_table:
            return False, False

        early_termination = False

        where_projections = self._get_where_projections(projected)
        where_projections = sorted(where_projections)
        support_projections = len(where_projections)
        projected_edges_lists, max_gid = self._collect_projected_edges_all_pdfs(
            projected)

        reject_early_termination = False

        for g in self.closed_graphs_hash_table[edge_hash_key]:
            has_equivalent_occurrence, isomorphism = g.check_equivalent_occurrence(support_projections,
                                                                                            where_projections,
                                                                                            projected_edges_lists)
            if not has_equivalent_occurrence:
                continue

            reject_early_termination = self._reject_early_termination(g, isomorphism)
            if reject_early_termination:
                break
            early_termination = True
            #print("dfs code ", self._DFScode, " terminated by gid ", g.gid, " early termination failure ", termination_by_dfs)

        if reject_early_termination:
            early_termination = False

        return early_termination, reject_early_termination

    def _reject_early_termination(self, g, isomorphism):
        max_dfs_index = 0
        for index in isomorphism.values():
            if len(g.DFScode) - index > max_dfs_index:
                max_dfs_index = len(g.DFScode) - index
        dfs = g.DFScode[0:max_dfs_index]
        is_in_trie = self.trie.search(dfs)
        return is_in_trie

    def _create_edges_hash_keys(self, projected):
        edges_hash_keys = set()
        if projected is None:
            return edges_hash_keys

        pdfs = list()
        for pdf in projected:
            if pdf is None:
                return edges_hash_keys
            pdfs.append(pdf)

        while True:
            projected_edges, g, edge = self._collect_projected_edges(pdfs)
            edge_hash_key = frozenset(projected_edges)
            edges_hash_keys.add(edge_hash_key)

            pdfs_temp = list()
            for pdf in pdfs:
                if pdf.prev is None:
                    return edges_hash_keys
                pdfs_temp.append(pdf.prev)
            pdfs = pdfs_temp

        return edges_hash_keys



    def _add_closed_graph(self, g):
        for edge_hash_key in g.edges_hash_keys:
            if edge_hash_key not in self.closed_graphs_hash_table:
                self.closed_graphs_hash_table[edge_hash_key] = list()
            self.closed_graphs_hash_table[edge_hash_key].append(g)


    def _collect_projected_edges(self, projected):
        projected_edges = []
        for pdfs in projected:
            g = self.graphs[pdfs.gid]
            edge = pdfs.edge
            enumerated_edge = self.graphs[pdfs.gid].enumerated_edges[edge.eid]
            projected_edges.append(enumerated_edge)
        return projected_edges, g, edge

    def _collect_projected_edges_all_pdfs(self, projected):
        max_gid = -1;
        for pdfs in projected:
            if pdfs.gid > max_gid:
                max_gid = pdfs.gid

        projected_edges_lists = dict()

        for pdfs in projected:
            gid = pdfs.gid
            if not gid in projected_edges_lists:
                projected_edges_lists[gid] = list()

            pdfs_projected = list()
            while pdfs is not None:
                enumerated_edge = self.graphs[pdfs.gid].enumerated_edges[pdfs.edge.eid]
                pdfs_projected.append(enumerated_edge)
                pdfs = pdfs.prev
            projected_edges_lists[gid].append(pdfs_projected)
        return projected_edges_lists, max_gid

