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

import pandas as pd


def record_timestamp(func):
    """Record timestamp before and after call of `func`."""

    def deco(self):
        self.timestamps[func.__name__ + '_in'] = time.time()
        func(self)
        self.timestamps[func.__name__ + '_out'] = time.time()

    return deco


class CloseGraphMode(enum.Enum):
    Normal = 1
    EarlyTerminationFailure = 2


class SupportMode(enum.Enum):
    Projections = 1
    Graphs = 2


class DFSedge(object):
    """DFSedge class."""

    def __init__(self, frm, to, vevlb):
        """Initialize DFSedge instance."""
        self.frm = frm
        self.to = to
        self.vevlb = vevlb

    def __eq__(self, other):
        """Check equivalence of DFSedge."""
        return (self.frm == other.frm and
                self.to == other.to and
                self.vevlb == other.vevlb)

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return '(frm={}, to={}, vevlb={})'.format(
            self.frm, self.to, self.vevlb
        )


class DFScode(list):
    """DFScode is a list of DFSedge."""

    def __init__(self):
        """Initialize DFScode."""
        self.rmpath = list()

    def __eq__(self, other):
        """Check equivalence of DFScode."""
        la, lb = len(self), len(other)
        if la != lb:
            return False
        for i in range(la):
            if self[i] != other[i]:
                return False
        return True

    def __ne__(self, other):
        """Check if not equal."""
        return not self.__eq__(other)

    def __repr__(self):
        """Represent DFScode in string way."""
        return ''.join(['[', ','.join(
            [str(dfsedge) for dfsedge in self]), ']']
                       )

    def push_back(self, frm, to, vevlb):
        """Update DFScode by adding one edge."""
        self.append(DFSedge(frm, to, vevlb))
        return self

    def to_graph(self, gid=VACANT_GRAPH_ID, is_undirected=True):
        """Construct a graph according to the dfs code."""
        g = Graph(gid,
                  is_undirected=is_undirected,
                  eid_auto_increment=True)
        for dfsedge in self:
            frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
            if vlb1 != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, vlb1)
            if vlb2 != VACANT_VERTEX_LABEL:
                g.add_vertex(to, vlb2)
            g.add_edge(AUTO_EDGE_ID, frm, to, elb)
        return g

    def to_frequent_graph(self, graph_edges_projection_sets, where_graphs, where_projections,
                          projected_edges_sets, projected_edges_lists,  example_gid, gid=VACANT_GRAPH_ID, is_undirected=True):
        """Construct a graph according to the dfs code."""
        g = FrequentGraph(graph_edges_projection_sets,
                          where_graphs,
                          where_projections,
                          projected_edges_sets,
                          projected_edges_lists,
                          example_gid,
                          gid,
                          is_undirected=is_undirected,
                          eid_auto_increment=True)
        for dfsedge in self:
            frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
            if vlb1 != VACANT_VERTEX_LABEL:
                g.add_vertex(frm, vlb1)
            if vlb2 != VACANT_VERTEX_LABEL:
                g.add_vertex(to, vlb2)
            g.add_edge(AUTO_EDGE_ID, frm, to, elb)
        return g

    def from_graph(self, g):
        """Build DFScode from graph `g`."""
        raise NotImplementedError('Not inplemented yet.')

    def build_rmpath(self):
        """Build right most path."""
        self.rmpath = list()
        old_frm = None
        for i in range(len(self) - 1, -1, -1):
            dfsedge = self[i]
            frm, to = dfsedge.frm, dfsedge.to
            if frm < to and (old_frm is None or to == old_frm):
                self.rmpath.append(i)
                old_frm = frm
        return self

    def get_num_vertices(self):
        """Return number of vertices in the corresponding graph."""
        return len(set(
            [dfsedge.frm for dfsedge in self] +
            [dfsedge.to for dfsedge in self]
        ))

    def set_root_minimal_labels(self):
        if len(self) == 0:
            return
        dfsedge = self[0];
        frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
        dfsedge.vevlb = ('0', '0', vlb2)
        return vlb1, elb

    def restore_root_original_labels(self, vlb1, elb):
        if len(self) == 0:
            return
        dfsedge = self[0];
        frm, to, (min_vlb1, min_elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
        dfsedge.vevlb = (vlb1, elb, vlb2)
        return


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
                 where=False,
                 support_mode=SupportMode.Projections):
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
        self._frequent_size1_subgraphs = list()
        # Include subgraphs with
        # any num(but >= 2, <= max_num_vertices) of vertices.
        self._frequent_subgraphs = list()
        self._counter = itertools.count()
        self._verbose = verbose
        self._visualize = visualize
        self._where = where
        self._support_mode = support_mode
        self._mode = CloseGraphMode.Normal
        self.timestamps = dict()
        if self._max_num_vertices < self._min_num_vertices:
            print('Max number of vertices can not be smaller than '
                  'min number of that.\n'
                  'Set max_num_vertices = min_num_vertices.')
            self._max_num_vertices = self._min_num_vertices
        self._report_df = pd.DataFrame()
        self.edge_projection_sets = dict()
        self.edge_projection_sets_closed_graphs = dict()
        self._report_df_cumulative = pd.DataFrame()

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
                    # prefix each vertex label with '0'
                    tgraph.add_vertex(cols[1], '0' + cols[2])
                elif cols[0] == 'e':
                    # prefix each edge label with '0'
                    tgraph.add_edge(AUTO_EDGE_ID, cols[1], cols[2], '0' + cols[3])
            # adapt to input files that do not end with 't # -1'
            if tgraph is not None:
                self.graphs[graph_cnt] = tgraph
        return self

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
                self._frequent_size1_subgraphs.append(g)
                # if self._min_num_vertices <= 1:
                # self._report_size1(g, support=cnt)
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
            self._subgraph_mining(root[vevlb])
            self._DFScode.pop()

        # self._remove_false_closed_graphs_with_internal_isomorphism()

        self._mode = CloseGraphMode.EarlyTerminationFailure

        #for vevlb in vevlbs:
        #    self._DFScode.append(DFSedge(0, 1, vevlb))
        #    self._subgraph_mining(root[vevlb])
        #    self._DFScode.pop()

        # self._remove_false_closed_graphs_with_internal_isomorphism()

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
        # print('\nSupport: {}'.format(self._support))
        support = g.support_projections if self._support_mode == SupportMode.Projections else g.support_graphs

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
            where = g.where_projections if self._support_mode == SupportMode.Projections else g.where_graphs
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
                        g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb) or (
                        self._mode == CloseGraphMode.EarlyTerminationFailure
                        and dfs_code_root_eid is not None
                        and e1.eid == dfs_code_root_eid):
                    return e
            else:
                if g.vertices[e1.frm].vlb < g.vertices[e2.to].vlb or (
                        g.vertices[e1.frm].vlb == g.vertices[e2.to].vlb and
                        e1.elb <= e.elb):
                    return e
            # if e1.elb < e.elb or (e1.elb == e.elb and
            #     g.vertices[e1.to].vlb <= g.vertices[e2.to].vlb):
            #     return e
        return None

    def _get_forward_pure_edges(self, g, rm_edge, min_vlb, history):
        result = []
        for to, e in g.vertices[rm_edge.to].edges.items():
            if (min_vlb <= g.vertices[e.to].vlb or self._mode == CloseGraphMode.EarlyTerminationFailure) and (
                    not history.has_vertex(e.to)):
                result.append(e)
        return result

    def _get_forward_rmpath_edges(self, g, rm_edge, min_vlb, history, dfs_code_root_eid=None):
        result = []
        to_vlb = g.vertices[rm_edge.to].vlb
        for to, e in g.vertices[rm_edge.frm].edges.items():
            new_to_vlb = g.vertices[to].vlb
            if (rm_edge.to == e.to or
                    (min_vlb > new_to_vlb and self._mode != CloseGraphMode.EarlyTerminationFailure) or
                    history.has_vertex(e.to)):
                continue
            if rm_edge.elb < e.elb or (
                    self._mode == CloseGraphMode.EarlyTerminationFailure
                    and dfs_code_root_eid is not None
                    and rm_edge.eid == dfs_code_root_eid) or (
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
            '''
            edges = []
            for to, e in g.vertices[history.edges[path[0]].to].edges.items():
                if not history.has_vertex(e.to):
                    edges.append(e)
            for e in edges:
                forward_root[
                    (maxtoc, e.elb, g.vertices[e.to].vlb)
                ].add(g.gid)
            '''
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
        # self._support = self._get_support_projections(projected) if self._support_mode == SupportMode.Projections else self._get_support_graphs(projected)
        self._support = self._get_support_graphs(projected)
        if self._support < self._min_support:
            return

        vlb1 = None
        elb = None
        if self._mode == CloseGraphMode.EarlyTerminationFailure:
            vlb1, elb = self._DFScode.set_root_minimal_labels()

        if not self._is_min():
            if self._mode == CloseGraphMode.EarlyTerminationFailure:
                self._DFScode.restore_root_original_labels(vlb1, elb)
            return

        if self._mode == CloseGraphMode.EarlyTerminationFailure:
            self._DFScode.restore_root_original_labels(vlb1, elb)

        if self._terminate_early(projected):
            return

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
            if self._mode == CloseGraphMode.EarlyTerminationFailure:
                dfs_code_root_eid = p._root_eid()
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
        # Begin searching through forward and backward edges

        # output = True

        # backward
        for to, elb in backward_root:
            self._DFScode.append(DFSedge(
                maxtoc, to,
                (VACANT_VERTEX_LABEL, elb, VACANT_VERTEX_LABEL))
            )
            following_projection = backward_root[(to, elb)]

            self._subgraph_mining(following_projection)
            # if self._should_output(projected,following_projection) is False:
            #    output = False

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
            # if self._should_output(projected,following_projection) is False:
            #    output = False

            self._DFScode.pop()

        has_equivalent_extended_occurrence = \
            self._has_equivalent_extended_occurrence_projections(projected, rmpath, num_vertices,
                                                                 maxtoc) if self._support_mode == SupportMode.Projections \
                else self._has_equivalent_extended_occurrence_graphs(projected, list(reversed(range(rmpath[0] + 1))),
                                                                     num_vertices, maxtoc)

        # if output != (not has_equivalent_extended_occurrence):
        #    print("output and has_equivalent_extended_occurrence disagree")

        if not has_equivalent_extended_occurrence:
            # if output:
            graph_edges_projection_set = self._projected_to_edges_projection_set(projected)
            where_graphs = self._get_where_graphs(projected)
            where_projections = self._get_where_projections(projected)
            projected_edges_sets, projected_edges_lists, max_gid = self._collect_projected_edges_all_pdfs(projected)
            g = self._DFScode.to_frequent_graph(graph_edges_projection_set,
                                                where_graphs,
                                                where_projections,
                                                projected_edges_sets,
                                                projected_edges_lists,
                                                max_gid,
                                                gid=next(self._counter),
                                                is_undirected=self._is_undirected)
            # g._dfscode = copy.copy(self._DFScode) # delete later
            self._add_closed_graph_to_projection_set(g)  # before reporting update the dictionary
            self._report(g)
        return self

    def _terminate_early_termination_failure(self, projected):
        projected_edges, g, edge = self._collect_projected_edges(projected)
        frmlbl, edgelbl, tolbl = g._get_DFSLabels(edge)
        dfs_labels = DFSlabel(frmlbl=frmlbl, edgelbl=edgelbl, tolbl=tolbl)

        root_projected_edges, g, root_edge = self._collect_root_projected_edges(projected)
        root_frmlbl, root_edgelbl, root_tolbl = g._get_DFSLabels(root_edge)
        root_dfs_labels = DFSlabel(frmlbl=root_frmlbl, edgelbl=root_edgelbl, tolbl=root_tolbl)

        set_of_proj_edges = frozenset(projected_edges)
        root_set_of_proj_edges = frozenset(root_projected_edges)

        for closed_graph in self._frequent_subgraphs:
            if not closed_graph.has_edge_projection_set(dfs_labels, set_of_proj_edges):
                continue
            if not closed_graph.has_edge_projection_set(root_dfs_labels, root_set_of_proj_edges):
                continue
            return True

        return False

    def _terminate_early(self, projected):
        """
        Checks if the subgraph mining should end early.
        """

        projected_edges, g, edge = self._collect_projected_edges(projected)
        frmlbl, edgelbl, tolbl = g._get_DFSLabels(edge)
        dfs_labels = DFSlabel(frmlbl=frmlbl, edgelbl=edgelbl, tolbl=tolbl)

        # Check if set of projected edges already exists in the values of the specified key
        # return true if yes, false otherwise
        set_of_proj_edges = frozenset(projected_edges)
        if dfs_labels not in self.edge_projection_sets:
            return False
        else:
            set_of_sets = self.edge_projection_sets[dfs_labels]
            if set_of_proj_edges in set_of_sets:
                if self._mode == CloseGraphMode.EarlyTerminationFailure:
                    return self._terminate_early_termination_failure(projected)
                else:
                    print("early termination")
                    where_projections = self._get_where_projections(projected)
                    where_projections = sorted(where_projections)
                    support_projections = len(where_projections)
                    projected_edges_sets, projected_edges_lists, max_gid = self._collect_projected_edges_all_pdfs(
                        projected)
                    for g in self.edge_projection_sets_closed_graphs[set_of_proj_edges]:
                        has_equivalent_occurrence = g.check_equivalent_occurrence(support_projections, where_projections, projected_edges_lists)
                        if not has_equivalent_occurrence:
                            print("equivalent occurrence not verified gid ",g.gid)
                        else:
                            print("equivalent occurrence verified")
                            return True
                    return False  # Early Termination case
            else:
                return False

    def _projected_to_edges_projection_set(self, projected):
        edges_projection_set = dict()
        if projected is None:
            return edges_projection_set

        pdfs = list()
        for pdf in projected:
            if pdf is None:
                return edges_projection_set
            pdfs.append(pdf)

        while True:
            projected_edges, g, edge = self._collect_projected_edges(pdfs)
            frmlbl, edgelbl, tolbl = g._get_DFSLabels(edge)
            dfs_labels = DFSlabel(frmlbl=frmlbl, edgelbl=edgelbl, tolbl=tolbl)

            # Update DFSlabels dictionary to reference new projected edge
            set_of_proj_edges = frozenset(projected_edges)
            if dfs_labels not in edges_projection_set:
                set_of_sets = set()
                set_of_sets.add(set_of_proj_edges)
                # Structure of DFSlabels_dict is DFSlabels -> set(frozenset(ProjectedEdges))
                edges_projection_set[dfs_labels] = set_of_sets
            else:
                set_of_sets = edges_projection_set[dfs_labels]
                if set_of_proj_edges not in set_of_sets:
                    set_of_sets.add(set_of_proj_edges)  # Add the new set to the dict

            pdfs_temp = list()
            for pdf in pdfs:
                if pdf.prev is None:
                    return edges_projection_set
                pdfs_temp.append(pdf.prev)
            pdfs = pdfs_temp

        return edges_projection_set

    def _add_closed_graph_to_projection_set(self, g):
        for dfs_labels in g.edges_projection_sets:
            if dfs_labels not in self.edge_projection_sets:
                self.edge_projection_sets[dfs_labels] = g.edges_projection_sets[dfs_labels]
            else:
                self.edge_projection_sets[dfs_labels].update(g.edges_projection_sets[dfs_labels])
            for edges_projection_set in g.edges_projection_sets[dfs_labels]:
                if edges_projection_set not in self.edge_projection_sets_closed_graphs:
                    self.edge_projection_sets_closed_graphs[edges_projection_set] = list()
                self.edge_projection_sets_closed_graphs[edges_projection_set].append(g)



    def _should_output(self, current_projection, following_projection):
        following_projection_support = self._get_support_graphs(following_projection)
        if self._get_support_graphs(current_projection) == following_projection_support:
            return False  # if the supports are the same, then do NOT output

        return True

    def _collect_projected_edges(self, projected):
        projected_edges = []
        for pdfs in projected:
            g = self.graphs[pdfs.gid]
            edge = pdfs.edge
            # new_proj_edge = ProjectedEdge(originalGraphId=pdfs.gid, edgeId=edge.eid)
            enumerated_edge = self.graphs[pdfs.gid].enumerated_edges[edge.eid]
            projected_edges.append(enumerated_edge)
        return projected_edges, g, edge

    def _collect_root_projected_edges(self, projected):
        projected_edges = []
        for pdfs in projected:
            while pdfs.prev is not None:
                pdfs = pdfs.prev
            g = self.graphs[pdfs.gid]
            edge = pdfs.edge
            # new_proj_edge = ProjectedEdge(originalGraphId=pdfs.gid, edgeId=edge.eid)
            enumerated_edge = self.graphs[pdfs.gid].enumerated_edges[edge.eid]
            projected_edges.append(enumerated_edge)
        return projected_edges, g, edge

    def _collect_projected_edges_all_pdfs(self, projected):
        max_gid = -1;
        for pdfs in projected:
            if pdfs.gid > max_gid:
                max_gid = pdfs.gid

        projected_edges_sets = dict()
        projected_edges_lists = dict()

        for pdfs in projected:
            gid = pdfs.gid
            if not gid in projected_edges_sets:
                projected_edges_sets[gid] = set()
                projected_edges_lists[gid] = list()

            pdfs_projected = list()
            while pdfs is not None:
                enumerated_edge = self.graphs[pdfs.gid].enumerated_edges[pdfs.edge.eid]
                pdfs_projected.append(enumerated_edge)
                pdfs = pdfs.prev
            projected_edges_lists[gid].append(pdfs_projected)
            set_of_proj_edges = frozenset(pdfs_projected)
            projected_edges_sets[gid].add(set_of_proj_edges)
        return projected_edges_sets, projected_edges_lists, max_gid

    def _remove_false_closed_graphs_with_internal_isomorphism(self):
        closed_graphs = list()
        for g in self._frequent_subgraphs:
            is_false_closed_graph = False
            for g_tag in self._frequent_subgraphs:
                if g.gid == g_tag.gid:
                    continue;
                is_supergraph = g_tag.is_supergraph_of_with_support_projections(
                    g) if self._support_mode == SupportMode.Projections else g_tag.is_supergraph_of_with_support_graphs(
                    g)
                if is_supergraph:
                    is_false_closed_graph = True
                    print("false closed graph found, gid ", g.gid, " g_tag gid ", g_tag.gid, " g where ",
                          g.where_projections, " g_tag where ", g_tag.where_projections)
                    # check
                    g_tag_edges = set()
                    g_edges = set()
                    for pdfs in g_tag.edges_projection_sets.keys():
                        for edges_set in g_tag.edges_projection_sets[pdfs]:
                            g_tag_edges.update(edges_set)
                    for pdfs in g.edges_projection_sets.keys():
                        for edges_set in g.edges_projection_sets[pdfs]:
                            g_edges.update(edges_set)
                    only_in_g = g_edges.difference(g_tag_edges)
                    print("difference " + str(len(only_in_g)))
                    # g_tag.plot()
                    # g.plot()
                    if (len(only_in_g) == 0):
                        print("fuck")
                        # g_tag.plot()
                        # g.plot()
                    break
            if not is_false_closed_graph:
                closed_graphs.append(g)
        print("final number of closed graphs " + str(len(closed_graphs)))
