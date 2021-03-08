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
from early_termination_failure import EarlyTerminationFailureTree

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

    def __init__(self, orig=None, last_index=None):
        """Initialize DFScode."""
        if orig is None:
            self.rmpath = list()
        else:
            for i in range(0, last_index):
                self.append(orig[i])
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
        g.dfs_code_edges_directions = self.edges_directions()
        return g

    def to_frequent_graph(self, graph_edges_projection_sets, where_graphs, where_projections,
                          projected_edges_sets, projected_edges_lists,  DFScode, example_gid, gid=VACANT_GRAPH_ID, is_undirected=True):
        """Construct a graph according to the dfs code."""
        g = FrequentGraph(graph_edges_projection_sets,
                          where_graphs,
                          where_projections,
                          projected_edges_sets,
                          projected_edges_lists,
                          DFScode.copy(),
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
        g.dfs_code_edges_directions = self.edges_directions()
        return g

    def edges_directions(self):
        directions = list()
        for dfsedge in reversed(self):
            frm, to, (vlb1, elb, vlb2) = dfsedge.frm, dfsedge.to, dfsedge.vevlb
            if frm < to:
                directions.append(EdgeDirection.Forward)
            else:
                directions.append(EdgeDirection.Backward)
        return directions

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

    def build_all_dfs_codes(self):
        """Build right most path."""
        dfs_codes = list()
        dfs_codes.append(self)
        old_frm = None
        for i in range(len(self) - 1, -1, -1):
            dfsedge = self[i]
            frm, to = dfsedge.frm, dfsedge.to
            if frm < to and (old_frm is None or to == old_frm):
                old_frm = frm
                continue
            else:
                if frm > to and (old_frm is None or frm == old_frm):
                    continue
                else:
                    old_frm = frm
                    dfs_code = DFScode(self, i + 1)
                    dfs_codes.append(dfs_code)

        for dfs_code in dfs_codes:
            dfs_code.build_rmpath()

        return dfs_codes

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

    def vertex_labels(self):
        vlbs = dict()
        for e in self:
            if e.vevlb[0] != VACANT_VERTEX_LABEL:
                vlbs[e.frm] = e.vevlb[0]
            if e.vevlb[2] != VACANT_VERTEX_LABEL:
                vlbs[e.to] = e.vevlb[2]
        return vlbs


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
        self._early_termination_failure_causing_graphs = dict()
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
        self._current_early_termination_failure_causing_graphs = set()
        self.i = 0
        self._current_frequent_gid = -1

        self._early_termination_failure_tree = EarlyTerminationFailureTree()
        self.etfDFScodes = list()

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
            self._frequent_graph_gid_before_root = 0 if len(self._frequent_subgraphs) == 0 else self._frequent_subgraphs[-1].gid
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

    def _can_create_early_termination_failure(self, projected, path):
        forward_root = collections.defaultdict(set)
        backward_root = collections.defaultdict(set)
        for i, p in enumerate(projected):
            g = self.graphs[p.gid]
            history = History(g, p)
            checked_vertices = set()

            for path_i in path:
                path_forward_edges = []
                path_backward_edges = []

                if self._DFScode[path_i].frm in checked_vertices:
                    continue
                checked_vertices.add(self._DFScode[path_i].frm)
                for to, e in g.vertices[history.edges[path_i].frm].edges.items():

                    if history.has_edge(e.eid):
                        continue
                    if e.frm != history.edges[path_i].frm:
                        continue
                    if history.has_vertex(e.to):
                        path_backward_edges.append(e)
                    else:
                        path_forward_edges.append(e)

                for e in path_forward_edges:
                    forward_root[
                        (self._DFScode[path_i].frm,
                         e.elb, g.vertices[e.to].vlb)
                    ].add(p.gid)

                for e in path_backward_edges:
                    backward_root[
                        (self._DFScode[path_i].frm,
                         e.elb, g.vertices[e.to].vlb)
                    ].add(p.gid)


            for path_i in path:
                path_forward_edges = []
                path_backward_edges = []

                if self._DFScode[path_i].to in checked_vertices:
                    continue
                checked_vertices.add(self._DFScode[path_i].to)

                for to, e in g.vertices[history.edges[path_i].to].edges.items():

                    if history.has_edge(e.eid):
                        continue
                    if e.frm != history.edges[path_i].to:
                        continue
                    if history.has_vertex(e.to):
                        path_backward_edges.append(e)
                    else:
                        path_forward_edges.append(e)

                for e in path_forward_edges:
                    forward_root[
                        (self._DFScode[path_i].to,
                         e.elb, g.vertices[e.to].vlb)
                    ].add(p.gid)

                for e in path_backward_edges:
                    backward_root[
                        (self._DFScode[path_i].to,
                         e.elb, g.vertices[e.to].vlb)
                    ].add(p.gid)
        
        for v, elb, vlb in backward_root:
            if (v, elb, vlb) in forward_root:
                if len(forward_root[(v, elb, vlb)]) >= self._min_support:
                    continue
                gids = backward_root[(v, elb, vlb)].union(forward_root[(v, elb, vlb)])
                if len(gids) >= self._min_support:
                    return True
            else:
                if len(backward_root[(v, elb, vlb)]) >= self._min_support:
                    return True
        return False

    def _is_early_termination_failure_causing_graph_subgraph(self, g, current_frequent_gid):
        for gid in self._early_termination_failure_causing_graphs.keys():
            if gid > current_frequent_gid:
                continue
            has_equivalent_occurrence, preserves_directions = self._early_termination_failure_causing_graphs[gid].is_supergraph_of_with_support_projections(g)
            if has_equivalent_occurrence:
                return True
        return False

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
        self.i = self.i + 1
        current_frequent_gid = self._current_frequent_gid
        # self._support = self._get_support_projections(projected) if self._support_mode == SupportMode.Projections else self._get_support_graphs(projected)
        self._support = self._get_support_graphs(projected)
        if self._support < self._min_support:
            return

        if not self._is_min():
            return

        self._DFScode.build_rmpath()
        new_early_termination, new_eraly_termination_failure = self._terminate_early3(projected)
        if new_early_termination:
            print(self.i, " yes new early termination")
            return
        else:
            print(self.i, " no new early termination")

        this_added_early_termination_failure_causing_graphs = set()
        early_termination, this_early_termination_failure_causing_graphs =  self._terminate_early(projected)
        for gid in this_early_termination_failure_causing_graphs:
            if gid not in self._current_early_termination_failure_causing_graphs:
                self._current_early_termination_failure_causing_graphs.add(gid)
                this_added_early_termination_failure_causing_graphs.add(gid)
        if early_termination and len(self._current_early_termination_failure_causing_graphs) == 0:
            print(self.i," early termination")
            # # # return
        else:
            print(self.i," no early termination")

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

        #self._terminate_early1(maxtoc, forward_root, backward_root)
        last_closed_gid = self._current_frequent_gid
        forward_etf_path, backward_etf_path = self._terminate_early2(maxtoc, forward_root, backward_root, projected, rmpath)

        # Begin searching through forward and backward edges

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

        for closed_gid in range(last_closed_gid + 1, self._current_frequent_gid + 1):
            if forward_etf_path is not None:
                self._early_termination_failure_tree.add_path_close_graphs(forward_etf_path, closed_gid)
            if backward_etf_path is not None:
                self._early_termination_failure_tree.add_path_close_graphs(backward_etf_path, closed_gid)
                #for i in range(1, len(backward_etf_path)):
                #    self._early_termination_failure_tree.add_path_close_graphs(backward_etf_path[0:i], closed_gid)

        # # #if len(this_early_termination_failure_causing_graphs) > 0:
        # # #    self._current_early_termination_failure_causing_graphs = self._current_early_termination_failure_causing_graphs - this_added_early_termination_failure_causing_graphs
        # # #    return

        if new_eraly_termination_failure:
            return

        has_equivalent_extended_occurrence = \
            self._has_equivalent_extended_occurrence_projections(projected, rmpath, num_vertices,
                                                                 maxtoc) if self._support_mode == SupportMode.Projections \
                else self._has_equivalent_extended_occurrence_graphs(projected, list(reversed(range(rmpath[0] + 1))),
                                                                     num_vertices, maxtoc)

        # if output != (not has_equivalent_extended_occurrence):
        #    print("output and has_equivalent_extended_occurrence disagree")

        if has_equivalent_extended_occurrence:
            return

        graph_edges_projection_set = self._projected_to_edges_projection_set(projected)
        where_graphs = self._get_where_graphs(projected)
        where_projections = self._get_where_projections(projected)
        projected_edges_sets, projected_edges_lists, max_gid = self._collect_projected_edges_all_pdfs(projected)
        self._current_frequent_gid = next(self._counter)
        g = self._DFScode.to_frequent_graph(graph_edges_projection_set,
                                            where_graphs,
                                            where_projections,
                                            projected_edges_sets,
                                            projected_edges_lists,
                                            self._DFScode,
                                            max_gid,
                                            gid=self._current_frequent_gid,
                                            is_undirected=self._is_undirected)

        etf = self._can_create_early_termination_failure(projected, list(range(0, len(self._DFScode))))
        if etf:
            self._early_termination_failure_causing_graphs[g.gid] = g
            print("graph ", str(g.gid), " can create etf")


        if forward_etf_path is not None:
            self._early_termination_failure_tree.add_path_close_graphs(forward_etf_path, self._current_frequent_gid)
        if backward_etf_path is not None:
            self._early_termination_failure_tree.add_path_close_graphs(backward_etf_path, self._current_frequent_gid)


        self._add_closed_graph_to_projection_set(g)  # before reporting update the dictionary
        self._report(g)

        return

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

    def _terminate_early2(self, maxtoc, forward_root, backward_root,projected, rmpath):
        forward_root_etf = collections.defaultdict(set)
        backward_root_etf = collections.defaultdict(set)
        backward_root_etf_rmpath = collections.defaultdict(set)
        backward_root_etf_rmpath_tail = collections.defaultdict(set)
        backward_root_etf_rmpath_gaps = collections.defaultdict(set)

        forward_etf_path = None
        backward_etf_path = None

        # type 1
        if self._DFScode[-1].to > self._DFScode[-1].frm:
            for p in projected:
                g = self.graphs[p.gid]
                history = History(g, p)

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

                if len(backward_root_etf[(elb, vlb)]) >= self._min_support:
                    path = self._early_termination_failure_tree.add_nodes_rmpath(self._DFScode, rmpath, list())
                    path_index = 0
                    for i, rmpath_i in enumerate(reversed(rmpath)):
                        frm = self._DFScode[rmpath_i].frm
                        count = 0
                        for e in self._DFScode:
                            if e.frm == frm and e.to > e.frm:
                                count += 1
                        if count == 1:
                            path_index += 1
                        else:
                            break
                    forward_etf_path = path[0: path_index + 1]
                    break

        # type 3
        if self._DFScode[-1].to > self._DFScode[-1].frm:
            for i,rmpath_i in enumerate(rmpath):
                v_start = self._DFScode[rmpath_i].frm
                v_end = self._DFScode[rmpath_i].to
                if v_end == v_start + 1:
                    continue
                for j in range(i + 1, len(rmpath)):
                    rmpath_j = rmpath[j]
                    v_frm = self._DFScode[rmpath_j].frm

                    for p in projected:
                        g = self.graphs[p.gid]
                        history = History(g, p)

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

                if len(backward_root_etf_rmpath_gaps[(rmpath_frm, elb, vlb)]) >= self._min_support:
                    path = self._early_termination_failure_tree.add_nodes_rmpath(self._DFScode, rmpath, list())
                    path_index = 0
                    for i, rmpath_i in enumerate(reversed(rmpath)):
                        frm = self._DFScode[rmpath_i].frm
                        count = 0
                        for e in self._DFScode:
                            if e.frm == frm and e.to > e.frm:
                                count += 1
                        if count == 1:
                            path_index += 1
                        else:
                            break
                    forward_etf_path = path[0: path_index + 1]
                    break

        # type 4
        if self._DFScode[-1].to > self._DFScode[-1].frm and len(rmpath) > 2 \
                and self._DFScode[rmpath[-1]].vevlb[1] == self._DFScode[rmpath[-2]].vevlb[1] \
                and self._DFScode[rmpath[-1]].vevlb[2] == self._DFScode[rmpath[-2]].vevlb[2]:
            for p in projected:
                g = self.graphs[p.gid]
                history = History(g, p)

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

                if len(backward_root_etf_rmpath_tail[(elb, vlb)]) >= self._min_support:
                    path = self._early_termination_failure_tree.add_nodes_rmpath(self._DFScode, rmpath, list())
                    path_index = 0
                    for i, rmpath_i in enumerate(reversed(rmpath)):
                        frm = self._DFScode[rmpath_i].frm
                        count = 0
                        for e in self._DFScode:
                            if e.frm == frm and e.to > e.frm:
                                count += 1
                        if count == 1:
                            path_index += 1
                        else:
                            break
                    forward_etf_path = path[0: path_index + 1]
                    break



        # type 5
        # check if the last edge is backward
        if self._DFScode[-1].to < self._DFScode[-1].frm:
            rmpath_loop = None
            rmpath_loop_index = None
            for i,rmpath_i in enumerate(rmpath):
                # if self._DFScode[rmpath_i].to != self._DFScode[-1].to:
                if self._DFScode[rmpath_i].frm != self._DFScode[-1].to:
                    continue
                else:
                    rmpath_loop = rmpath_i
                    rmpath_loop_index = i
                    break

            #if rmpath_loop_index is None:
            #    if self._DFScode[rmpath[-1]].frm == self._DFScode[-1].to:
            #        rmpath_loop_index = -1

            for i,rmpath_i in enumerate(rmpath):
                #if rmpath_i == rmpath_loop:
                #    break
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

                for p in projected:
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

                        reversed_rmpath = list()
                        reversed_rmpath.append(len(self._DFScode) - 1)
                        reversed_rmpath.extend(rmpath[0:rmpath_loop_index + 1])
                        path = self._early_termination_failure_tree.add_nodes_rmpath(self._DFScode, rmpath[rmpath_loop_index + 1:] if rmpath_loop_index + 1 < len(rmpath) else list(), reversed_rmpath)
                        last_index = len(rmpath) - rmpath_loop_index if rmpath_loop_index < len(rmpath) else 1
                        backward_etf_path = path[0:last_index]
                        break

                    if backward_etf_path is not None:
                        break
                if backward_etf_path is not None:
                    break

        # type 6
        if backward_etf_path is None:
            dfs_codes = self._DFScode.build_all_dfs_codes()
            for dfs_code in dfs_codes[1:]:
                if dfs_code[-1].to < dfs_code[-1].frm:
                    rmpath_loop = None
                    rmpath_loop_index = None
                    for i, rmpath_i in enumerate(dfs_code.rmpath):
                        # if self._DFScode[rmpath_i].to != self._DFScode[-1].to:
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
                    # if rmpath_loop_index is None:
                    #    if self._DFScode[rmpath[-1]].frm == self._DFScode[-1].to:
                    #        rmpath_loop_index = -1

                    dfs_code_projected = self._projected_tail(projected, len(self._DFScode) - len(dfs_code))

                    for i, rmpath_i in enumerate(dfs_code.rmpath):
                        # if rmpath_i == rmpath_loop:
                        #    break
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

                                reversed_rmpath = list()
                                reversed_rmpath.append(len(dfs_code) - 1)
                                reversed_rmpath.extend(dfs_code.rmpath[0:rmpath_loop_index + 1])
                                path = self._early_termination_failure_tree.add_nodes_rmpath(dfs_code, dfs_code.rmpath[
                                                                                                            rmpath_loop_index + 1:] if rmpath_loop_index + 1 < len(
                                    dfs_code.rmpath) else list(), reversed_rmpath)
                                last_index = len(dfs_code.rmpath) - rmpath_loop_index if rmpath_loop_index < len(dfs_code.rmpath) else 1
                                backward_etf_path = path[0:last_index]
                                break

                            if backward_etf_path is not None:
                                break
                        if backward_etf_path is not None:
                            break
                    if backward_etf_path is not None:
                        break

# end types

        if forward_etf_path is not None or backward_etf_path is not None:
            self.etfDFScodes.append(self._DFScode.copy())

        return forward_etf_path, backward_etf_path

    def _terminate_early1(self, maxtoc, forward_root, backward_root):
        vlbs = self._DFScode.vertex_labels()
        for forward_edge in forward_root.keys():
            toc = forward_edge[0]
            if toc != maxtoc:
                continue
            felb = forward_edge[1]
            fvlb = forward_edge[2]
            gids = set()
            for pdfs in forward_root[forward_edge]:
                gids.add(pdfs.gid)
            found_in_backward = False
            for backward_edge in backward_root.keys():
                belb = backward_edge[1]
                if belb != felb:
                    continue
                bvlb = vlbs[backward_edge[0]]
                if bvlb != fvlb:
                    continue
                for pdfs in backward_root[backward_edge]:
                    found_in_backward = True
                    gids.add(pdfs.gid)
            if found_in_backward and len(gids) >= self._min_support:
                print("early termination failure detected")
                return True
        print("early termination failure not detected")
        return False

    def _terminate_early3(self, projected):
        projected_edges, g, edge = self._collect_projected_edges(projected)
        frmlbl, edgelbl, tolbl = g._get_DFSLabels(edge)
        dfs_labels = DFSlabel(frmlbl=frmlbl, edgelbl=edgelbl, tolbl=tolbl)

        # Check if set of projected edges already exists in the values of the specified key
        # return true if yes, false otherwise
        set_of_proj_edges = frozenset(projected_edges)
        if dfs_labels not in self.edge_projection_sets:
            return False, False
        set_of_sets = self.edge_projection_sets[dfs_labels]
        if not set_of_proj_edges in set_of_sets:
            return False, False

        early_termination = False
        if set_of_proj_edges in set_of_sets:
            where_projections = self._get_where_projections(projected)
            where_projections = sorted(where_projections)
            support_projections = len(where_projections)
            projected_edges_sets, projected_edges_lists, max_gid = self._collect_projected_edges_all_pdfs(
                projected)

            edges_directions = self._DFScode.edges_directions()

            termination_by_dfs = False
            should_allow_early_termination = True
            for g in self.edge_projection_sets_closed_graphs[set_of_proj_edges]:
                has_equivalent_occurrence, preserves_directions, isomorphism = g.check_equivalent_occurrence(support_projections,
                                                                                                where_projections,
                                                                                                projected_edges_lists,
                                                                                                edges_directions)
                if not has_equivalent_occurrence:
                    continue

                should_allow_early_termination_1 = self._early_termination_failure_tree.should_allow_early_terminations(self._DFScode, self._DFScode.rmpath, g.gid)
                if not should_allow_early_termination_1:
                    should_allow_early_termination = False

                max_dfs_index = 0
                for index in isomorphism.values():
                    if len(g.DFScode) - index > max_dfs_index:
                        max_dfs_index = len(g.DFScode) - index
                dfs = g.DFScode[0:max_dfs_index]
                for etfDFScode in self.etfDFScodes:
                    if len(etfDFScode) < max_dfs_index:
                        continue
                    #if dfs == etfDFScode:
                    if dfs == etfDFScode[0:max_dfs_index]:
                        termination_by_dfs = True
                        break

                ###if not should_allow_early_termination:
                ####    return False, True

                early_termination = True
                print("dfs code ", self._DFScode, " terminated by gid ", g.gid, " early termination failure ", termination_by_dfs)

        ####
        #### if not should_allow_early_termination:
        ####    early_termination = False
        ###

        if termination_by_dfs:
            early_termination = False

        ### return early_termination, not should_allow_early_termination
        return early_termination, termination_by_dfs

    def _terminate_early(self, projected):
        """
        Checks if the subgraph mining should end early.
        """

        this_early_termination_failure_causing_graphs = list()
        this_terminating_graphs = list()

        projected_edges, g, edge = self._collect_projected_edges(projected)
        frmlbl, edgelbl, tolbl = g._get_DFSLabels(edge)
        dfs_labels = DFSlabel(frmlbl=frmlbl, edgelbl=edgelbl, tolbl=tolbl)

        # Check if set of projected edges already exists in the values of the specified key
        # return true if yes, false otherwise
        set_of_proj_edges = frozenset(projected_edges)
        if dfs_labels not in self.edge_projection_sets:
            return False, this_early_termination_failure_causing_graphs
        set_of_sets = self.edge_projection_sets[dfs_labels]
        if not set_of_proj_edges in set_of_sets:
            return False, this_early_termination_failure_causing_graphs
        if set_of_proj_edges in set_of_sets:
            where_projections = self._get_where_projections(projected)
            where_projections = sorted(where_projections)
            support_projections = len(where_projections)
            projected_edges_sets, projected_edges_lists, max_gid = self._collect_projected_edges_all_pdfs(
                projected)

            edges_directions = self._DFScode.edges_directions()
            early_termination = False
            for g in self.edge_projection_sets_closed_graphs[set_of_proj_edges]:
                has_equivalent_occurrence, preserves_directions, isomorphism = g.check_equivalent_occurrence(support_projections, where_projections, projected_edges_lists, edges_directions)
                if not has_equivalent_occurrence:
                    continue


                if not preserves_directions and g.gid > self._frequent_graph_gid_before_root:
                    early_termination = False
                    this_early_termination_failure_causing_graphs.append(g.gid)
                    continue
                if g.gid in self._early_termination_failure_causing_graphs and g.gid > self._frequent_graph_gid_before_root:
                    early_termination = False
                    this_early_termination_failure_causing_graphs.append(g.gid)
                    continue

                this_terminating_graphs.append(g.gid)
                if len(this_early_termination_failure_causing_graphs) == 0:
                    early_termination = True

            if early_termination:
                for gid in self._early_termination_failure_causing_graphs:
                    can_be_descendant_of = self._early_termination_failure_causing_graphs[gid].can_be_descendant_of(
                        support_projections, where_projections, projected_edges_lists)
                    if can_be_descendant_of:
                        this_early_termination_failure_causing_graphs.extend(this_terminating_graphs)
                        early_termination = False
                        break

            return early_termination, this_early_termination_failure_causing_graphs


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
                is_supergraph, preserves_directions = g_tag.is_supergraph_of_with_support_projections(
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
