"""GapTree Sort Algorithm for reading order sorting"""
from typing import Callable


class GapTree:
    """
    GapTree reading order sort algorithm.
    Sorts text blocks in human reading order.
    """

    def __init__(self, get_bbox: Callable):
        """
        :param get_bbox: Function that takes a text block and returns (x0, y0, x1, y1)
        """
        self.get_bbox = get_bbox

    def sort(self, text_blocks: list):
        """Sort text blocks in human reading order"""
        units, page_l, page_r = self._get_units(text_blocks, self.get_bbox)
        cuts, rows = self._get_cuts_rows(units, page_l, page_r)
        root = self._get_layout_tree(cuts, rows)
        nodes = self._preorder_traversal(root)
        new_text_blocks = self._get_text_blocks(nodes)

        self.current_rows = rows
        self.current_cuts = cuts
        self.current_nodes = nodes

        return new_text_blocks

    def get_nodes_text_blocks(self):
        """Get text blocks grouped by layout nodes. Must be called after sort."""
        result = []
        for node in self.current_nodes:
            tbs = []
            if node["units"]:
                for unit in node["units"]:
                    tbs.append(unit[1])
                result.append(tbs)
        return result

    def _get_units(self, text_blocks, get_bbox):
        """Get unit list and page boundaries"""
        units = []
        page_l, page_r = float("inf"), -1
        for tb in text_blocks:
            x0, y0, x2, y2 = get_bbox(tb)
            units.append(((x0, y0, x2, y2), tb))
            if x0 < page_l:
                page_l = x0
            if x2 > page_r:
                page_r = x2
        units.sort(key=lambda a: a[0][1])
        return units, page_l, page_r

    def _get_cuts_rows(self, units, page_l, page_r):
        """Get vertical cuts and rows"""
        def update_gaps(gaps1, gaps2):
            flags1 = [True for _ in gaps1]
            flags2 = [True for _ in gaps2]
            new_gaps1 = []
            for i1, g1 in enumerate(gaps1):
                l1, r1, _ = g1
                for i2, g2 in enumerate(gaps2):
                    l2, r2, _ = g2
                    inter_l = max(l1, l2)
                    inter_r = min(r1, r2)
                    if inter_l <= inter_r:
                        new_gaps1.append((inter_l, inter_r, g1[2]))
                        flags1[i1] = False
                        flags2[i2] = False
            for i2, f2 in enumerate(flags2):
                if f2:
                    new_gaps1.append(gaps2[i2])
            del_gaps1 = []
            for i1, f1 in enumerate(flags1):
                if f1:
                    del_gaps1.append(gaps1[i1])
            return new_gaps1, del_gaps1

        page_l -= 1
        page_r += 1
        rows = []
        completed_cuts = []
        gaps = []
        row_index = 0
        unit_index = 0
        l_units = len(units)

        while unit_index < l_units:
            unit = units[unit_index]
            u_bottom = unit[0][3]
            row = [unit]
            for i in range(unit_index + 1, len(units)):
                next_u = units[i]
                next_top = next_u[0][1]
                if next_top > u_bottom:
                    break
                row.append(next_u)
                unit_index = i
            row.sort(key=lambda x: (x[0][0], x[0][2]))
            row_gaps = []
            search_start = page_l
            for u in row:
                l = u[0][0]
                r = u[0][2]
                if l > search_start:
                    row_gaps.append((search_start, l, row_index))
                if r > search_start:
                    search_start = r
            row_gaps.append((search_start, page_r, row_index))
            gaps, del_gaps = update_gaps(gaps, row_gaps)
            row_max = row_index - 1
            for dg1 in del_gaps:
                completed_cuts.append((*dg1, row_max))
            rows.append(row)
            unit_index += 1
            row_index += 1

        row_max = len(rows) - 1
        for g in gaps:
            completed_cuts.append((*g, row_max))
        completed_cuts.sort(key=lambda c: c[0])
        return completed_cuts, rows

    def _get_layout_tree(self, cuts, rows):
        """Build layout tree"""
        rows_gaps = [[] for _ in rows]
        for g_i, cut in enumerate(cuts):
            for r_i in range(cut[2], cut[3] + 1):
                rows_gaps[r_i].append((cut[0], cut[1]))

        root = {
            "x_left": cuts[0][0] - 1,
            "x_right": cuts[-1][1] + 1,
            "r_top": -1,
            "r_bottom": -1,
            "units": [],
            "children": [],
        }
        completed_nodes = [root]
        now_nodes = []

        def complete(node):
            node_r = node["x_right"] - 2
            max_nodes = []
            max_r = -2
            for com_node in completed_nodes:
                if node_r < com_node["x_left"] or node_r > com_node["x_right"] + 0.0001:
                    continue
                if com_node["r_bottom"] >= node["r_top"]:
                    continue
                if com_node["r_bottom"] > max_r:
                    max_r = com_node["r_bottom"]
                    max_nodes = [com_node]
                    continue
                if com_node["r_bottom"] == max_r:
                    max_nodes.append(com_node)
                    continue
            max_node = max(max_nodes, key=lambda n: n["x_right"])
            max_node["children"].append(node)
            completed_nodes.append(node)

        for r_i, row in enumerate(rows):
            row_gaps = rows_gaps[r_i]
            u_i = g_i = 0
            new_nodes = []
            for node in now_nodes:
                l_flag = r_flag = False
                completed_flag = False
                x_left = node["x_left"]
                x_right = node["x_right"]
                for gap in row_gaps:
                    if gap[1] == x_left:
                        l_flag = True
                    if gap[0] == x_right:
                        r_flag = True
                    if x_left < gap[0] < x_right or x_left < gap[1] < x_right:
                        completed_flag = True
                        break
                if not l_flag or not r_flag:
                    completed_flag = True
                if completed_flag:
                    complete(node)
                else:
                    node["r_bottom"] = r_i
                    new_nodes.append(node)
            now_nodes = new_nodes

            while u_i < len(row):
                unit = row[u_i]
                x_l = row_gaps[g_i][1]
                x_r = row_gaps[g_i + 1][0]
                if unit[0][0] + 0.0001 > x_r:
                    g_i += 1
                    continue
                flag = False
                for node in now_nodes:
                    if node["x_left"] == x_l and node["x_right"] == x_r:
                        node["units"].append(unit)
                        flag = True
                        break
                if flag:
                    u_i += 1
                    continue
                now_nodes.append({
                    "x_left": x_l,
                    "x_right": x_r,
                    "r_top": r_i,
                    "r_bottom": r_i,
                    "units": [unit],
                    "children": [],
                })
                u_i += 1

        for node in now_nodes:
            complete(node)
        for node in completed_nodes:
            node["children"].sort(key=lambda n: n["x_left"])
            node["units"].sort(key=lambda u: u[0][1])
        return root

    def _preorder_traversal(self, root):
        """Preorder traversal of layout tree"""
        if not root:
            return []
        stack = [root]
        result = []
        while stack:
            node = stack.pop()
            result.append(node)
            stack += reversed(node["children"])
        return result

    def _get_text_blocks(self, nodes):
        """Extract text blocks from node sequence"""
        result = []
        for node in nodes:
            for unit in node["units"]:
                result.append(unit[1])
        return result
