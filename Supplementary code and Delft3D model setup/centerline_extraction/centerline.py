import os
import numpy as np
from skimage.morphology import medial_axis
from skimage.draw import polygon
from svgpathtools import svg2paths
from shapely.geometry import LineString, Polygon, MultiPolygon, Point
from shapely.ops import unary_union
import networkx as nx
from scipy.ndimage import gaussian_filter
from scipy.interpolate import splprep, splev
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


class CenterlineExtractor:
    """Polygon centerline extractor - optimized version (remove external points and connect with straight lines)"""

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 resolution: int = 5000,
                 smoothness: float = 0.01,
                 gauss_sigma: float = 1.0,
                 distance_threshold: float = 2.0,
                 spline_smoothness: float = 0.01,  # spline smoothing parameter
                 chaikin_iterations: int = 2,  # number of Chaikin smoothing iterations
                 verbose: bool = True,
                 include_background: bool = True,
                 output_skeleton: bool = False,
                 random_seed: int = None,
                 random_seed_by_filename: dict = None):
        """
        Initialize the centerline extractor.

        Args:
            input_dir: path of the input SVG folder
            output_dir: path of the output centerline folder
            resolution: resolution (pixels), default 5000
            smoothness: boundary simplification tolerance (relative to boundary length)
            gauss_sigma: Gaussian smoothing parameter
            distance_threshold: skeleton pruning threshold (pixels)
            spline_smoothness: spline smoothing parameter (0-1, smaller is smoother)
            chaikin_iterations: number of Chaikin smoothing iterations
            verbose: whether to print detailed information to the console
            include_background: whether to include the original polygon background in the output
            output_skeleton: whether to output skeleton visualization SVG for debugging
            random_seed: optional seed used by scikit-image's medial_axis
                tie-breaking. Use None to leave the tie-breaking unfixed.
            random_seed_by_filename: optional per-file seed map.  This is useful
                when reproducing a figure whose centerline choices were fixed
                separately for each basin.
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.verbose = verbose
        self.include_background = include_background
        self.output_skeleton = output_skeleton
        self.random_seed = random_seed
        self.default_random_seed = random_seed
        self.random_seed_by_filename = random_seed_by_filename or {}

        # Processing parameters
        self.resolution = resolution
        self.smoothness = smoothness
        self.gauss_sigma = gauss_sigma
        self.distance_threshold = distance_threshold
        self.spline_smoothness = spline_smoothness
        self.chaikin_iterations = chaikin_iterations

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Store results
        self.results = []

    def smooth_boundary(self, poly, tolerance_factor=0.01):
        """Smooth the boundary using the Douglas-Peucker algorithm."""
        if isinstance(poly, MultiPolygon):
            largest = max(poly.geoms, key=lambda p: p.area)
            boundary = largest.exterior
        else:
            boundary = poly.exterior

        # Compute total boundary length
        boundary_length = boundary.length

        # Set the simplification tolerance
        tolerance = boundary_length * tolerance_factor

        # Apply the simplification algorithm
        simplified = poly.simplify(tolerance, preserve_topology=True)

        # If simplification fails, try buffer-based smoothing
        if simplified.is_empty or simplified.geom_type not in ['Polygon', 'MultiPolygon']:
            smoothed = poly.buffer(tolerance).buffer(-tolerance)
            if smoothed.is_empty or smoothed.geom_type not in ['Polygon', 'MultiPolygon']:
                return poly
            return smoothed

        return simplified

    def chaikin_smoothing(self, points, iterations=2):
        """
        Chaikin curve smoothing algorithm.
        Each iteration doubles the number of points, making the curve smoother.
        """
        if len(points) < 3:
            return points

        for _ in range(iterations):
            new_points = []

            # Keep the first point
            new_points.append(points[0])

            # Apply the Chaikin algorithm to each pair of adjacent points
            for i in range(len(points) - 1):
                p0 = points[i]
                p1 = points[i + 1]

                # Compute the quarter points
                q = (
                    0.75 * p0[0] + 0.25 * p1[0],
                    0.75 * p0[1] + 0.25 * p1[1]
                )
                r = (
                    0.25 * p0[0] + 0.75 * p1[0],
                    0.25 * p0[1] + 0.75 * p1[1]
                )

                new_points.append(q)
                new_points.append(r)

            # Keep the last point
            new_points.append(points[-1])
            points = new_points

        return points

    def spline_smoothing(self, points, s=0.01, num_points=None):
        """
        Smooth a point sequence using a B-spline.

        Args:
            points: original point list [(x1,y1), (x2,y2), ...]
            s: smoothing parameter (0-1); smaller is smoother
            num_points: number of points after smoothing; defaults to twice the original count

        Returns:
            The smoothed point list.
        """
        if len(points) < 4:
            # Too few points: fall back to Chaikin smoothing
            return self.chaikin_smoothing(points, iterations=2)

        try:
            # Separate x and y coordinates
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]

            # Compute cumulative distance as the parameter
            t = np.zeros(len(x_coords))
            for i in range(1, len(x_coords)):
                dx = x_coords[i] - x_coords[i - 1]
                dy = y_coords[i] - y_coords[i - 1]
                t[i] = t[i - 1] + np.sqrt(dx * dx + dy * dy)

            # Normalize the parameter to [0, 1]
            if t[-1] > 0:
                t = t / t[-1]
            else:
                t = np.linspace(0, 1, len(x_coords))

            # Set spline parameters
            if num_points is None:
                num_points = min(100, max(len(points) * 2, 20))

            # Compute the spline curve
            # For closed curves per=1 may be needed, but centerlines are usually open curves
            tck, u = splprep([x_coords, y_coords], u=t, s=s, k=3)

            # Generate new parameter points
            u_new = np.linspace(0, 1, num_points)

            # Evaluate the smoothed points
            x_new, y_new = splev(u_new, tck)

            # Combine into a point list
            smoothed_points = list(zip(x_new, y_new))

            return smoothed_points

        except Exception as e:
            if self.verbose:
                print(f"Spline smoothing failed, falling back to Chaikin smoothing: {e}")
            # On spline failure, fall back to Chaikin
            return self.chaikin_smoothing(points, iterations=self.chaikin_iterations)

    def gaussian_smooth_image(self, img, sigma=1.0):
        """Smooth a binary image with a Gaussian filter."""
        img_float = img.astype(np.float32)
        smoothed = gaussian_filter(img_float, sigma=sigma)
        return (smoothed > 0.5).astype(np.uint8)

    def extract_robust_skeleton(self, skel, distance, distance_threshold=2):
        """Extract a robust skeleton and prune minor branches.

        Args:
            skel: the full skeleton (boolean array) produced by medial_axis
            distance: the distance transform returned alongside the skeleton
            distance_threshold: skeleton pruning threshold (pixels)
        """
        skeleton_distances = distance[skel]

        if len(skeleton_distances) > 0:
            median_distance = np.median(skeleton_distances)

            if distance_threshold > 0:
                robust_skel = skel & (distance > distance_threshold)

                if np.sum(robust_skel) == 0:
                    robust_skel = skel
                elif np.sum(robust_skel) < np.sum(skel) * 0.1:
                    robust_skel = skel & (distance > median_distance * 0.5)
            else:
                robust_skel = skel
        else:
            robust_skel = skel

        return robust_skel

    def build_skeleton_graph(self, skel):
        """Build a graph representation of the skeleton."""
        G = nx.Graph()
        pts = np.column_stack(np.nonzero(skel))

        if len(pts) == 0:
            return G, {}

        node_indices = {}
        for idx, (y, x) in enumerate(pts):
            node_indices[(x, y)] = idx
            G.add_node(idx, pos=(x, y))

        for (x, y), node_id in node_indices.items():
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == dy == 0:
                        continue
                    nb = (x + dx, y + dy)
                    if nb in node_indices:
                        nb_id = node_indices[nb]
                        if not G.has_edge(node_id, nb_id):
                            weight = np.sqrt(dx * dx + dy * dy)
                            G.add_edge(node_id, nb_id, weight=weight)

        return G, node_indices

    def find_main_skeleton_path(self, G, node_indices):
        """Find the main path of the skeleton (the path with the largest endpoint distance)."""
        if len(G.nodes) == 0:
            return None

        # Find all endpoints (nodes with degree 1)
        endpoints = [node for node in G.nodes if G.degree(node) == 1]

        if len(endpoints) < 2:
            # No endpoints or only one: use all nodes
            all_nodes = list(G.nodes)
            if len(all_nodes) == 0:
                return None

            max_len = 0
            best_path = None

            # Compute shortest paths between all node pairs and keep the longest
            for i in range(len(all_nodes)):
                try:
                    lengths = nx.single_source_dijkstra_path_length(G, all_nodes[i])
                    for j in range(i + 1, len(all_nodes)):
                        if all_nodes[j] in lengths:
                            d = lengths[all_nodes[j]]
                            if d > max_len:
                                max_len = d
                                try:
                                    best_path = nx.shortest_path(G, all_nodes[i], all_nodes[j], weight='weight')
                                except:
                                    continue
                except:
                    continue

            return best_path

        # Multiple endpoints: find the path with the largest inter-endpoint distance
        max_len = 0
        best_path = None

        for i, start in enumerate(endpoints):
            try:
                lengths = nx.single_source_dijkstra_path_length(G, start)

                for j, end in enumerate(endpoints[i + 1:], i + 1):
                    if end in lengths:
                        d = lengths[end]
                        if d > max_len:
                            max_len = d
                            try:
                                best_path = nx.shortest_path(G, start, end, weight='weight')
                            except:
                                continue
            except:
                continue

        return best_path

    def snap_points_to_boundary(self, points, poly):
        """Snap points onto the polygon boundary (if a point lies outside)."""
        snapped_points = []

        for x, y in points:
            point = Point(x, y)

            # If the point is inside or on the boundary, keep the original coordinates
            if poly.contains(point) or poly.touches(point):
                snapped_points.append((x, y))
            else:
                # Point is outside: find the nearest point on the boundary
                nearest_point = poly.exterior.interpolate(poly.exterior.project(point))
                snapped_points.append((nearest_point.x, nearest_point.y))

        return snapped_points

    def simplify_line_points(self, points, tolerance=0.5):
        """
        Simplify line points to reduce redundant points.
        A simple implementation based on the Douglas-Peucker algorithm.
        """
        if len(points) < 3:
            return points

        def point_line_distance(point, start, end):
            """Distance from a point to a line segment."""
            if start == end:
                dx = point[0] - start[0]
                dy = point[1] - start[1]
                return np.sqrt(dx * dx + dy * dy)

            # Squared length of the segment
            line_length_sq = (end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2

            # Projection parameter
            if line_length_sq == 0:
                return 0

            t = max(0, min(1, ((point[0] - start[0]) * (end[0] - start[0]) +
                               (point[1] - start[1]) * (end[1] - start[1])) / line_length_sq))

            # Projection point
            projection = (start[0] + t * (end[0] - start[0]),
                          start[1] + t * (end[1] - start[1]))

            # Distance
            dx = point[0] - projection[0]
            dy = point[1] - projection[1]
            return np.sqrt(dx * dx + dy * dy)

        # Recursive simplification
        def douglas_peucker(points, start_idx, end_idx, tolerance):
            """Recursively apply the Douglas-Peucker algorithm."""
            if end_idx <= start_idx + 1:
                return []

            # Find the point farthest from the segment
            max_distance = 0
            max_index = start_idx

            start_point = points[start_idx]
            end_point = points[end_idx]

            for i in range(start_idx + 1, end_idx):
                distance = point_line_distance(points[i], start_point, end_point)
                if distance > max_distance:
                    max_distance = distance
                    max_index = i

            result_points = []

            if max_distance > tolerance:
                # Recurse on the left and right sub-segments
                left_result = douglas_peucker(points, start_idx, max_index, tolerance)
                right_result = douglas_peucker(points, max_index, end_idx, tolerance)

                # Merge results (without duplicating max_index)
                result_points = left_result + [points[max_index]] + right_result
            else:
                # All points within tolerance: keep only the endpoints
                result_points = []

            return result_points

        # Apply the simplification algorithm
        simplified = douglas_peucker(points, 0, len(points) - 1, tolerance)

        # Ensure the start and end points are included
        final_points = [points[0]] + simplified + [points[-1]]

        # Remove possibly duplicated points
        unique_points = []
        for i, point in enumerate(final_points):
            if i == 0 or point != final_points[i - 1]:
                unique_points.append(point)

        return unique_points

    def remove_external_points_and_connect(self, points, poly):
        """
        Remove points outside the polygon, then connect the breakpoints with straight lines.

        Args:
            points: point list [(x1,y1), (x2,y2), ...]
            poly: the polygon

        Returns:
            The processed point list.
        """
        if len(points) < 2:
            return points

        # 1. Mark which points are inside the polygon
        inside_flags = []
        for x, y in points:
            point = Point(x, y)
            inside_flags.append(poly.contains(point) or poly.touches(point))

        # 2. If all points are inside, return directly
        if all(inside_flags):
            return points

        # 3. Remove external points and collect internal segments
        segments = []
        current_segment = []

        for i, (point, is_inside) in enumerate(zip(points, inside_flags)):
            if is_inside:
                current_segment.append(point)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []

        # Append the last segment
        if current_segment:
            segments.append(current_segment)

        # 4. If there is only one internal segment, return it directly
        if len(segments) == 1:
            return segments[0]

        # 5. Connect the segments with straight lines
        final_points = []

        for i in range(len(segments)):
            # Append the current segment
            if i == 0:
                # Append the entire first segment
                final_points.extend(segments[i])
            else:
                # Connect the last point of the previous segment to the first point of the current segment
                prev_last = final_points[-1]
                curr_first = segments[i][0]

                # Append the connection points (simple straight-line connection)
                final_points.append(prev_last)
                final_points.append(curr_first)

                # Append the rest of the current segment (skip the first point, already added as connection)
                final_points.extend(segments[i][1:])

        # 6. If too few final points remain, return the original points
        if len(final_points) < 2:
            if self.verbose:
                print("Warning: too few points after connection, returning original points")
            return points

        return final_points

    def calculate_centerline(self, poly, smoothed_poly):
        """
        Compute the centerline (snap, smooth, remove external points and connect with straight lines).
        
        Returns:
            Tuple: (centerline, debug_info) where debug_info is a dict containing skeleton data for visualization
        """
        minx, miny, maxx, maxy = poly.bounds

        sx = (self.resolution - 1) / (maxx - minx)
        sy = (self.resolution - 1) / (maxy - miny)
        scale = min(sx, sy)

        def img_to_world(x, y):
            return x / scale + minx, y / scale + miny

        def world_to_img(x, y):
            return (x - minx) * scale, (y - miny) * scale

        # Create the binary image
        img = np.zeros((self.resolution, self.resolution), dtype=np.uint8)

        if isinstance(poly, MultiPolygon):
            for geom in poly.geoms:
                xs, ys = geom.exterior.xy
                rr, cc = polygon(
                    [(y - miny) * scale for y in ys],
                    [(x - minx) * scale for x in xs],
                    img.shape
                )
                img[rr.astype(int), cc.astype(int)] = 1
        else:
            xs, ys = poly.exterior.xy
            rr, cc = polygon(
                [(y - miny) * scale for y in ys],
                [(x - minx) * scale for x in xs],
                img.shape
            )
            img[rr.astype(int), cc.astype(int)] = 1

        # Gaussian smoothing
        img = self.gaussian_smooth_image(img, self.gauss_sigma)

        # Extract the skeleton (returns both full and pruned skeleton).
        # medial_axis uses a random tie-breaker for equal-distance pixels;
        # always pass an explicit seed so the centerline is reproducible.
        try:
            skel, distance = medial_axis(
                img, return_distance=True, rng=self.random_seed
            )
        except TypeError:
            # Compatibility with older scikit-image releases that called the
            # argument ``random_state`` instead of ``rng``.
            skel, distance = medial_axis(
                img, return_distance=True, random_state=self.random_seed
            )
        robust_skel = self.extract_robust_skeleton(skel, distance, self.distance_threshold)

        # Build the skeleton graph
        G, node_indices = self.build_skeleton_graph(robust_skel)

        if len(G.nodes) == 0:
            return None, None

        # Find the main path directly on the original skeleton graph (no trimming)
        best_path = self.find_main_skeleton_path(G, node_indices)

        if best_path is None:
            return None, None

        # Store debug info for skeleton visualization
        debug_info = {
            'img': img,
            'skel': skel,
            'robust_skel': robust_skel,
            'best_path': best_path,
            'node_indices': node_indices,
            'G': G,
            'minx': minx,
            'miny': miny,
            'maxx': maxx,
            'maxy': maxy,
            'scale': scale
        }

        # Get the pixel coordinates along the path
        pixel_path = [G.nodes[node]['pos'] for node in best_path]

        # Convert to world coordinates
        world_coords = [img_to_world(x, y) for x, y in pixel_path]

        # Snap external points to the boundary (only once)
        snapped_coords = self.snap_points_to_boundary(world_coords, smoothed_poly)

        # Simplify the point sequence (reduce redundant points)
        simplified_coords = self.simplify_line_points(snapped_coords, tolerance=0.5)

        # Smooth the centerline
        if len(simplified_coords) > 2:
            # First apply spline smoothing
            smoothed_coords = self.spline_smoothing(
                simplified_coords,
                s=self.spline_smoothness,
                num_points=min(100, max(len(simplified_coords) * 2, 30))
            )

            # Optionally apply Chaikin smoothing afterwards
            if self.chaikin_iterations > 0:
                smoothed_coords = self.chaikin_smoothing(
                    smoothed_coords,
                    iterations=min(self.chaikin_iterations, 3)
                )
        else:
            smoothed_coords = simplified_coords

        # Remove external points and connect with straight lines
        final_coords = self.remove_external_points_and_connect(smoothed_coords, smoothed_poly)

        # Create the line object
        centerline = LineString(final_coords)

        return centerline, debug_info

    def create_skeleton_debug_svg(self, img, skel, robust_skel, best_path, G,
                                   output_path, minx, miny, maxx, maxy, scale, 
                                   poly=None, ext_triangle=None, entrance_pts=None):
        """
        Create an SVG for skeleton visualization and debugging.
        
        Shows:
        - Original polygon outline
        - Extension triangle
        - Inlet line (blue dashed)
        - Full skeleton (light cyan dots, semi-transparent for context)
        - Pruned skeleton (dark purple dots, prominent for emphasis)
        - Main path (red line, the final centerline)
        """
        def img_to_world(x, y):
            return x / scale + minx, y / scale + miny
        
        w, h = maxx - minx, maxy - miny
        extent = max(w, h)
        margin = extent * 0.05
        
        view_box = f"{minx - margin} {miny - margin} {w + 2 * margin} {h + 2 * margin}"
        
        # Compute adaptive line widths
        bg_stroke = max(1.0, extent / 1000.0)
        skel_stroke = max(3.0, extent / 300.0)
        entrance_stroke = max(2.0, extent / 500.0)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">\n')
            f.write('<!-- Skeleton Debug Visualization with Context -->\n')
            f.write('<!-- Original outline: no fill | Triangle: outline only | Inlet: blue dashed -->\n')
            f.write('<!-- Full skeleton: light cyan dots (60% opacity, context) | Pruned skeleton: dark purple dots (prominent) | Main path: red -->\n')
            
            # 1. Draw the original polygon outline
            if poly:
                f.write('<!-- Original polygon outline -->\n')
                if isinstance(poly, MultiPolygon):
                    for geom in poly.geoms:
                        xs, ys = geom.exterior.xy
                        f.write('<path d="')
                        for i, (x, y) in enumerate(zip(xs, ys)):
                            f.write(f'{"M" if i == 0 else "L"}{x},{y} ')
                        f.write(f'Z" fill="none" stroke="#999999" stroke-width="{bg_stroke}"/>\n')
                else:
                    xs, ys = poly.exterior.xy
                    f.write('<path d="')
                    for i, (x, y) in enumerate(zip(xs, ys)):
                        f.write(f'{"M" if i == 0 else "L"}{x},{y} ')
                    f.write(f'Z" fill="none" stroke="#999999" stroke-width="{bg_stroke}"/>\n')
            
            # 2. Draw the extension triangle outline only
            if ext_triangle:
                f.write('<!-- Extension triangle (outline only) -->\n')
                xs, ys = ext_triangle.exterior.xy
                f.write('<path d="')
                for i, (x, y) in enumerate(zip(xs, ys)):
                    f.write(f'{"M" if i == 0 else "L"}{x},{y} ')
                f.write(f'Z" fill="none" stroke="#00cc00" stroke-width="{bg_stroke}" opacity="0.6"/>\n')
            
            # 3. Draw inlet line
            if entrance_pts and len(entrance_pts) == 2:
                f.write('<!-- Inlet line -->\n')
                p1, p2 = entrance_pts
                f.write(f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" ')
                f.write(f'stroke="#0066ff" stroke-width="{entrance_stroke}" stroke-dasharray="{entrance_stroke*2},{entrance_stroke}"/>\n')
            
            # 4. Draw full skeleton (light cyan - visible but not dominant)
            f.write('<!-- Full skeleton (light cyan dots) -->\n')
            for y in range(skel.shape[0]):
                for x in range(skel.shape[1]):
                    if skel[y, x]:
                        wx, wy = img_to_world(x, y)
                        f.write(f'<circle cx="{wx}" cy="{wy}" r="1.2" fill="#40d9ff" stroke="none" opacity="0.6"/>\n')
            
            # 5. Draw pruned skeleton (dark purple - prominent)
            f.write('<!-- Pruned skeleton (dark purple dots - prominent) -->\n')
            for y in range(robust_skel.shape[0]):
                for x in range(robust_skel.shape[1]):
                    if robust_skel[y, x]:
                        wx, wy = img_to_world(x, y)
                        f.write(f'<circle cx="{wx}" cy="{wy}" r="2" fill="#7722dd" stroke="none"/>\n')
            
            # 6. Draw main path (red line)
            if best_path and len(best_path) > 1:
                f.write('<!-- Main path (red line) -->\n')
                main_path_coords = [G.nodes[node]['pos'] for node in best_path]
                world_main_path = [img_to_world(x, y) for x, y in main_path_coords]
                
                f.write('<polyline points="')
                for x, y in world_main_path:
                    f.write(f'{x},{y} ')
                f.write('" fill="none" stroke="#ff3300" stroke-width="{}" stroke-linecap="round" stroke-linejoin="round"/>\n'.format(skel_stroke))
                
                # Mark endpoints
                for x, y in world_main_path:
                    f.write(f'<circle cx="{x}" cy="{y}" r="2" fill="#ff3300" stroke="white" stroke-width="0.5"/>\n')
            
            f.write('</svg>\n')

    def create_svg_with_background(self, poly, centerline, output_path, entrance_pts=None):
        """Create an SVG file containing the original polygon background, the centerline, and the inlet line."""
        minx, miny, maxx, maxy = poly.bounds
        w, h = maxx - minx, maxy - miny

        # Compute adaptive line widths and margin
        extent = max(w, h)
        margin = extent * 0.05
        bg_stroke = max(1.0, extent / 1000.0)
        skel_stroke = max(3.0, extent / 300.0)
        entrance_stroke = max(2.0, extent / 500.0)  # inlet line width
        point_radius = max(5.0, extent / 150.0)

        view_box = f"{minx - margin} {miny - margin} {w + 2 * margin} {h + 2 * margin}"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{view_box}">\n')

            # 1. Draw the original polygon background
            if self.include_background:
                f.write(f'<!-- Original polygon outline -->\n')
                if isinstance(poly, MultiPolygon):
                    for geom_idx, geom in enumerate(poly.geoms):
                        xs, ys = geom.exterior.xy
                        f.write('<path d="')
                        for i, (x, y) in enumerate(zip(xs, ys)):
                            f.write(f'{"M" if i == 0 else "L"}{x},{y} ')
                        f.write(f'Z" fill="none" stroke="#666666" stroke-width="{bg_stroke}"/>\n')
                else:
                    xs, ys = poly.exterior.xy
                    f.write('<path d="')
                    for i, (x, y) in enumerate(zip(xs, ys)):
                        f.write(f'{"M" if i == 0 else "L"}{x},{y} ')
                    f.write(f'Z" fill="none" stroke="#666666" stroke-width="{bg_stroke}"/>\n')

            # 2. Draw the inlet line (blue dashed line)
            if entrance_pts and len(entrance_pts) == 2:
                p1, p2 = entrance_pts
                f.write(f'<!-- Inlet line -->\n')
                f.write(f'<line x1="{p1[0]}" y1="{p1[1]}" x2="{p2[0]}" y2="{p2[1]}" ')
                f.write(f'stroke="#0066ff" stroke-width="{entrance_stroke}" stroke-dasharray="{entrance_stroke*2},{entrance_stroke}"/>\n')

            # Draw the centerline
            f.write(f'<!-- Extracted centerline -->\n')
            if hasattr(centerline, 'coords'):
                world_coords = list(centerline.coords)
            else:
                world_coords = list(centerline.exterior.coords) if hasattr(centerline, 'exterior') else []

            f.write('<path d="')
            for i, (x, y) in enumerate(world_coords):
                f.write(f'{"M" if i == 0 else "L"}{x},{y} ')
            f.write(
                f'" stroke="#ff3300" fill="none" stroke-width="{skel_stroke}" stroke-linecap="round" stroke-linejoin="round"/>\n')

            # Red marker: centerline-inlet intersection (effective bay length origin).
            centerline_line = LineString(world_coords)
            if entrance_pts and len(entrance_pts) == 2:
                inlet_line = LineString(list(entrance_pts))
                intersection = centerline_line.intersection(inlet_line)
                intersection_pts = []
                if not intersection.is_empty:
                    if intersection.geom_type == "Point":
                        intersection_pts = [intersection]
                    elif intersection.geom_type == "MultiPoint":
                        intersection_pts = list(intersection.geoms)

                if intersection_pts:
                    f.write("<!-- Centerline-inlet intersection (effective bay length origin) -->\n")
                    for pt in intersection_pts:
                        f.write(
                            f'<circle cx="{pt.x}" cy="{pt.y}" r="{point_radius}" fill="#ff3333" stroke="#ffffff" stroke-width="{bg_stroke}"/>\n')

            f.write('</svg>\n')

    def process_single_svg(self, svg_path: str):
        """
        Process a single SVG file.

        Args:
            svg_path: path of the SVG file

        Returns:
            Dict: processing result dictionary
        """
        filename = os.path.basename(svg_path)
        base_name = os.path.splitext(filename)[0]

        if self.verbose:
            print(f"\nProcessing file: {filename}")

        try:
            # 1. Read the SVG polygon features
            if self.verbose:
                print(f"  [1] Reading SVG file...")

            paths, _ = svg2paths(svg_path)
            polys = []

            for p in paths:
                pts = [complex(seg.start) for seg in p]
                pts.append(complex(p[-1].end))
                coords = [(pt.real, pt.imag) for pt in pts]
                if len(coords) >= 3:
                    polys.append(Polygon(coords))

            if not polys:
                raise RuntimeError("No valid polygon features found in SVG")

            # Merge all polygons
            poly = unary_union(polys)

            # Extract the extension triangle (the part beyond the inlet).
            # Path structure produced by close_outline.py: [start, ..., end, extension_point]
            # Extension triangle vertices: start (coords[0]), end (coords[-3]), extension vertex (coords[-2])
            coords = list(poly.exterior.coords)
            start_pt = coords[0]
            end_pt = coords[-3]
            mid_ext_pt = coords[-2]
            ext_triangle = Polygon([start_pt, end_pt, mid_ext_pt])

            total_area = poly.area
            triangle_area = ext_triangle.area
            bay_area = total_area - triangle_area

            if self.verbose:
                print(f"    Total area: {total_area:.2f}")
                print(f"    Extension triangle area: {triangle_area:.2f}")
                print(f"    Effective bay area: {bay_area:.2f}")

            # 2. Boundary smoothing preprocessing
            if self.verbose:
                print(f"  [2] Boundary smoothing preprocessing...")

            smoothed_poly = self.smooth_boundary(poly, self.smoothness)
            smoothed_area = smoothed_poly.area

            if self.verbose:
                print(f"    Smoothed area: {smoothed_area:.2f}")
                print(f"    Area change: {(smoothed_area - total_area) / total_area * 100:.2f}%")

            # 3. Centerline extraction
            if self.verbose:
                print(f"  [3] Centerline extraction (resolution: {self.resolution})...")

            centerline, debug_info = self.calculate_centerline(poly, smoothed_poly)

            if centerline is None:
                raise RuntimeError("Failed to extract centerline")

            # Output skeleton visualization if enabled
            if self.output_skeleton and debug_info:
                skeleton_svg_path = os.path.join(self.output_dir, f"{base_name}_skeleton_debug.svg")
                self.create_skeleton_debug_svg(
                    debug_info['img'],
                    debug_info['skel'],
                    debug_info['robust_skel'],
                    debug_info['best_path'],
                    debug_info['G'],
                    skeleton_svg_path,
                    debug_info['minx'],
                    debug_info['miny'],
                    debug_info['maxx'],
                    debug_info['maxy'],
                    debug_info['scale'],
                    poly=poly,
                    ext_triangle=ext_triangle,
                    entrance_pts=(start_pt, end_pt)
                )
                if self.verbose:
                    print(f"    Skeleton debug SVG: {os.path.basename(skeleton_svg_path)}")

            # 4. Length deduction and index computation
            total_length = centerline.length

            # Length of the skeleton inside the extension triangle
            if ext_triangle.intersects(centerline):
                ext_line_part = centerline.intersection(ext_triangle)
                extension_length = ext_line_part.length
            else:
                extension_length = 0

            bay_length = total_length - extension_length

            # Compute the ratio using the deducted bay area and length
            ratio = (bay_length ** 2) / bay_area if bay_area > 0 else 0

            if self.verbose:
                print(f"  [4] Computing index (extension part deducted)...")
                print(f"    Total length: {total_length:.2f}")
                print(f"    Extension length: {extension_length:.2f}")
                print(f"    Effective bay length: {bay_length:.2f}")
                print(f"    Final L^2/A ratio: {ratio:.2f}")

            # 5. Output the skeleton SVG (with background)
            output_filename = f"{base_name}_centerline.svg"
            output_path = os.path.join(self.output_dir, output_filename)

            # Create the SVG file with background
            self.create_svg_with_background(poly, centerline, output_path, entrance_pts=(start_pt, end_pt))

            if self.verbose:
                print(f"  [5] Writing output...")
                print(f"    Centerline SVG: {output_filename}")
                if self.include_background:
                    print(f"    Includes original polygon background: yes")

            # Return the result
            result = {
                'filename': filename,
                'basename': base_name,
                'total_area': total_area,
                'bay_area': bay_area,
                'total_length': total_length,
                'bay_length': bay_length,
                'extension_length': extension_length,
                'ratio': ratio,
                'centerline_svg': output_path,
                'success': True,
                'error': None
            }

            return result

        except Exception as e:
            if self.verbose:
                print(f"  Processing failed: {e}")

            result = {
                'filename': filename,
                'basename': base_name,
                'total_area': None,
                'bay_area': None,
                'total_length': None,
                'bay_length': None,
                'extension_length': None,
                'ratio': None,
                'centerline_svg': None,
                'success': False,
                'error': str(e)
            }

            return result

    def process_batch(self, file_extensions: list = None):
        """
        Batch-process all SVG files in a folder.

        Args:
            file_extensions: list of file extensions, default ['.svg']

        Returns:
            List: all processing results
        """
        if file_extensions is None:
            file_extensions = ['.svg']

        if self.verbose:
            print(f"{'=' * 60}")
            print(f"Polygon centerline batch extraction tool")
            print(f"{'=' * 60}")
            print(f"Input folder: {self.input_dir}")
            print(f"Output folder: {self.output_dir}")
            print(f"Resolution: {self.resolution}")
            print(f"Spline smoothing parameter: {self.spline_smoothness}")
            print(f"Chaikin iterations: {self.chaikin_iterations}")
            print(f"Include polygon background: {'yes' if self.include_background else 'no'}")
            print(f"{'=' * 60}")

        # Get all SVG files
        svg_files = []
        for ext in file_extensions:
            svg_files.extend(
                [f for f in os.listdir(self.input_dir) if f.lower().endswith(ext.lower())]
            )

        if self.verbose:
            print(f"Found {len(svg_files)} SVG file(s)")

        if len(svg_files) == 0:
            if self.verbose:
                print("No SVG files found")
            return []

        # Process each file
        self.results = []
        success_count = 0

        for svg_file in svg_files:
            svg_path = os.path.join(self.input_dir, svg_file)
            self.random_seed = self.random_seed_by_filename.get(
                svg_file, self.default_random_seed
            )
            result = self.process_single_svg(svg_path)

            self.results.append(result)

            if result['success']:
                success_count += 1

                if self.verbose:
                    print(f"\n[OK] {result['basename']}: processing succeeded")
                    print(f"   bay_length (L_bay) = {result['bay_length']:.2f}")
                    print(f"   bay_area (A_bay)   = {result['bay_area']:.2f}")
                    print(f"   L^2/A (effective)  = {result['ratio']:.2f}")
            else:
                if self.verbose:
                    print(f"\n[FAIL] {result['basename']}: processing failed - {result['error']}")

        # Save the summary table
        if success_count > 0:
            self.save_summary_table()

        if self.verbose:
            print(f"\n{'=' * 60}")
            print(f"Batch processing complete!")
            print(f"Successfully processed: {success_count}/{len(svg_files)} file(s)")
            print(f"{'=' * 60}")

        return self.results

    def save_summary_table(self):
        """Save the summary table to a CSV file."""
        # Extract successful results
        successful_results = [r for r in self.results if r['success']]

        if not successful_results:
            if self.verbose:
                print("No successful results; summary table not generated")
            return

        # Build the DataFrame
        data = []
        for result in successful_results:
            data.append({
                'filename': result['basename'],
                'bay_length': result['bay_length'],
                'bay_area': result['bay_area'],
                'total_length': result['total_length'],
                'total_area': result['total_area'],
                'extension_length': result['extension_length'],
                'shape_index': result['ratio'],
                'centerline_file': os.path.basename(result['centerline_svg'])
            })

        df = pd.DataFrame(data)

        # Save to CSV
        csv_path = os.path.join(self.output_dir, "centerline_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        if self.verbose:
            print(f"Summary table saved: {csv_path}")

            # Print the table
            print("\n" + "=" * 80)
            print("Centerline extraction results summary")
            print("=" * 80)
            print(df.to_string(index=False))
            print("=" * 80)


# =========================
# Entry function
# =========================
def extract_centerlines_batch(
        input_folder: str,
        output_folder: str,
        resolution: int = 5000,
        smoothness: float = 0.01,
        gauss_sigma: float = 1.0,
        distance_threshold: float = 2.0,
        spline_smoothness: float = 0.01,
        chaikin_iterations: int = 2,
        verbose: bool = True,
        include_background: bool = True,
        random_seed: int = None,
        random_seed_by_filename: dict = None
):
    """
    Main entry function for batch centerline extraction.

    Args:
        input_folder: input folder containing SVG files
        output_folder: output folder for centerline files
        resolution: resolution (pixels), default 5000
        smoothness: boundary simplification tolerance
        gauss_sigma: Gaussian smoothing parameter
        distance_threshold: skeleton pruning threshold
        spline_smoothness: spline smoothing parameter
        chaikin_iterations: number of Chaikin smoothing iterations
        verbose: whether to print detailed information
        include_background: whether to include the original polygon background in the output
        random_seed: optional medial_axis seed when no per-file seed is supplied;
            use None to leave the tie-breaking unfixed
        random_seed_by_filename: optional mapping from SVG filename to seed

    Returns:
        CenterlineExtractor: the extractor instance
    """
    # Create the extractor instance
    extractor = CenterlineExtractor(
        input_dir=input_folder,
        output_dir=output_folder,
        resolution=resolution,
        smoothness=smoothness,
        gauss_sigma=gauss_sigma,
        distance_threshold=distance_threshold,
        spline_smoothness=spline_smoothness,
        chaikin_iterations=chaikin_iterations,
        verbose=verbose,
        include_background=include_background,
        random_seed=random_seed,
        random_seed_by_filename=random_seed_by_filename,
    )

    # Batch processing
    extractor.process_batch()

    return extractor
