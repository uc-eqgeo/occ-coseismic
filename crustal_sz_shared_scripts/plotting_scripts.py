import numpy as np
import geopandas as gpd
from collections import deque
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point, Polygon
from shapely import union_all, unary_union
import triangle as tr
import matplotlib.tri as mtri

def constrained_triangulation_grid(
    sample_points,
    fault_lines,
    bounds=None,
    epsilon=None,
):
    """
    sample_points : (N, 2) array
    fault_lines   : list of (M, 2) arrays -- fault polylines, need NOT span
                    the full domain
    bounds        : (xmin, ymin, xmax, ymax); inferred from data if None
    epsilon       : half-width of the excluded slot around each fault.
                    Default: 0.15x the median nearest-neighbor sample
                    spacing -- small enough not to eat real coverage, large
                    enough to guarantee separation.

    Returns
    -------
    grid_x, grid_y, grid_z : 2D arrays (grid_z is NaN outside the mesh,
                              including inside the fault slots themselves)
    triang : the matplotlib Triangulation actually used (handy for QC plots)
    """
    sample_points = np.asarray(sample_points, dtype=float)

    if epsilon is None:
        tree = cKDTree(sample_points)
        d, _ = tree.query(sample_points, k=2)
        epsilon = np.median(d[:, 1]) * 0.15

    slot_polys = build_fault_slot_regions(fault_lines, epsilon)

    if bounds is None:
        pad_x = (sample_points[:, 0].max() - sample_points[:, 0].min()) * 0.02
        pad_y = (sample_points[:, 1].max() - sample_points[:, 1].min()) * 0.02
        bounds = (
            sample_points[:, 0].min() - pad_x,
            sample_points[:, 1].min() - pad_y,
            sample_points[:, 0].max() + pad_x,
            sample_points[:, 1].max() + pad_y,
        )

    # --- Build the PSLG (planar straight line graph) for `triangle` ---
    # IMPORTANT: with the 'p' switch, Triangle only fills the region enclosed
    # by segments -- so the outer domain boundary must be a segment loop too,
    # or Triangle will only mesh the tiny fault-slot rectangles and nothing
    # else.
    vertices = [tuple(v) for v in sample_points]
    vertex_index = {v: i for i, v in enumerate(vertices)}
    segments = []
    holes = []

    def add_vertex(v):
        v = tuple(v)
        if v not in vertex_index:
            vertex_index[v] = len(vertices)
            vertices.append(v)
        return vertex_index[v]

    xmin, ymin, xmax, ymax = bounds
    corners = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
    corner_idxs = [add_vertex(c) for c in corners]
    segments += [
        (corner_idxs[i], corner_idxs[(i + 1) % 4]) for i in range(4)
    ]

    # vertex_role tracks, for each PSLG vertex added from a slot polygon's
    # boundary, whether it came from an EXTERIOR ring (general "outside" the
    # fault band) or an INTERIOR ring (the boundary of a closed loop's
    # fillable interior) -- and, for the latter, which loop.
    vertex_role = {}         # idx -> ('exterior', None) | ('loop', loop_id)
    loop_interior_polys = []  # loop_id -> shapely Polygon of that loop's interior

    for poly in slot_polys:
        # Exterior ring: separates "outside everything" from the excluded
        # fault band.
        coords = list(poly.exterior.coords)[:-1]
        idxs = [add_vertex(v) for v in coords]
        n = len(idxs)
        segments += [(idxs[i], idxs[(i + 1) % n]) for i in range(n)]
        for idx in idxs:
            vertex_role.setdefault(idx, ("exterior", None))

        # Interior rings (holes): each one bounds a closed loop's fillable
        # interior. Skipping these is exactly what causes loop interiors to
        # get silently swallowed into the excluded region instead of being
        # triangulated.
        for interior in poly.interiors:
            coords = list(interior.coords)[:-1]
            idxs = [add_vertex(v) for v in coords]
            n = len(idxs)
            segments += [(idxs[i], idxs[(i + 1) % n]) for i in range(n)]
            loop_id = len(loop_interior_polys)
            loop_interior_polys.append(Polygon(interior))
            for idx in idxs:
                vertex_role[idx] = ("loop", loop_id)

        # Only mark the band material itself as excluded. representative_point()
        # is guaranteed to land in the polygon's actual area, i.e. in the band,
        # never inside one of its holes -- so loop interiors stay unmarked and
        # get triangulated normally.
        holes.append(poly.representative_point().coords[0])

    pslg = dict(vertices=np.array(vertices), segments=np.array(segments))
    if holes:
        pslg["holes"] = np.array(holes)

    # 'p' = treat input as a PSLG and respect all segments as forced edges
    mesh = tr.triangulate(pslg, "p")
    mesh_vertices = mesh["vertices"]
    mesh_triangles = mesh["triangles"]

    # --- Assign values to every mesh vertex ---
    n_samples = len(sample_points)

    # Precompute, for every REAL sample, which loop (if any) contains it.
    # -1 means "not inside any loop" (the general outside/exterior region).
    # This is a well-defined geometric test and doesn't depend on which
    # fault segment happens to be nearest -- unlike a local side test, it
    # can't misfire right at a loop's corners.
    sample_loop_tags = np.full(n_samples, -1, dtype=int)
    for loop_id, loop_poly in enumerate(loop_interior_polys):
        inside = np.array([loop_poly.contains(Point(p)) for p in sample_points])
        sample_loop_tags[inside] = loop_id


    # Compute how many steps a point is away from a sample point
    vertex_steps = compute_vertex_steps(mesh_triangles, mesh_vertices.shape[0], sample_points.shape[0])
    # Order vertices for interpolation, starting with nearest to sample points and expand out for each step
    ordered_vertices, stats = [], {}
    sample_ix = list(np.arange(n_samples))

    for step in range(1, vertex_steps.max()):
        step_ordered_vertices, step_stats = rank_vertices_by_confidence(mesh_triangles, mesh_vertices.shape[0], sample_ix, np.where(vertex_steps == step)[0].tolist(), vertex_steps)
        ordered_vertices += step_ordered_vertices
        sample_ix += step_ordered_vertices
        stats = stats | step_stats

    # Add sites to the nearest dictionary in order of proximity to 
    nearest = {}    
    for i in ordered_vertices:
        v = mesh_vertices[i]
        verts = np.unique(mesh_triangles[np.where(mesh_triangles == i)[0]]).tolist()
        if len(verts) > 0:
            verts.remove(i)
            d = np.linalg.norm(mesh_vertices[verts] - v, axis=1)
            nearest[i] = {'sites': [verts[d] for d in np.argsort(d)[:]], 'dists': d[np.argsort(d)[:]], 'weights': 1 / (d[np.argsort(d)[:]] * 1e-3) ** 0.5}

    # --- Interpolate on the constrained, fault-slotted mesh ---
    triang = mtri.Triangulation(
        mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_triangles
    )

    return triang, nearest


def build_fault_slot_regions(fault_lines, epsilon):
    """
    Buffer every fault line and UNION them together into one or more
    'excluded material' polygons.

    Doing the union (rather than treating each fault independently) matters
    when several fault traces join up to enclose an area: the merged shape
    then has a genuine interior ring (a hole) around the loop's interior.
    That interior is real, fillable ground -- it should be triangulated
    normally with whatever sample points fall inside it -- so we must NOT
    treat it as excluded. Only the thin band of buffered fault material
    itself (the polygon's own area, excluding any holes) gets excluded.
    """
    slivers = [
        LineString(coords).buffer(epsilon, cap_style=1, quad_segs=2) for coords in fault_lines
    ]
    merged = unary_union(slivers)
    polys = list(merged.geoms) if hasattr(merged, "geoms") else [merged]
    return polys


def compute_vertex_steps(mesh_triangles, n_vertices, n_samples):
    """
    Multi-source BFS over the triangulation's edge graph, starting from all
    real sample vertices (by construction these are vertex indices
    0..n_samples-1 -- see constrained_triangulation_grid). Returns, for
    every mesh vertex, the number of triangulation edges ("steps") to the
    nearest sample vertex. Sample vertices themselves get 0.

    Vertices in a triangulated pocket that contains no sample at all (e.g.
    a fault loop enclosing zero samples) can never be reached and are left
    as -1.
    """
    adjacency = [[] for _ in range(n_vertices)]
    for a, b, c in mesh_triangles:
        adjacency[a] += [b, c]
        adjacency[b] += [a, c]
        adjacency[c] += [a, b]

    steps = np.full(n_vertices, -1, dtype=int)
    q = deque()
    for i in range(n_samples):
        steps[i] = 0
        q.append(i)

    while q:
        u = q.popleft()
        for v in adjacency[u]:
            if steps[v] == -1:
                steps[v] = steps[u] + 1
                q.append(v)

    return steps


def rank_vertices_by_confidence(mesh_triangles, n_vertices, sample_ix, candidates, vertex_steps):
    """
    Order every non-sample vertex from most- to least-trustworthy, using
    three keys in priority order:

      1. Fewest non-sample (interpolated) direct neighbours -- primary.
      2. Highest proportion of direct neighbours that ARE real samples --
         this differs from (1) once vertices have different total degree:
         e.g. 4 non-sample neighbours out of 5 is worse than 4 out of 20,
         even though both have "4" as a raw count.
      3. Fewest triangulation hops to the nearest real sample (the
         `vertex_steps` BFS result) -- used only to break remaining ties.

    Returns
    -------
    ordered_vertices : list of vertex indices (non-samples only), best to
                        worst
    stats : dict {vertex_index: (non_sample_count, sample_proportion, degree)}
    """

    adjacency = [set() for _ in range(n_vertices)]
    for a, b, c in mesh_triangles:
        adjacency[a].update((b, c))
        adjacency[b].update((a, c))
        adjacency[c].update((a, b))

    stats = {}
    for v in candidates:
        neighbours = adjacency[v]
        degree = len(neighbours)
        non_sample_count = len(set(neighbours) - set(sample_ix)) if degree > 0 else np.inf
        sample_count = degree - non_sample_count
        sample_proportion = sample_count / degree if degree > 0 else 0.0
        stats[v] = (non_sample_count, sample_proportion, degree)

    candidates.sort(
        key=lambda v: (
            stats[v][0],           # 1) minimum non-sample neighbor count
            -stats[v][1],          # 2) maximum sample-neighbor proportion
            vertex_steps[v],       # 3) minimum step count
        )
    )
    return candidates, stats


def triangulation_to_gdf(triang, z=None, crs=None):
    """
    Convert a matplotlib.tri.Triangulation into a GeoDataFrame of triangle
    polygons (one row per triangle), respecting triang.mask if it's set.
 
    z : optional (N,) array of per-vertex values -- e.g. the same `z` you
        fed to LinearTriInterpolator/CubicTriInterpolator. If given, each
        triangle's mean vertex value is stored in a 'mean_z' column.
    crs : optional CRS to assign to the output (e.g. faults_gdf.crs).
    """
    x = triang.x
    y = triang.y
    triangles = triang.triangles
    mask = (
        triang.mask
        if triang.mask is not None
        else np.zeros(len(triangles), dtype=bool)
    )
 
    tri_ids, polys, mean_zs = [], [], []
    for i, (tri, masked) in enumerate(zip(triangles, mask)):
        if masked:
            continue
        coords = [(x[v], y[v]) for v in tri]
        polys.append(Polygon(coords))
        tri_ids.append(i)
        if z is not None:
            mean_zs.append(float(np.mean([z[v] for v in tri])))
 
    data = {"triangle_id": tri_ids}
    if z is not None:
        data["mean_z"] = mean_zs
 
    return gpd.GeoDataFrame(data, geometry=polys, crs=crs)
 
 
def save_triangulation(triang, path, z=None, crs=None, driver=None):
    """
    Write a triangulation to disk as GeoJSON or Shapefile.
 
    path   : output path, e.g. 'triangles.geojson' or 'triangles.shp'
    z      : optional per-vertex values, see triangulation_to_gdf
    crs    : optional CRS to assign
    driver : override auto-detection; inferred from the file extension
             ('.geojson'/'.json' -> 'GeoJSON', '.shp' -> 'ESRI Shapefile')
             if not given.
 
    Returns the GeoDataFrame that was written (handy for further use/QC).
    """
    gdf = triangulation_to_gdf(triang, z=z, crs=crs)
 
    if driver is None:
        suffix = str(path).lower()
        if suffix.endswith(".geojson") or suffix.endswith(".json"):
            driver = "GeoJSON"
        elif suffix.endswith(".shp"):
            driver = "ESRI Shapefile"
 
    gdf.to_file(path, driver=driver)
    return gdf