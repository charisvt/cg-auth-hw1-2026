import numpy as np


def vector_interp(p1, p2, V1, V2, coord, dim):
    """
    Linearly interpolates vector V at a point `p` on the line segment from p1 to p2.
    V can be n-dimensional.

    Parameters:
        p1 (tuple): Coordinates (x1, y1) of the first point.
        p2 (tuple): Coordinates (x2, y2) of the second point.
        V1 (array-like): Vector value at p1.
        V2 (array-like): Vector value at p2.
        coord (float): The x or y coordinate of the point `p` to interpolate at.
        dim (int): 1 for x-coordinate, 2 for y-coordinate.

    Returns:
        numpy.ndarray: Interpolated vector value at point `p`.
    """
    c1 = p1[dim - 1]
    c2 = p2[dim - 1]

    if c1 == c2:
        return np.array(V1)

    t = (coord - c1) / (c2 - c1)
    return np.array(V1) + t * (np.array(V2) - np.array(V1))


def vector_mean(v1, v2, v3):
    """
    Calculates the mean of three vectors.
    """
    return (v1 + v2 + v3) / 3


def f_shading(img, vertices, vcolors):
    """
    Flat shading: fills a triangle with the average color of its three vertices.
    Uses a scanline algorithm to rasterize the triangle.

    Parameters:
        img (ndarray): Canvas of shape (M, N, 3) with existing triangles.
        vertices (ndarray): Integer array of shape (3, 2) with vertex 2D coordinates.
        vcolors (ndarray): Array of shape (3, 3) with per-vertex RGB colors in [0, 1].

    Returns:
        ndarray: Updated canvas with the filled triangle.
    """
    updated_img = img.copy()
    height, width = img.shape[:2]
    triangle = vertices.tolist()
    flat_color = vector_mean(vcolors[0], vcolors[1], vcolors[2])

    ymin = max(int(np.min(vertices[:, 1])), 0)
    ymax = min(int(np.max(vertices[:, 1])) + 1, height)

    for y in range(ymin, ymax):
        intersections = []
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i + 1) % 3]
            y1, y2 = p1[1], p2[1]

            if y1 == y2:
                continue  # Skip horizontal edges

            if (y >= y1 and y < y2) or (y >= y2 and y < y1):
                x = p1[0] + (y - y1) * (p2[0] - p1[0]) / (y2 - y1)
                intersections.append(x)

        if len(intersections) < 2:
            continue

        intersections.sort()
        x_start = max(int(np.ceil(intersections[0])), 0)
        x_end = min(int(np.floor(intersections[1])), width - 1)

        updated_img[y, x_start:x_end + 1] = flat_color

    return updated_img


def g_shading(img, vertices, vcolors):
    """
    Gouraud shading: fills a triangle by linearly interpolating vertex colors.
    Uses a two-phase scanline algorithm:
      Phase 1: For each scanline y, find intersection points A and B on triangle
               edges and interpolate colors at those intersection points.
      Phase 2: For each pixel P=(x, y) on the scanline, interpolate the color
               between the colors at A and B.

    Parameters:
        img (ndarray): Canvas of shape (M, N, 3) with existing triangles.
        vertices (ndarray): Integer array of shape (3, 2) with vertex 2D coordinates.
        vcolors (ndarray): Array of shape (3, 3) with per-vertex RGB colors in [0, 1].

    Returns:
        ndarray: Updated canvas with the Gouraud-shaded triangle.
    """
    img = img.copy()
    M, N, _ = img.shape

    v0, v1, v2 = vertices
    c0, c1, c2 = vcolors[0], vcolors[1], vcolors[2]

    ymin = max(int(np.floor(min(v0[1], v1[1], v2[1]))), 0)
    ymax = min(int(np.ceil(max(v0[1], v1[1], v2[1]))), M - 1)

    # Each edge: (endpoint1, endpoint2, color_at_endpoint1, color_at_endpoint2)
    edges = [(v0, v1, c0, c1), (v1, v2, c1, c2), (v2, v0, c2, c0)]

    for y in range(ymin, ymax + 1):
        # Phase 1: Find intersection points and interpolate colors at edges
        intersections = []
        colors_at = []

        for p1, p2, col1, col2 in edges:
            if p1[1] == p2[1]:
                continue  # Skip horizontal edges

            if (y >= min(p1[1], p2[1])) and (y <= max(p1[1], p2[1])):
                x = vector_interp(p1, p2, p1[0], p2[0], y, 2)
                c_interp = vector_interp(p1, p2, col1, col2, y, 2)
                intersections.append(x)
                colors_at.append(c_interp)

        if len(intersections) < 2:
            continue

        # Sort by x coordinate and pick leftmost/rightmost (handles vertex-on-scanline cases)
        order = np.argsort(intersections)
        x_left = intersections[order[0]]
        x_right = intersections[order[-1]]
        c_left = colors_at[order[0]]
        c_right = colors_at[order[-1]]

        x_start = max(int(np.ceil(x_left)), 0)
        x_end = min(int(np.floor(x_right)), N - 1)

        if x_start > x_end:
            continue

        # Phase 2: Per-pixel color interpolation using vector_interp
        for x in range(x_start, x_end + 1):
            color = vector_interp((x_left, y), (x_right, y), c_left, c_right, x, 1)
            img[y, x] = np.clip(color, 0, 1)

    return img


def t_shading(img, vertices, uv, textImg):
    """
    Texture shading: fills a triangle by mapping a texture image using UV coordinates.
    Uses a two-phase scanline algorithm:
      Phase 1: For each scanline y, find edge intersection points A and B and
               interpolate their UV coordinates.
      Phase 2: For each pixel P=(x, y), interpolate UV between A and B, then
               sample the texture using nearest-neighbor filtering.

    Parameters:
        img (ndarray): Canvas of shape (M, N, 3) with existing triangles.
        vertices (ndarray): Integer array of shape (3, 2) with vertex 2D coordinates.
        uv (ndarray): Array of shape (3, 2) with normalized UV coords in [0, 1].
        textImg (ndarray): Texture image of shape (K, L, 3) in [0, 1].

    Returns:
        ndarray: Updated canvas with the texture-mapped triangle.
    """
    img = img.copy()
    M, N, _ = img.shape
    K, L, _ = textImg.shape

    v0, v1, v2 = vertices
    uv0, uv1, uv2 = uv

    ymin = max(int(np.floor(min(v0[1], v1[1], v2[1]))), 0)
    ymax = min(int(np.ceil(max(v0[1], v1[1], v2[1]))), M - 1)

    # Each edge: (endpoint1, endpoint2, uv_at_endpoint1, uv_at_endpoint2)
    edges = [(v0, v1, uv0, uv1), (v1, v2, uv1, uv2), (v2, v0, uv2, uv0)]

    for y in range(ymin, ymax + 1):
        # Phase 1: Find intersection points and interpolate UV at edges
        intersections = []
        uvs_at = []

        for p1, p2, uvA, uvB in edges:
            if p1[1] == p2[1]:
                continue  # Skip horizontal edges

            if (y >= min(p1[1], p2[1])) and (y <= max(p1[1], p2[1])):
                x = vector_interp(p1, p2, p1[0], p2[0], y, 2)
                uv_interp = vector_interp(p1, p2, uvA, uvB, y, 2)
                intersections.append(x)
                uvs_at.append(uv_interp)

        if len(intersections) < 2:
            continue

        # Sort by x coordinate and pick leftmost/rightmost (handles vertex-on-scanline cases)
        order = np.argsort(intersections)
        x_left = intersections[order[0]]
        x_right = intersections[order[-1]]
        uv_left = uvs_at[order[0]]
        uv_right = uvs_at[order[-1]]

        x_start = max(int(np.ceil(x_left)), 0)
        x_end = min(int(np.floor(x_right)), N - 1)

        if x_start > x_end:
            continue

        # Phase 2: Per-pixel UV interpolation using vector_interp, then nearest-neighbor sample
        for x in range(x_start, x_end + 1):
            uv_p = vector_interp((x_left, y), (x_right, y), uv_left, uv_right, x, 1)
            tx = int(np.clip(np.round(uv_p[0] * (L - 1)), 0, L - 1))
            ty = int(np.clip(np.round(uv_p[1] * (K - 1)), 0, K - 1))
            img[y, x] = textImg[ty, tx]

    return img


def render_img(faces, vertices, vcolors, uvs, depth, shading, texImg=None):
    """
    Renders a 3D object projected onto a 2D canvas by shading its triangular faces.

    Process:
      1. Initialize a white canvas of size 512x512.
      2. Sort faces by average vertex depth (farthest to closest, painter's algorithm).
      3. For each face, apply the chosen shading routine.
      4. Clip and convert the result to 8-bit RGB.

    Parameters:
        faces (ndarray): Triangle indices, shape (K, 3).
        vertices (ndarray): 2D vertex positions, shape (L, 2).
        vcolors (ndarray): Per-vertex RGB colors in [0, 1], shape (L, 3).
        uvs (ndarray or None): Per-vertex UV coordinates in [0, 1], shape (L, 2).
        depth (ndarray): Per-vertex depth values, shape (L,) or (L, 1).
        shading (str): 'f' for flat, 'g' for Gouraud, 't' for texture.
        texImg (ndarray or None): Texture image of shape (K, L, 3) in [0, 1].

    Returns:
        ndarray: Rendered image of shape (512, 512, 3), dtype uint8.
    """
    M = 512
    N = 512
    img = np.ones((M, N, 3), dtype=np.float32)

    depth_flat = depth.flatten()

    # Sort faces back-to-front by average vertex depth (painter's algorithm)
    avg_depth = np.mean(depth_flat[faces], axis=1)
    sorted_indices = np.argsort(avg_depth)[::-1]

    for idx in sorted_indices:
        face = faces[idx]
        tri_vertices = vertices[face]
        tri_colors = vcolors[face]

        if shading == 'f':
            img = f_shading(img, tri_vertices, tri_colors)
        elif shading == 'g':
            img = g_shading(img, tri_vertices, tri_colors)
        elif shading == 't':
            tri_uvs = uvs[face]
            img = t_shading(img, tri_vertices, tri_uvs, texImg)
        else:
            raise ValueError(f"Unknown shading mode '{shading}'. Use 'f', 'g', or 't'.")

    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)
